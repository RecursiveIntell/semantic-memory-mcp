//! HTTP search server for semantic-memory-mcp.
//!
//! A minimal HTTP server that exposes the most-used semantic-memory
//! operations over a local TCP port. Runs alongside the stdio MCP
//! transport so the same warm process serves both MCP clients and
//! HTTP clients (hooks, benchmarks, scripts).
//!
//! Endpoints:
//!   POST /search   {"query": "...", "top_k": 10} -> search results
//!   POST /stats    {} -> DB stats
//!   POST /add      {"content": "...", "namespace": "...", "source": "..."} -> authority receipt
//!   GET  /health   -> {"ok": true}

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::time::Duration;
use tokio::runtime::Handle;
use tokio::task::block_in_place;

use crate::bridge::MemoryBridge;

const MAX_TOP_K: u64 = 100;
const MAX_DIRECT_IDS: usize = 100;
const MAX_RERANK_RESULTS: usize = 50;
const MAX_RERANK_CONTENT_BYTES: usize = 2_000;
const RERANK_MODEL: &str = "granite4.1:3b";

fn truncate_rerank_content(content: &str) -> &str {
    if content.len() <= MAX_RERANK_CONTENT_BYTES {
        return content;
    }
    let mut end = MAX_RERANK_CONTENT_BYTES;
    while !content.is_char_boundary(end) {
        end -= 1;
    }
    &content[..end]
}

fn rerank_candidate_limit(top_k: usize) -> usize {
    top_k.saturating_mul(2).min(MAX_RERANK_RESULTS)
}

/// Call Ollama to rate each result's relevance to the query (1-5) and sort descending.
/// Returns a new vec with a `rerank_score` field added to each result object.
fn rerank_results(
    query: &str,
    results: &[serde_json::Value],
    model: &str,
) -> (Vec<serde_json::Value>, &'static str) {
    rerank_results_at(query, results, model, "http://127.0.0.1:11434")
}

fn rerank_results_at(
    query: &str,
    results: &[serde_json::Value],
    model: &str,
    base_url: &str,
) -> (Vec<serde_json::Value>, &'static str) {
    let client = match reqwest::blocking::Client::builder()
        .connect_timeout(Duration::from_secs(1))
        .timeout(Duration::from_secs(3))
        .build()
    {
        Ok(client) => client,
        Err(_) => return (results.to_vec(), "degraded"),
    };
    let scored: Result<Vec<(f64, serde_json::Value)>, ()> = results
        .iter()
        .map(|r| -> Result<(f64, serde_json::Value), ()> {
            let content = r.get("content").and_then(|v| v.as_str()).unwrap_or("");
            let truncated = truncate_rerank_content(content);
            let prompt = format!(
                "Rate the relevance of this document to the query on a scale of 1-5. Reply with ONLY the number.\nQuery: {query}\nDocument: {truncated}\nRating:"
            );
            let body = serde_json::json!({
                "model": model,
                "prompt": prompt,
                "stream": false,
                "options": {"temperature": 0, "num_predict": 1}
            });
            let response = client
                .post(format!("{base_url}/api/generate"))
                .json(&body)
                .send()
                .map_err(|_| ())?
                .error_for_status()
                .map_err(|_| ())?
                .json::<serde_json::Value>()
                .map_err(|_| ())?;
            let rating = response
                .get("response")
                .and_then(|r| r.as_str())
                .and_then(|s| s.trim().chars().next())
                .and_then(|c| c.to_digit(10))
                .map(|d| d as f64)
                /* parsed values must be an actual 1–5 rating */
                .filter(|rating| (1.0..=5.0).contains(rating))
                .ok_or(())?;
            Ok((rating, r.clone()))
        })
        .collect::<Result<_, _>>();
    let Ok(mut scored) = scored else {
        return (results.to_vec(), "degraded");
    };
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    (
        scored
            .into_iter()
            .map(|(score, mut r)| {
                if let Some(obj) = r.as_object_mut() {
                    obj.insert("rerank_score".to_string(), serde_json::json!(score));
                }
                r
            })
            .collect::<Vec<_>>(),
        "applied",
    )
}

pub struct HttpServerHandle {
    pub local_addr: std::net::SocketAddr,
    _thread: std::thread::JoinHandle<()>,
}

struct ConnectionSlot {
    active: Arc<AtomicUsize>,
}

impl ConnectionSlot {
    fn new(active: Arc<AtomicUsize>) -> Self {
        Self { active }
    }
}

impl Drop for ConnectionSlot {
    fn drop(&mut self) {
        self.active.fetch_sub(1, Ordering::AcqRel);
    }
}

pub fn start_http_server(
    port: u16,
    auth_token: &str,
    bridge: MemoryBridge,
    handle: Handle,
    profile: crate::profile::ToolProfile,
) -> std::io::Result<HttpServerHandle> {
    let listener = TcpListener::bind(("127.0.0.1", port))?;
    let local_addr = listener.local_addr()?;
    let auth_token = auth_token.to_string();
    let active = Arc::new(AtomicUsize::new(0));
    let thread = std::thread::spawn(move || {
        eprintln!("HTTP search server listening on {local_addr}");
        for stream in listener.incoming() {
            let stream = match stream {
                Ok(s) => s,
                Err(_) => continue,
            };

            if active.fetch_add(1, Ordering::AcqRel) >= 32 {
                active.fetch_sub(1, Ordering::AcqRel);
                let _ = stream.shutdown(std::net::Shutdown::Both);
                continue;
            }
            let bridge = bridge.clone();
            let h = handle.clone();
            let token = auth_token.clone();
            let active = active.clone();
            std::thread::spawn(move || {
                let _slot = ConnectionSlot::new(active);
                handle_connection(stream, &token, bridge, h, profile);
            });
        }
    });
    Ok(HttpServerHandle {
        local_addr,
        _thread: thread,
    })
}

fn handle_connection(
    mut stream: TcpStream,
    auth_token: &str,
    bridge: MemoryBridge,
    handle: Handle,
    profile: crate::profile::ToolProfile,
) {
    const MAX_HEADER_BYTES: usize = 16 * 1024;
    const MAX_HEADER_COUNT: usize = 64;
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(2)));
    let Ok(reader_stream) = stream.try_clone() else {
        return;
    };
    let mut reader = BufReader::new(reader_stream);
    let mut request_line = String::new();
    if !read_bounded_line(&mut reader, 4096, &mut request_line) {
        return;
    }

    let parts: Vec<&str> = request_line.split_whitespace().collect();
    if parts.len() < 2 {
        return;
    }
    let method = parts[0];
    let path = parts[1];

    let mut content_length = 0;
    let mut auth_header: Option<String> = None;
    let mut host_header: Option<String> = None;
    let mut origin_header: Option<String> = None;
    let mut header_bytes = request_line.len();
    let mut header_count = 0usize;
    loop {
        let mut header = String::new();
        let remaining = MAX_HEADER_BYTES.saturating_sub(header_bytes);
        if !read_bounded_line(&mut reader, remaining, &mut header) {
            return;
        }
        if header.trim().is_empty() {
            break;
        }
        header_count += 1;
        header_bytes += header.len();
        if header_count > MAX_HEADER_COUNT || header_bytes > MAX_HEADER_BYTES {
            let _ = stream.write_all(b"HTTP/1.1 431 Request Header Fields Too Large\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
            return;
        }
        let Some((name, value)) = header.split_once(':') else {
            return;
        };
        match name.to_ascii_lowercase().as_str() {
            "content-length" => {
                content_length = match value.trim().parse() {
                    Ok(v) => v,
                    Err(_) => return,
                }
            }
            "authorization" => auth_header = Some(value.trim().to_string()),
            "host" => host_header = Some(value.trim().to_string()),
            "origin" => origin_header = Some(value.trim().to_string()),
            _ => {}
        }
    }

    // Auth check
    let authorized = auth_header
        .as_deref()
        .map(|h| h == format!("Bearer {}", auth_token))
        .unwrap_or(false);
    if !authorized {
        let response =
            "HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
        let _ = stream.write_all(response.as_bytes());
        return;
    }

    // Host check
    let host_ok = host_header.as_deref().is_some_and(is_loopback_authority);
    let origin_ok = origin_header.as_deref().map_or(true, is_loopback_origin);
    if !host_ok || !origin_ok {
        let response = "HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
        let _ = stream.write_all(response.as_bytes());
        return;
    }

    // Content-Length cap: 10MB
    const MAX_BODY_SIZE: usize = 10 * 1024 * 1024;
    if content_length > MAX_BODY_SIZE {
        let response =
            "HTTP/1.1 413 Payload Too Large\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
        let _ = stream.write_all(response.as_bytes());
        return;
    }

    let mut body = vec![0u8; content_length];
    if content_length > 0 && reader.read_exact(&mut body).is_err() {
        return;
    }
    let body_str = String::from_utf8_lossy(&body);

    let (status, response) = match (method, path) {
        ("GET", "/health") => (
            "200 OK",
            serde_json::json!({"ok": true, "service": "semantic-memory-mcp"}),
        ),
        (_, _) if !profile.allows_http_route() => (
            "404 Not Found",
            serde_json::json!({"error": "not found", "path": path}),
        ),
        ("POST", "/search") => handle_search(&body_str, &bridge, &handle),
        ("POST", "/search-routed") => handle_search_routed(&body_str, &bridge, &handle),
        ("POST", "/rerank") => handle_rerank(&body_str),
        ("POST", "/stats") => handle_stats(&bridge, &handle),
        ("POST", "/add") if profile.allows_http_write() => {
            handle_add_fact(&body_str, &bridge, &handle)
        }
        ("POST", "/record-outcome") if profile.allows_http_write() => {
            handle_record_outcome(&body_str, &bridge, &handle)
        }
        ("GET", "/verify-integrity") => handle_verify_integrity(&bridge, &handle),
        ("POST", "/discord") => handle_discord(&body_str, &bridge, &handle),
        ("POST", "/maintenance/check") if profile.allows_http_maintenance() => {
            handle_maintenance_check(&bridge, &handle)
        }
        ("POST", "/maintenance/vacuum") if profile.allows_http_maintenance() => {
            handle_maintenance_vacuum(&bridge, &handle)
        }
        ("POST", "/maintenance/reembed") if profile.allows_http_maintenance() => {
            handle_maintenance_reembed(&bridge, &handle)
        }
        ("POST", "/maintenance/reconcile") if profile.allows_http_maintenance() => {
            handle_maintenance_reconcile(&body_str, &bridge, &handle)
        }
        ("POST", "/maintenance/rebuild-hnsw") if profile.allows_http_maintenance() => {
            handle_maintenance_rebuild_hnsw(&bridge, &handle)
        }
        ("POST", "/maintenance/compact-hnsw") if profile.allows_http_maintenance() => {
            handle_maintenance_compact_hnsw(&bridge, &handle)
        }
        _ => (
            "404 Not Found",
            serde_json::json!({"error": "not found", "path": path}),
        ),
    };

    let response_str = serde_json::to_string(&response).unwrap_or_default();
    let response_bytes = response_str.as_bytes();
    let http_response = format!(
        "HTTP/1.1 {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        status,
        response_bytes.len()
    );

    let _ = stream.write_all(http_response.as_bytes());
    let _ = stream.write_all(response_bytes);
    let _ = stream.flush();
}

fn read_bounded_line(reader: &mut BufReader<TcpStream>, max: usize, output: &mut String) -> bool {
    if max == 0 {
        return false;
    }
    let Ok(read) = reader.take((max + 1) as u64).read_line(output) else {
        return false;
    };
    read > 0 && read <= max && output.ends_with('\n')
}

fn is_loopback_authority(value: &str) -> bool {
    matches!(value, "localhost" | "127.0.0.1" | "[::1]")
        || value.strip_prefix("localhost:").is_some_and(valid_port)
        || value.strip_prefix("127.0.0.1:").is_some_and(valid_port)
        || value.strip_prefix("[::1]:").is_some_and(valid_port)
}

fn valid_port(value: &str) -> bool {
    value.parse::<u16>().is_ok()
}

fn is_loopback_origin(value: &str) -> bool {
    value
        .strip_prefix("http://")
        .is_some_and(is_loopback_authority)
        || value
            .strip_prefix("https://")
            .is_some_and(is_loopback_authority)
}

fn handle_search(
    body: &str,
    bridge: &MemoryBridge,
    handle: &Handle,
) -> (&'static str, serde_json::Value) {
    let params: serde_json::Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(e) => {
            return (
                "400 Bad Request",
                serde_json::json!({"ok": false, "error": format!("invalid JSON: {e}")}),
            )
        }
    };

    let query = params.get("query").and_then(|v| v.as_str()).unwrap_or("");
    let top_k = match bounded_top_k(&params, 5) {
        Ok(k) => k,
        Err(response) => return response,
    };
    let namespaces: Option<Vec<String>> = params
        .get("namespaces")
        .and_then(|v| serde_json::from_value(v.clone()).ok());
    let do_rerank = params
        .get("rerank")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    if query.is_empty() {
        return (
            "400 Bad Request",
            serde_json::json!({"ok": false, "error": "missing 'query' field"}),
        );
    }

    let store = &bridge.store;
    let ns_slice: Option<Vec<&str>> = namespaces
        .as_ref()
        .map(|v| v.iter().map(|s| s.as_str()).collect());
    // Fetch top_k * 2 candidates when reranking so the LLM has a richer pool to sort.
    let fetch_k = if do_rerank {
        rerank_candidate_limit(top_k)
    } else {
        top_k
    };
    let result = block_in_place(|| {
        handle.block_on(store.search(query, Some(fetch_k), ns_slice.as_deref(), None))
    });

    match result {
        Ok(results) => {
            let json_results: Vec<serde_json::Value> = results
                .iter()
                .map(|r| {
                    let namespace = match &r.source {
                        semantic_memory::SearchSource::Fact { namespace, .. } => namespace.clone(),
                        semantic_memory::SearchSource::Chunk { document_title, .. } => {
                            document_title.clone()
                        }
                        _ => String::new(),
                    };
                    serde_json::json!({
                        "result_id": r.source.result_id(),
                        "content": r.content,
                        "score": r.score,
                        "cosine_similarity": r.cosine_similarity,
                        "namespace": namespace,
                    })
                })
                .collect();

            let (final_results, rerank_status): (Vec<serde_json::Value>, &str) =
                if do_rerank && !json_results.is_empty() {
                    let (results, status) = rerank_results(query, &json_results, RERANK_MODEL);
                    (results.into_iter().take(top_k).collect(), status)
                } else {
                    (json_results, "not_applicable")
                };

            let count = final_results.len();
            let provenance = serde_json::json!({
                "stages_fired": {
                    "bm25": true,
                    "vector": true,
                    "late_interaction": false,
                    "rerank": rerank_status == "applied",
                },
                "rerank_requested": do_rerank,
                "result_count": count,
                "view": "semantic",
                "widening_occurred": null,
                "widening_reason": null,
                "verification_status": "unverified",
                "proof_reference": null,
            });
            (
                "200 OK",
                serde_json::json!({
                    "ok": true,
                    "query": query,
                    "top_k": top_k,
                    "results": final_results,
                    "count": count,
                    "reranked": rerank_status == "applied",
                    "rerank_status": rerank_status,
                    "provenance": provenance,
                }),
            )
        }
        Err(e) => (
            "500 Internal Server Error",
            serde_json::json!({"ok": false, "error": format!("search error: {e}")}),
        ),
    }
}

/// Handle /search-routed: routing-aware search with full pipeline.
///
/// Uses the library's routing system to profile the query and decide which
/// retrieval stages to activate. For class C/D queries with contradictions,
/// runs factor graph belief propagation and decoder syndrome detection.
/// When discord is enabled, runs second-order retrieval via graph neighborhood.
/// Optionally groups results by community.
fn handle_search_routed(
    body: &str,
    bridge: &MemoryBridge,
    handle: &Handle,
) -> (&'static str, serde_json::Value) {
    use semantic_memory::integration::plan_execution;
    use semantic_memory::rl_routing::{is_trained, route_with_policy};
    use semantic_memory::routing::{QueryProfile, RetrievalRouter};

    let params: serde_json::Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(e) => {
            return (
                "400 Bad Request",
                serde_json::json!({"ok": false, "error": format!("invalid JSON: {e}")}),
            )
        }
    };

    let query = params.get("query").and_then(|v| v.as_str()).unwrap_or("");
    let base_top_k = match bounded_top_k(&params, 12) {
        Ok(k) => k,
        Err(response) => return response,
    };
    let query_class = params
        .get("query_class")
        .and_then(|v| v.as_str())
        .unwrap_or("A");
    let namespaces: Option<Vec<String>> = params
        .get("namespaces")
        .and_then(|v| serde_json::from_value(v.clone()).ok());
    let contradictions: Vec<(String, String)> = params
        .get("contradictions")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default();
    let group_by_community = params
        .get("group_by_community")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    if query.is_empty() {
        return (
            "400 Bad Request",
            serde_json::json!({"ok": false, "error": "missing 'query' field"}),
        );
    }

    // Use the routing system to profile the query
    let router = RetrievalRouter {
        decoder_enabled: true,
        discord_enabled: true,
        corpus_density: 0.5,
        ..Default::default()
    };
    let store = &bridge.store;
    let policy = match block_in_place(|| handle.block_on(store.load_routing_policy())) {
        Ok(policy) => policy,
        Err(e) => {
            return (
                "500 Internal Server Error",
                serde_json::json!({"ok": false, "error": format!("load routing policy error: {e}")}),
            )
        }
    };
    let profile = QueryProfile::from_query(query);
    let (decision, routing_source) = match policy.as_ref().filter(|p| is_trained(p)) {
        Some(policy) => (route_with_policy(policy, &profile), "trained_policy"),
        None => (router.route(&profile), "heuristic"),
    };
    let contras = contradictions.clone();
    let plan = plan_execution(&decision, contras.clone());

    // Class D (SYNTHESIS): retrieve more candidates to support comprehensive answers
    let top_k = if query_class == "D" {
        (base_top_k * 2).min(20)
    } else {
        base_top_k
    };

    let ns_slice: Option<Vec<&str>> = namespaces
        .as_ref()
        .map(|v| v.iter().map(|s| s.as_str()).collect());

    // Class C (CONTRADICTION): use ExactSearch context for higher-fidelity results
    let result = if query_class == "C" {
        use semantic_memory::{ExactnessProfile, SearchContext};
        let mut ctx = SearchContext::default_now();
        ctx.exactness_profile = ExactnessProfile::PreferExact;
        block_in_place(|| {
            handle.block_on(store.search_with_context(
                query,
                Some(top_k),
                ns_slice.as_deref(),
                None,
                ctx,
            ))
        })
        .map(|r| r.results)
    } else {
        block_in_place(|| {
            handle.block_on(store.search(query, Some(top_k), ns_slice.as_deref(), None))
        })
    };

    match result {
        Ok(results) => {
            let json_results: Vec<serde_json::Value> = results
                .iter()
                .map(|r| {
                    let namespace = match &r.source {
                        semantic_memory::SearchSource::Fact { namespace, .. } => namespace.clone(),
                        semantic_memory::SearchSource::Chunk { document_title, .. } => {
                            document_title.clone()
                        }
                        _ => String::new(),
                    };
                    serde_json::json!({
                        "result_id": r.source.result_id(),
                        "content": r.content,
                        "score": r.score,
                        "cosine_similarity": r.cosine_similarity,
                        "namespace": namespace,
                        "source_type": match &r.source {
                            semantic_memory::SearchSource::Fact { .. } => "fact",
                            semantic_memory::SearchSource::Chunk { .. } => "chunk",
                            semantic_memory::SearchSource::Message { .. } => "message",
                            _ => "unknown",
                        },
                    })
                })
                .collect();

            let mut factor_graph_payload = serde_json::json!({"enabled": false});
            let mut decoder_executed = false;
            let mut discord_executed = false;
            let mut discord_results_payload: Vec<serde_json::Value> = Vec::new();

            // Factor graph belief propagation for class C/D with contradictions
            if decision.decoder {
                #[cfg(feature = "full")]
                {
                    use semantic_memory::factor_graph::{
                        factors_from_edges, FactorGraph, FactorGraphConfig,
                    };

                    let graph_edges =
                        block_in_place(|| handle.block_on(store.list_all_graph_edges()));

                    if let Ok(edges) = graph_edges {
                        let raw_edges: Vec<(
                            String,
                            String,
                            semantic_memory::GraphEdgeType,
                            f64,
                            Option<String>,
                        )> = edges
                            .iter()
                            .map(|edge| {
                                let parsed_type = edge
                                    .edge_type_parsed
                                    .clone()
                                    .or_else(|| serde_json::from_str(&edge.edge_type).ok())
                                    .unwrap_or(semantic_memory::GraphEdgeType::Entity {
                                        relation: "unknown".to_string(),
                                    });
                                (
                                    edge.source.clone(),
                                    edge.target.clone(),
                                    parsed_type,
                                    edge.weight,
                                    edge.metadata.clone(),
                                )
                            })
                            .collect();

                        let nodes: Vec<(String, f64)> = results
                            .iter()
                            .map(|r| (r.source.result_id(), r.score))
                            .collect();
                        let factors = factors_from_edges(&raw_edges);
                        let graph = FactorGraph::new(&nodes, factors, FactorGraphConfig::default());
                        let propagated = graph.propagate();
                        let top_beliefs = propagated.top_k(top_k);

                        factor_graph_payload = serde_json::json!({
                            "enabled": true,
                            "top_k_beliefs": top_beliefs
                                .into_iter()
                                .map(|(item_id, belief)| serde_json::json!({
                                    "item_id": item_id,
                                    "belief": belief,
                                }))
                                .collect::<Vec<_>>(),
                            "iterations": propagated.iterations,
                            "converged": propagated.converged,
                            "elapsed_ms": propagated.elapsed_ms,
                            "factor_counts": {
                                "semantic": propagated.factor_counts.semantic,
                                "temporal": propagated.factor_counts.temporal,
                                "causal": propagated.factor_counts.causal,
                                "entity": propagated.factor_counts.entity,
                                "total": propagated.factor_counts.total(),
                            },
                        });
                        decoder_executed = true;
                    }
                }

                // Decoder syndrome detection for contradictions
                if !plan.contradictions.is_empty() {
                    use semantic_memory::decoder::{compute_correction, detect_syndromes};
                    let result_scores: Vec<(String, f64)> = results
                        .iter()
                        .map(|r| (r.source.result_id(), r.score))
                        .collect();
                    let syndromes = detect_syndromes(&result_scores, &plan.contradictions);
                    let _ = compute_correction(&syndromes, 10.0);
                    decoder_executed = true;
                }
            }

            // Discord second-order retrieval
            if plan.use_discord {
                use semantic_memory::discord::DiscordScorer;
                let direct_ids: Vec<String> =
                    results.iter().map(|r| r.source.result_id()).collect();
                let existing_ids: std::collections::HashSet<String> =
                    direct_ids.iter().cloned().collect();
                let edges_result = block_in_place(|| {
                    handle.block_on(store.list_graph_edges_for_neighborhood(
                        direct_ids.clone(),
                        2,
                        200,
                    ))
                });
                if let Ok(raw_edges) = edges_result {
                    let edge_refs: Vec<semantic_memory::discord::GraphEdgeRef> = raw_edges
                        .iter()
                        .map(|edge| {
                            let parsed_type = edge
                                .edge_type_parsed
                                .clone()
                                .or_else(|| serde_json::from_str(&edge.edge_type).ok())
                                .unwrap_or(semantic_memory::GraphEdgeType::Entity {
                                    relation: "unknown".to_string(),
                                });
                            let type_str = match parsed_type {
                                semantic_memory::GraphEdgeType::Semantic { .. } => "semantic",
                                semantic_memory::GraphEdgeType::Temporal { .. } => "temporal",
                                semantic_memory::GraphEdgeType::Causal { .. } => "causal",
                                semantic_memory::GraphEdgeType::Entity { .. } => "entity",
                            };
                            semantic_memory::discord::GraphEdgeRef {
                                source: edge.source.clone(),
                                target: edge.target.clone(),
                                edge_type: type_str.to_string(),
                                weight: edge.weight,
                            }
                        })
                        .collect();
                    let scorer = DiscordScorer::with_defaults();
                    let discord_hits = scorer.score(&direct_ids, &edge_refs);
                    for hit in &discord_hits {
                        if !existing_ids.contains(&hit.item_id) {
                            discord_results_payload.push(serde_json::json!({
                                "result_id": hit.item_id,
                                "discord_score": hit.discord_score,
                                "anchor_ids": hit.anchor_ids,
                                "relationship_types": hit.relationship_types,
                            }));
                        }
                    }
                    discord_executed = true;
                }
            }

            // Community grouping (opt-in)
            let grouped_results_payload: serde_json::Value = if group_by_community {
                let seed_ids: Vec<String> = results.iter().map(|r| r.source.result_id()).collect();
                let edges_result = block_in_place(|| {
                    handle.block_on(store.list_graph_edges_for_neighborhood(
                        seed_ids.clone(),
                        2,
                        200,
                    ))
                });
                let edges: Vec<(String, String)> = match edges_result {
                    Ok(raw_edges) => raw_edges
                        .iter()
                        .map(|edge| (edge.source.clone(), edge.target.clone()))
                        .collect(),
                    Err(_) => Vec::new(),
                };
                if !edges.is_empty() {
                    use semantic_memory::community::detect_communities;
                    let communities = detect_communities(&edges, 1.0, 42);
                    let mut member_to_comm: std::collections::HashMap<String, String> =
                        std::collections::HashMap::new();
                    for c in &communities {
                        for m in &c.members {
                            member_to_comm.insert(m.clone(), c.id.clone());
                        }
                    }
                    let mut groups: std::collections::HashMap<String, Vec<serde_json::Value>> =
                        std::collections::HashMap::new();
                    let mut ungrouped: Vec<serde_json::Value> = Vec::new();
                    for r in &json_results {
                        if let Some(rid) = r.get("result_id").and_then(|v| v.as_str()) {
                            match member_to_comm.get(rid).cloned() {
                                Some(cid) => groups.entry(cid).or_default().push(r.clone()),
                                None => ungrouped.push(r.clone()),
                            }
                        }
                    }
                    let mut map = serde_json::Map::new();
                    for (cid, items) in groups {
                        map.insert(format!("community_{cid}"), serde_json::json!(items));
                    }
                    if !ungrouped.is_empty() {
                        map.insert("ungrouped".to_string(), serde_json::json!(ungrouped));
                    }
                    serde_json::Value::Object(map)
                } else {
                    serde_json::Value::Null
                }
            } else {
                serde_json::Value::Null
            };

            // Query provenance: declare which retrieval stages contributed
            let provenance = serde_json::json!({
                "stages_fired": {
                    "bm25": results.iter().any(|r| r.bm25_rank.is_some()),
                    "vector": results.iter().any(|r| r.vector_rank.is_some()),
                    "late_interaction": true,
                    "discord": discord_executed,
                    "decoder": decoder_executed,
                },
                "result_count": results.len(),
                "view": "routed",
                "query_class": query_class,
                "widening_occurred": null, // TODO: derive from execution receipt
                "widening_reason": null,
                "verification_status": "unverified",
                "proof_reference": null,
            });

            (
                "200 OK",
                serde_json::json!({
                    "ok": true,
                    "query": query,
                    "top_k": base_top_k,
                    "results": json_results,
                    "provenance": provenance,
                    "query_class": query_class,
                    "routed": true,
                    "routing_decision": {
                        "source": routing_source,
                        "bm25_coarse": decision.bm25_coarse,
                        "vector_medium": decision.vector_medium,
                        "rerank_fine": decision.rerank_fine,
                        "graph_expansion": decision.graph_expansion,
                        "decoder": decision.decoder,
                        "discord": decision.discord,
                        "no_retrieval": decision.no_retrieval,
                        "reasoning": decision.reasoning,
                    },
                    "decoder_planned": plan.use_decoder,
                    "decoder_executed": decoder_executed,
                    "discord_planned": plan.use_discord,
                    "discord_executed": discord_executed,
                    "discord_results": discord_results_payload,
                    "factor_graph": factor_graph_payload,
                    "grouped_results": grouped_results_payload,
                }),
            )
        }
        Err(e) => (
            "500 Internal Server Error",
            serde_json::json!({"ok": false, "error": format!("search error: {e}")}),
        ),
    }
}

fn handle_stats(bridge: &MemoryBridge, handle: &Handle) -> (&'static str, serde_json::Value) {
    let store = &bridge.store;
    let core = block_in_place(|| handle.block_on(store.stats()));
    let graph = block_in_place(|| handle.block_on(store.list_all_graph_edges()));
    let core_health = match &core {
        Ok(_) => serde_json::json!({"health":"healthy","error":null}),
        Err(e) => serde_json::json!({"health":"error","error":e.to_string()}),
    };
    let graph_health = match &graph {
        Ok(_) => serde_json::json!({"health":"healthy","error":null}),
        Err(e) => serde_json::json!({"health":"error","error":e.to_string()}),
    };
    let stats = core.ok();
    let graph_edges = graph.ok().map(|edges| edges.len());
    let ok = stats.is_some() && graph_edges.is_some();
    (
        if ok {
            "200 OK"
        } else {
            "503 Service Unavailable"
        },
        serde_json::json!({
            "ok": ok,
            "components": {"core": core_health, "graph": graph_health},
            "facts": stats.as_ref().map(|s| s.total_facts),
            "documents": stats.as_ref().map(|s| s.total_documents),
            "chunks": stats.as_ref().map(|s| s.total_chunks),
            "graph_edges": graph_edges,
            "db_size_mb": stats.as_ref().map(|s| (s.database_size_bytes as f64) / (1024.0 * 1024.0)),
        }),
    )
}

fn handle_rerank(body: &str) -> (&'static str, serde_json::Value) {
    let params: serde_json::Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(e) => {
            return (
                "400 Bad Request",
                serde_json::json!({"ok": false, "error": format!("invalid JSON: {e}")}),
            )
        }
    };

    let query = params.get("query").and_then(|v| v.as_str()).unwrap_or("");
    let results = match params.get("results").and_then(|v| v.as_array()) {
        Some(r) => r.clone(),
        None => {
            return (
                "400 Bad Request",
                serde_json::json!({"ok": false, "error": "missing 'results' array"}),
            )
        }
    };

    if query.is_empty() {
        return (
            "400 Bad Request",
            serde_json::json!({"ok": false, "error": "missing 'query' field"}),
        );
    }
    if results.len() > MAX_RERANK_RESULTS {
        return (
            "400 Bad Request",
            serde_json::json!({"ok": false, "error": format!("results exceeds maximum of {MAX_RERANK_RESULTS}")}),
        );
    }
    if results.iter().any(|r| {
        r.get("content")
            .and_then(|v| v.as_str())
            .is_some_and(|s| s.len() > MAX_RERANK_CONTENT_BYTES)
    }) {
        return (
            "400 Bad Request",
            serde_json::json!({"ok": false, "error": format!("result content exceeds maximum of {MAX_RERANK_CONTENT_BYTES} bytes")}),
        );
    }

    let (reranked, rerank_status) = rerank_results(query, &results, RERANK_MODEL);
    let count = reranked.len();
    (
        "200 OK",
        serde_json::json!({
            "ok": true,
            "results": reranked,
            "count": count,
            "rerank_status": rerank_status,
            "model": RERANK_MODEL,
        }),
    )
}

fn handle_add_fact(
    _body: &str,
    _bridge: &MemoryBridge,
    _handle: &Handle,
) -> (&'static str, serde_json::Value) {
    (
        "503 Service Unavailable",
        serde_json::json!({
            "ok": false,
            "error": "HTTP evidence admission is disabled: no trusted authenticated authority issuer or immutable evidence resolver is configured"
        }),
    )
}

/// Handle /record-outcome: record a search outcome for RL routing feedback.
fn handle_record_outcome(
    body: &str,
    bridge: &MemoryBridge,
    handle: &Handle,
) -> (&'static str, serde_json::Value) {
    use semantic_memory::rl_routing::{record_routing_outcome, RoutingOutcome};
    use semantic_memory::routing::{QueryProfile, RetrievalRouter};

    let params: serde_json::Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(e) => {
            return (
                "400 Bad Request",
                serde_json::json!({"ok": false, "error": format!("invalid JSON: {e}")}),
            )
        }
    };

    let query = params.get("query").and_then(|v| v.as_str()).unwrap_or("");
    let outcome = params
        .get("outcome")
        .and_then(|v| v.as_str())
        .unwrap_or("neutral");
    let _query_class = params
        .get("query_class")
        .and_then(|v| v.as_str())
        .unwrap_or("A");

    if query.is_empty() {
        return (
            "400 Bad Request",
            serde_json::json!({"ok": false, "error": "missing 'query' field"}),
        );
    }

    let outcome_enum = match outcome.to_lowercase().as_str() {
        "good" => RoutingOutcome::Good,
        "bad" => RoutingOutcome::Bad,
        "neutral" => RoutingOutcome::Neutral,
        _ => {
            return (
                "400 Bad Request",
                serde_json::json!({"ok": false, "error": "outcome must be 'good', 'bad', or 'neutral'"}),
            )
        }
    };

    let profile = QueryProfile::from_query(query);
    let router = RetrievalRouter::default();
    let decision = router.route(&profile);

    let store = &bridge.store;
    // Load persisted policy (or default if none saved yet)
    let mut policy = match block_in_place(|| handle.block_on(store.load_routing_policy())) {
        Ok(policy) => policy.unwrap_or_default(),
        Err(e) => {
            return (
                "500 Internal Server Error",
                serde_json::json!({"ok": false, "error": format!("load routing policy error: {e}")}),
            )
        }
    };
    record_routing_outcome(&mut policy, &profile, &decision, outcome_enum);
    // Save updated policy
    if let Err(e) = block_in_place(|| handle.block_on(store.save_routing_policy(&policy))) {
        return (
            "500 Internal Server Error",
            serde_json::json!({"ok": false, "error": format!("persist routing policy error: {e}")}),
        );
    }

    (
        "200 OK",
        serde_json::json!({
            "ok": true,
            "mutating": true,
            "recorded": true,
            "feedback": {"kind": "ProxyLabel", "label": outcome},
            "routing_decision": {
                "bm25_coarse": decision.bm25_coarse,
                "vector_medium": decision.vector_medium,
                "rerank_fine": decision.rerank_fine,
                "graph_expansion": decision.graph_expansion,
                "decoder": decision.decoder,
                "discord": decision.discord,
                "no_retrieval": decision.no_retrieval,
                "reasoning": decision.reasoning,
            },
            "policy_state": {
                "trained_examples": policy.trained_examples,
                "baseline": policy.baseline,
            },
        }),
    )
}

/// Handle GET /verify-integrity: check DB integrity using real library checks.
fn handle_verify_integrity(
    bridge: &MemoryBridge,
    handle: &Handle,
) -> (&'static str, serde_json::Value) {
    let store = &bridge.store;
    let result = block_in_place(|| {
        handle.block_on(store.verify_integrity(semantic_memory::VerifyMode::Quick))
    });

    match result {
        Ok(report) => (
            "200 OK",
            serde_json::json!({
                "ok": report.ok,
                "integrity": report.ok,
                "schema_version": report.schema_version,
                "fact_count": report.fact_count,
                "chunk_count": report.chunk_count,
                "message_count": report.message_count,
                "facts_missing_embeddings": report.facts_missing_embeddings,
                "chunks_missing_embeddings": report.chunks_missing_embeddings,
                "issues": report.issues,
                "issue_count": report.issues.len(),
                "message": if report.ok { "All integrity checks passed".to_string() } else { format!("{} integrity issues found", report.issues.len()) },
            }),
        ),
        Err(e) => (
            "500 Internal Server Error",
            serde_json::json!({"ok": false, "integrity": false, "error": format!("verify_integrity error: {e}")}),
        ),
    }
}

/// Handle POST /discord: second-order retrieval via graph neighborhood.
///
/// Accepts {"query": "...", "top_k": 5, "direct_ids": ["fact:uuid1", ...]}.
/// If direct_ids not provided, runs a search first to get top_k results.
fn handle_discord(
    body: &str,
    bridge: &MemoryBridge,
    handle: &Handle,
) -> (&'static str, serde_json::Value) {
    use semantic_memory::discord::DiscordScorer;

    let params: serde_json::Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(e) => {
            return (
                "400 Bad Request",
                serde_json::json!({"ok": false, "error": format!("invalid JSON: {e}")}),
            )
        }
    };

    let query = params.get("query").and_then(|v| v.as_str()).unwrap_or("");
    let top_k = match bounded_top_k(&params, 5) {
        Ok(k) => k,
        Err(response) => return response,
    };

    // Get direct_ids from params, or run a search to get them
    let direct_ids: Vec<String> = match params.get("direct_ids").and_then(|v| v.as_array()) {
        Some(arr) => {
            if arr.len() > MAX_DIRECT_IDS || arr.iter().any(|v| !v.is_string()) {
                return (
                    "400 Bad Request",
                    serde_json::json!({"ok": false, "error": format!("direct_ids must contain at most {MAX_DIRECT_IDS} strings")}),
                );
            }
            arr.iter()
                .filter_map(|v| v.as_str().map(str::to_owned))
                .collect()
        }
        None => {
            // Need a query to search
            if query.is_empty() {
                return (
                    "400 Bad Request",
                    serde_json::json!({"ok": false, "error": "either 'direct_ids' or 'query' must be provided"}),
                );
            }
            let store = &bridge.store;
            let search_result =
                block_in_place(|| handle.block_on(store.search(query, Some(top_k), None, None)));
            match search_result {
                Ok(results) => results.iter().map(|r| r.source.result_id()).collect(),
                Err(e) => {
                    return (
                        "500 Internal Server Error",
                        serde_json::json!({"ok": false, "error": format!("search error: {e}")}),
                    )
                }
            }
        }
    };

    if direct_ids.is_empty() {
        return (
            "200 OK",
            serde_json::json!({"ok": true, "discord_results": [], "count": 0, "edges_loaded": 0}),
        );
    }

    let store = &bridge.store;
    // Load graph edges for the neighborhood
    let edges_result = block_in_place(|| {
        handle.block_on(store.list_graph_edges_for_neighborhood(direct_ids.clone(), 2, 200))
    });

    let edges: Vec<semantic_memory::discord::GraphEdgeRef> = match edges_result {
        Ok(raw_edges) => raw_edges
            .iter()
            .map(|edge| {
                let parsed_type = edge
                    .edge_type_parsed
                    .clone()
                    .or_else(|| serde_json::from_str(&edge.edge_type).ok())
                    .unwrap_or(semantic_memory::GraphEdgeType::Entity {
                        relation: "unknown".to_string(),
                    });
                let type_str = match parsed_type {
                    semantic_memory::GraphEdgeType::Semantic { .. } => "semantic",
                    semantic_memory::GraphEdgeType::Temporal { .. } => "temporal",
                    semantic_memory::GraphEdgeType::Causal { .. } => "causal",
                    semantic_memory::GraphEdgeType::Entity { .. } => "entity",
                };
                semantic_memory::discord::GraphEdgeRef {
                    source: edge.source.clone(),
                    target: edge.target.clone(),
                    edge_type: type_str.to_string(),
                    weight: edge.weight,
                }
            })
            .collect(),
        Err(e) => {
            return (
                "500 Internal Server Error",
                serde_json::json!({"ok": false, "error": format!("failed to load graph edges: {e}")}),
            )
        }
    };

    let edges_loaded = edges.len();
    let scorer = DiscordScorer::with_defaults();
    let discord_hits = scorer.score(&direct_ids, &edges);

    // Filter out items already in direct_ids
    let existing: std::collections::HashSet<String> = direct_ids.iter().cloned().collect();
    let filtered_hits: Vec<serde_json::Value> = discord_hits
        .iter()
        .filter(|hit| !existing.contains(&hit.item_id))
        .map(|hit| {
            serde_json::json!({
                "result_id": hit.item_id,
                "discord_score": hit.discord_score,
                "anchor_ids": hit.anchor_ids,
                "relationship_types": hit.relationship_types,
            })
        })
        .collect();

    (
        "200 OK",
        serde_json::json!({
            "ok": true,
            "discord_results": filtered_hits,
            "count": filtered_hits.len(),
            "edges_loaded": edges_loaded,
            "direct_ids": direct_ids,
        }),
    )
}

fn bounded_top_k(
    params: &serde_json::Value,
    default: usize,
) -> Result<usize, (&'static str, serde_json::Value)> {
    let Some(value) = params.get("top_k") else {
        return Ok(default);
    };
    let Some(value) = value.as_u64() else {
        return Err((
            "400 Bad Request",
            serde_json::json!({"ok": false, "error": "top_k must be an unsigned integer"}),
        ));
    };
    if value == 0 || value > MAX_TOP_K {
        return Err((
            "400 Bad Request",
            serde_json::json!({"ok": false, "error": format!("top_k must be between 1 and {MAX_TOP_K}")}),
        ));
    }
    Ok(value as usize)
}

// ---------------------------------------------------------------------------
// Maintenance endpoints — for hooks and cron jobs to trigger auto-management
// without needing MCP tools to be visible (they're hidden in lean profile).
// ---------------------------------------------------------------------------

/// Handle POST /maintenance/check: returns embeddings_are_dirty + verify_integrity(Quick)
/// in one call. This is the "health check" for auto-management.
fn handle_maintenance_check(
    bridge: &MemoryBridge,
    handle: &Handle,
) -> (&'static str, serde_json::Value) {
    let store = &bridge.store;
    let embeddings_dirty = block_in_place(|| handle.block_on(store.embeddings_are_dirty()));
    let embeddings_dirty = match embeddings_dirty {
        Ok(v) => v,
        Err(e) => {
            return (
                "500 Internal Server Error",
                serde_json::json!({"ok": false, "error": format!("embeddings_are_dirty error: {e}")}),
            )
        }
    };
    let integrity_result = block_in_place(|| {
        handle.block_on(store.verify_integrity(semantic_memory::VerifyMode::Quick))
    });

    match integrity_result {
        Ok(report) => (
            "200 OK",
            serde_json::json!({
                "ok": report.ok,
                "embeddings_are_dirty": embeddings_dirty,
                "integrity": {
                    "ok": report.ok,
                    "schema_version": report.schema_version,
                    "fact_count": report.fact_count,
                    "chunk_count": report.chunk_count,
                    "message_count": report.message_count,
                    "facts_missing_embeddings": report.facts_missing_embeddings,
                    "chunks_missing_embeddings": report.chunks_missing_embeddings,
                    "issues": report.issues,
                    "issue_count": report.issues.len(),
                },
                "message": if report.ok && !embeddings_dirty {
                    "All checks passed".to_string()
                } else if report.ok && embeddings_dirty {
                    "Integrity OK but embeddings need re-embedding".to_string()
                } else {
                    format!("{} integrity issues found", report.issues.len())
                },
            }),
        ),
        Err(e) => (
            "500 Internal Server Error",
            serde_json::json!({
                "ok": false,
                "embeddings_are_dirty": embeddings_dirty,
                "error": format!("verify_integrity error: {e}"),
            }),
        ),
    }
}

/// Handle POST /maintenance/vacuum: calls store.vacuum(). Returns ok.
fn handle_maintenance_vacuum(
    bridge: &MemoryBridge,
    handle: &Handle,
) -> (&'static str, serde_json::Value) {
    let store = &bridge.store;
    let result = block_in_place(|| handle.block_on(store.vacuum()));

    match result {
        Ok(()) => (
            "200 OK",
            serde_json::json!({"ok": true, "action": "vacuum", "message": "Database vacuumed successfully"}),
        ),
        Err(e) => (
            "500 Internal Server Error",
            serde_json::json!({"ok": false, "error": format!("vacuum error: {e}")}),
        ),
    }
}

/// Handle POST /maintenance/reembed: calls store.reembed_all(). Returns count.
/// This is expensive so the handler just calls it and returns the count.
fn handle_maintenance_reembed(
    bridge: &MemoryBridge,
    handle: &Handle,
) -> (&'static str, serde_json::Value) {
    let store = &bridge.store;
    let result = block_in_place(|| handle.block_on(store.reembed_all()));

    match result {
        Ok(count) => (
            "200 OK",
            serde_json::json!({"ok": true, "action": "reembed", "reembedded_count": count, "message": format!("Re-embedded {count} items")}),
        ),
        Err(e) => (
            "500 Internal Server Error",
            serde_json::json!({"ok": false, "error": format!("reembed_all error: {e}")}),
        ),
    }
}

/// Handle POST /maintenance/reconcile: accepts {"action": "ReportOnly"|"RebuildFts"|"ReEmbed"}.
/// Calls store.reconcile(action). Returns the IntegrityReport.
fn handle_maintenance_reconcile(
    body: &str,
    bridge: &MemoryBridge,
    handle: &Handle,
) -> (&'static str, serde_json::Value) {
    let params: serde_json::Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(e) => {
            return (
                "400 Bad Request",
                serde_json::json!({"ok": false, "error": format!("invalid JSON: {e}")}),
            )
        }
    };

    let action_str = params
        .get("action")
        .and_then(|v| v.as_str())
        .unwrap_or("ReportOnly");

    let action = match action_str {
        "RebuildFts" => semantic_memory::ReconcileAction::RebuildFts,
        "ReEmbed" => semantic_memory::ReconcileAction::ReEmbed,
        _ => semantic_memory::ReconcileAction::ReportOnly,
    };

    let store = &bridge.store;
    let result = block_in_place(|| handle.block_on(store.reconcile(action)));

    match result {
        Ok(report) => (
            "200 OK",
            serde_json::json!({
                "ok": report.ok,
                "action": "reconcile",
                "reconcile_action": action_str,
                "integrity": {
                    "ok": report.ok,
                    "schema_version": report.schema_version,
                    "fact_count": report.fact_count,
                    "chunk_count": report.chunk_count,
                    "message_count": report.message_count,
                    "facts_missing_embeddings": report.facts_missing_embeddings,
                    "chunks_missing_embeddings": report.chunks_missing_embeddings,
                    "issues": report.issues,
                    "issue_count": report.issues.len(),
                },
                "message": if report.ok {
                    "Reconciliation completed, no issues found".to_string()
                } else {
                    format!("Reconciliation completed with {} issues", report.issues.len())
                },
            }),
        ),
        Err(e) => (
            "500 Internal Server Error",
            serde_json::json!({"ok": false, "error": format!("reconcile error: {e}")}),
        ),
    }
}

/// Handle POST /maintenance/rebuild-hnsw: calls store.rebuild_hnsw_index().
///
/// Rebuilds the HNSW sidecar from current SQLite embeddings. Use this when
/// the index is stale (e.g. after bulk imports, model changes, or long periods
/// without automatic sync).
fn handle_maintenance_rebuild_hnsw(
    bridge: &MemoryBridge,
    handle: &Handle,
) -> (&'static str, serde_json::Value) {
    #[cfg(feature = "hnsw")]
    {
        let store = &bridge.store;
        let result = block_in_place(|| handle.block_on(store.rebuild_hnsw_index()));

        match result {
            Ok(receipt) => (
                "200 OK",
                serde_json::json!({
                    "ok": true,
                    "action": "rebuild-hnsw",
                    "message": "HNSW index rebuilt successfully",
                    "generation_id": receipt.generation_id,
                    "vector_count": receipt.source_row_count,
                }),
            ),
            Err(e) => (
                "500 Internal Server Error",
                serde_json::json!({"ok": false, "error": format!("rebuild_hnsw error: {e}")}),
            ),
        }
    }

    #[cfg(not(feature = "hnsw"))]
    {
        let _ = bridge;
        let _ = handle;
        (
            "200 OK",
            serde_json::json!({
                "ok": true,
                "action": "rebuild-hnsw",
                "message": "HNSW rebuild not applicable — usearch backend does not require rebuild",
                "skipped": true,
            }),
        )
    }
}

/// Handle POST /maintenance/compact-hnsw: calls store.compact_hnsw(). Returns ok.
///
/// Only available when the `hnsw` feature is enabled. The default backend is
/// usearch, so this endpoint returns a not-applicable response without the feature.
fn handle_maintenance_compact_hnsw(
    bridge: &MemoryBridge,
    handle: &Handle,
) -> (&'static str, serde_json::Value) {
    #[cfg(feature = "hnsw")]
    {
        let store = &bridge.store;
        let result = block_in_place(|| handle.block_on(store.compact_hnsw()));

        match result {
            Ok(()) => (
                "200 OK",
                serde_json::json!({"ok": true, "action": "compact-hnsw", "message": "HNSW index compacted successfully"}),
            ),
            Err(e) => (
                "500 Internal Server Error",
                serde_json::json!({"ok": false, "error": format!("compact_hnsw error: {e}")}),
            ),
        }
    }

    #[cfg(not(feature = "hnsw"))]
    {
        let _ = bridge;
        let _ = handle;
        (
            "200 OK",
            serde_json::json!({
                "ok": true,
                "action": "compact-hnsw",
                "message": "HNSW compaction not applicable — usearch backend does not require compaction",
                "skipped": true,
            }),
        )
    }
}

#[cfg(test)]
mod connection_slot_tests {
    use super::*;

    #[test]
    fn connection_slot_is_released_when_handler_unwinds() {
        let active = Arc::new(AtomicUsize::new(1));
        let result = std::panic::catch_unwind({
            let active = active.clone();
            move || {
                let _slot = ConnectionSlot::new(active);
                panic!("injected handler panic");
            }
        });
        assert!(result.is_err());
        assert_eq!(active.load(Ordering::Acquire), 0);
    }

    #[test]
    fn rerank_failure_preserves_original_order_and_scores() {
        let results = vec![
            serde_json::json!({"result_id": "fact:first", "score": 0.9, "content": "first"}),
            serde_json::json!({"result_id": "fact:second", "score": 0.8, "content": "second"}),
        ];
        let (reranked, status) =
            rerank_results_at("query", &results, RERANK_MODEL, "http://127.0.0.1:0");

        assert_eq!(status, "degraded");
        assert_eq!(reranked, results);
        assert!(reranked
            .iter()
            .all(|result| result.get("rerank_score").is_none()));
    }

    #[test]
    fn rerank_limits_use_utf8_bytes_and_bound_search_candidates() {
        let content = "🦀".repeat(MAX_RERANK_CONTENT_BYTES);
        let truncated = truncate_rerank_content(&content);
        assert!(truncated.len() <= MAX_RERANK_CONTENT_BYTES);
        assert!(truncated.is_char_boundary(truncated.len()));

        assert_eq!(rerank_candidate_limit(1), 2);
        assert_eq!(
            rerank_candidate_limit(MAX_TOP_K as usize),
            MAX_RERANK_RESULTS
        );
    }
}
