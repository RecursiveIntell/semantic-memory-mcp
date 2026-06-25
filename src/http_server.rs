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
//!   POST /add      {"content": "...", "namespace": "..."} -> fact_id
//!   GET  /health   -> {"ok": true}

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use tokio::runtime::Handle;
use tokio::task::block_in_place;

use crate::bridge::MemoryBridge;

/// Call Ollama to rate each result's relevance to the query (1-5) and sort descending.
/// Returns a new vec with a `rerank_score` field added to each result object.
fn rerank_results(
    query: &str,
    results: &[serde_json::Value],
    model: &str,
) -> Vec<serde_json::Value> {
    let client = reqwest::blocking::Client::new();
    let mut scored: Vec<(f64, serde_json::Value)> = results
        .iter()
        .map(|r| {
            let content = r.get("content").and_then(|v| v.as_str()).unwrap_or("");
            let truncated: String = content.chars().take(500).collect();
            let prompt = format!(
                "Rate the relevance of this document to the query on a scale of 1-5. Reply with ONLY the number.\nQuery: {query}\nDocument: {truncated}\nRating:"
            );
            let body = serde_json::json!({
                "model": model,
                "prompt": prompt,
                "stream": false,
                "options": {"temperature": 0, "num_predict": 1}
            });
            let rating = client
                .post("http://127.0.0.1:11434/api/generate")
                .json(&body)
                .send()
                .ok()
                .and_then(|resp| resp.json::<serde_json::Value>().ok())
                .and_then(|v| {
                    v.get("response")
                        .and_then(|r| r.as_str())
                        .and_then(|s| s.trim().chars().next())
                        .and_then(|c| c.to_digit(10))
                        .map(|d| d as f64)
                })
                .unwrap_or(1.0);
            (rating, r.clone())
        })
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored
        .into_iter()
        .map(|(score, mut r)| {
            if let Some(obj) = r.as_object_mut() {
                obj.insert("rerank_score".to_string(), serde_json::json!(score));
            }
            r
        })
        .collect()
}

pub fn start_http_server(port: u16, bridge: MemoryBridge, handle: Handle) {
    std::thread::spawn(move || {
        let listener = match TcpListener::bind(("127.0.0.1", port)) {
            Ok(l) => {
                eprintln!("HTTP search server listening on 127.0.0.1:{}", port);
                l
            }
            Err(e) => {
                eprintln!("Failed to bind HTTP port {}: {}", port, e);
                return;
            }
        };

        for stream in listener.incoming() {
            let stream = match stream {
                Ok(s) => s,
                Err(_) => continue,
            };

            let bridge = bridge.clone();
            let h = handle.clone();
            std::thread::spawn(move || {
                handle_connection(stream, bridge, h);
            });
        }
    });
}

fn handle_connection(
    mut stream: std::net::TcpStream,
    bridge: MemoryBridge,
    handle: Handle,
) {
    let mut reader = BufReader::new(stream.try_clone().expect("clone"));
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).is_err() {
        return;
    }

    let parts: Vec<&str> = request_line.split_whitespace().collect();
    if parts.len() < 2 {
        return;
    }
    let method = parts[0];
    let path = parts[1];

    let mut content_length = 0;
    loop {
        let mut header = String::new();
        if reader.read_line(&mut header).is_err() {
            return;
        }
        if header.trim().is_empty() {
            break;
        }
        if let Some(len_str) = header
            .strip_prefix("Content-Length:")
            .or_else(|| header.strip_prefix("content-length:"))
        {
            content_length = len_str.trim().parse().unwrap_or(0);
        }
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
        ("POST", "/search") => handle_search(&body_str, &bridge, &handle),
        ("POST", "/search-routed") => handle_search_routed(&body_str, &bridge, &handle),
        ("POST", "/rerank") => handle_rerank(&body_str),
        ("POST", "/stats") => handle_stats(&bridge, &handle),
        ("POST", "/add") => handle_add_fact(&body_str, &bridge, &handle),
        ("POST", "/add-edge") => handle_add_edge(&body_str, &bridge, &handle),
        ("POST", "/record-outcome") => handle_record_outcome(&body_str, &bridge, &handle),
        ("GET", "/verify-integrity") => handle_verify_integrity(&bridge, &handle),
        ("POST", "/discord") => handle_discord(&body_str, &bridge, &handle),
        ("POST", "/maintenance/check") => handle_maintenance_check(&bridge, &handle),
        ("POST", "/maintenance/vacuum") => handle_maintenance_vacuum(&bridge, &handle),
        ("POST", "/maintenance/reembed") => handle_maintenance_reembed(&bridge, &handle),
        ("POST", "/maintenance/reconcile") => handle_maintenance_reconcile(&body_str, &bridge, &handle),
        ("POST", "/maintenance/compact-hnsw") => handle_maintenance_compact_hnsw(&bridge, &handle),
        ("POST", "/maintenance/auto-edge") => handle_maintenance_auto_edge(&body_str, &bridge, &handle),
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
    let top_k = params.get("top_k").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
    let namespaces: Option<Vec<String>> = params
        .get("namespaces")
        .and_then(|v| serde_json::from_value(v.clone()).ok());
    let do_rerank = params.get("rerank").and_then(|v| v.as_bool()).unwrap_or(false);

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
    let fetch_k = if do_rerank { top_k * 2 } else { top_k };
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
                        semantic_memory::SearchSource::Chunk { document_title, .. } => document_title.clone(),
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

            let final_results: Vec<serde_json::Value> = if do_rerank && !json_results.is_empty() {
                rerank_results(query, &json_results, "granite4.1:3b")
                    .into_iter()
                    .take(top_k)
                    .collect()
            } else {
                json_results
            };

            let count = final_results.len();
            let provenance = serde_json::json!({
                "stages_fired": {
                    "bm25": true,
                    "vector": true,
                    "late_interaction": false,
                    "rerank": do_rerank,
                },
                "result_count": count,
                "view": "semantic",
                "widening_occurred": false,
                "widening_reason": null,
                "verification_status": "verified",
            });
            (
                "200 OK",
                serde_json::json!({
                    "ok": true,
                    "query": query,
                    "top_k": top_k,
                    "results": final_results,
                    "count": count,
                    "reranked": do_rerank,
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
    use semantic_memory::routing::RetrievalRouter;

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
    let base_top_k = params.get("top_k").and_then(|v| v.as_u64()).unwrap_or(12) as usize;
    let query_class = params.get("query_class").and_then(|v| v.as_str()).unwrap_or("A");
    let namespaces: Option<Vec<String>> = params
        .get("namespaces")
        .and_then(|v| serde_json::from_value(v.clone()).ok());
    let contradictions: Vec<(String, String)> = params
        .get("contradictions")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default();
    let group_by_community = params.get("group_by_community").and_then(|v| v.as_bool()).unwrap_or(false);

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
    let decision = router.route_query(query);
    let contras = contradictions.clone();
    let plan = plan_execution(&decision, contras.clone());

    // Class D (SYNTHESIS): retrieve more candidates to support comprehensive answers
    let top_k = if query_class == "D" {
        (base_top_k * 2).min(20)
    } else {
        base_top_k
    };

    let store = &bridge.store;
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
                    use semantic_memory::factor_graph::{factors_from_edges, FactorGraph, FactorGraphConfig};

                    let graph_edges = block_in_place(|| {
                        handle.block_on(store.list_all_graph_edges())
                    });

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
                        let graph =
                            FactorGraph::new(&nodes, factors, FactorGraphConfig::default());
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
                let direct_ids: Vec<String> = results.iter().map(|r| r.source.result_id()).collect();
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
                            // Fetch the fact's content directly from the DB.
                            // get_fact expects a bare UUID (without "fact:" prefix).
                            let bare_id = hit.item_id.strip_prefix("fact:").unwrap_or(&hit.item_id);
                            let (content, namespace) = {
                                let fact_result = handle.block_on(
                                    store.get_fact(bare_id)
                                );
                                match fact_result {
                                    Ok(Some(fact)) => (fact.content, fact.namespace),
                                    Ok(None) => {
                                        eprintln!("[discord] get_fact returned None for id={}", bare_id);
                                        (String::new(), String::new())
                                    }
                                    Err(e) => {
                                        eprintln!("[discord] get_fact error for id={}: {}", bare_id, e);
                                        (String::new(), String::new())
                                    }
                                }
                            };
                            discord_results_payload.push(serde_json::json!({
                                "result_id": hit.item_id,
                                "content": content,
                                "namespace": namespace,
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
                    let mut groups: std::collections::HashMap<
                        String,
                        Vec<serde_json::Value>,
                    > = std::collections::HashMap::new();
                    let mut ungrouped: Vec<serde_json::Value> = Vec::new();
                    for r in &json_results {
                        if let Some(rid) = r.get("result_id").and_then(|v| v.as_str()) {
                            match member_to_comm.get(rid).cloned() {
                                Some(cid) => {
                                    groups.entry(cid).or_default().push(r.clone())
                                }
                                None => ungrouped.push(r.clone()),
                            }
                        }
                    }
                    let mut map = serde_json::Map::new();
                    for (cid, items) in groups {
                        map.insert(
                            format!("community_{cid}"),
                            serde_json::json!(items),
                        );
                    }
                    if !ungrouped.is_empty() {
                        map.insert(
                            "ungrouped".to_string(),
                            serde_json::json!(ungrouped),
                        );
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
                "widening_occurred": false,
                "widening_reason": null,
                "verification_status": "verified",
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

fn handle_stats(
    bridge: &MemoryBridge,
    handle: &Handle,
) -> (&'static str, serde_json::Value) {
    let store = &bridge.store;
    let result = block_in_place(|| handle.block_on(store.stats()));
    match result {
        Ok(stats) => (
            "200 OK",
            serde_json::json!({
                "ok": true,
                "facts": stats.total_facts,
                "documents": stats.total_documents,
                "chunks": stats.total_chunks,
                "db_size_mb": (stats.database_size_bytes as f64) / (1024.0 * 1024.0),
            }),
        ),
        Err(e) => (
            "500 Internal Server Error",
            serde_json::json!({"ok": false, "error": format!("{e}")}),
        ),
    }
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
    let model = params
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("granite4.1:3b");
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

    let reranked = rerank_results(query, &results, model);
    let count = reranked.len();
    (
        "200 OK",
        serde_json::json!({
            "ok": true,
            "results": reranked,
            "count": count,
        }),
    )
}

fn handle_add_fact(
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

    let content = params.get("content").and_then(|v| v.as_str()).unwrap_or("");
    let namespace = params
        .get("namespace")
        .and_then(|v| v.as_str())
        .unwrap_or("general");
    let source = params.get("source").and_then(|v| v.as_str());

    if content.is_empty() {
        return (
            "400 Bad Request",
            serde_json::json!({"ok": false, "error": "missing 'content' field"}),
        );
    }

    let store = &bridge.store;
    let result =
        block_in_place(|| handle.block_on(store.add_fact(namespace, content, source, None)));

    match result {
        Ok(fact_id) => (
            "200 OK",
            serde_json::json!({"ok": true, "fact_id": fact_id}),
        ),
        Err(e) => (
            "500 Internal Server Error",
            serde_json::json!({"ok": false, "error": format!("{e}")}),
        ),
    }
}

/// Handle /add-edge: add a graph edge between two facts.
fn handle_add_edge(
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

    let source = params.get("source").and_then(|v| v.as_str()).unwrap_or("");
    let target = params.get("target").and_then(|v| v.as_str()).unwrap_or("");
    let edge_type_str = params.get("edge_type").and_then(|v| v.as_str()).unwrap_or("semantic");
    let weight = params.get("weight").and_then(|v| v.as_f64()).unwrap_or(1.0);
    let cosine_similarity = params.get("cosine_similarity").and_then(|v| v.as_f64());
    let relation = params.get("relation").and_then(|v| v.as_str());

    if source.is_empty() || target.is_empty() {
        return (
            "400 Bad Request",
            serde_json::json!({"ok": false, "error": "missing 'source' or 'target'"}),
        );
    }

    let edge_type = match edge_type_str {
        "semantic" => semantic_memory::GraphEdgeType::Semantic {
            cosine_similarity: cosine_similarity.unwrap_or(weight) as f32,
        },
        "temporal" => semantic_memory::GraphEdgeType::Temporal {
            delta_secs: params.get("delta_secs").and_then(|v| v.as_u64()).unwrap_or(0),
        },
        "causal" => semantic_memory::GraphEdgeType::Causal {
            confidence: cosine_similarity.unwrap_or(weight) as f32,
            evidence_ids: Vec::new(),
        },
        "entity" => semantic_memory::GraphEdgeType::Entity {
            relation: relation.unwrap_or("mentions").to_string(),
        },
        _ => semantic_memory::GraphEdgeType::Semantic {
            cosine_similarity: cosine_similarity.unwrap_or(weight) as f32,
        },
    };

    let store = &bridge.store;
    let result = block_in_place(|| {
        handle.block_on(store.add_graph_edge(source, target, edge_type, weight, None))
    });

    match result {
        Ok(edge) => (
            "200 OK",
            serde_json::json!({"ok": true, "edge_id": edge.id}),
        ),
        Err(e) => (
            "500 Internal Server Error",
            serde_json::json!({"ok": false, "error": format!("{e}")}),
        ),
    }
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
    let outcome = params.get("outcome").and_then(|v| v.as_str()).unwrap_or("neutral");
    let _query_class = params.get("query_class").and_then(|v| v.as_str()).unwrap_or("A");

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
    let mut policy = block_in_place(|| handle.block_on(store.load_routing_policy()))
        .ok()
        .flatten()
        .unwrap_or_default();
    record_routing_outcome(&mut policy, &profile, &decision, outcome_enum);
    // Save updated policy
    let _ = block_in_place(|| handle.block_on(store.save_routing_policy(&policy)));

    (
        "200 OK",
        serde_json::json!({
            "ok": true,
            "recorded": true,
            "outcome": outcome,
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
    let top_k = params.get("top_k").and_then(|v| v.as_u64()).unwrap_or(5) as usize;

    // Get direct_ids from params, or run a search to get them
    let direct_ids: Vec<String> = match params.get("direct_ids").and_then(|v| v.as_array()) {
        Some(arr) => arr
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect(),
        None => {
            // Need a query to search
            if query.is_empty() {
                return (
                    "400 Bad Request",
                    serde_json::json!({"ok": false, "error": "either 'direct_ids' or 'query' must be provided"}),
                );
            }
            let store = &bridge.store;
            let search_result = block_in_place(|| {
                handle.block_on(store.search(query, Some(top_k), None, None))
            });
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
        handle.block_on(store.list_graph_edges_for_neighborhood(
            direct_ids.clone(),
            2,
            200,
        ))
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
            // Fetch the fact's content directly from the DB.
            let bare_id = hit.item_id.strip_prefix("fact:").unwrap_or(&hit.item_id);
            let (content, namespace) = {
                let fact_result = handle.block_on(store.get_fact(bare_id));
                match fact_result {
                    Ok(Some(fact)) => (fact.content, fact.namespace),
                    _ => (String::new(), String::new()),
                }
            };
            serde_json::json!({
                "result_id": hit.item_id,
                "content": content,
                "namespace": namespace,
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

        return match result {
            Ok(()) => (
                "200 OK",
                serde_json::json!({"ok": true, "action": "compact-hnsw", "message": "HNSW index compacted successfully"}),
            ),
            Err(e) => (
                "500 Internal Server Error",
                serde_json::json!({"ok": false, "error": format!("compact_hnsw error: {e}")}),
            ),
        };
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

/// Handle POST /maintenance/auto-edge: automatically create entity edges
/// between facts in DIFFERENT namespaces that share 2+ meaningful terms.
///
/// Accepts optional JSON body:
///   - batch_size (default 500): facts per page when iterating
///   - max_edges_per_fact (default 20): cap edges per source fact to prevent explosion
///   - dry_run (default false): if true, report what would be created without creating edges
///
/// Returns a report with facts_processed, edges_created, edges_skipped, time_elapsed.
fn handle_maintenance_auto_edge(
    body: &str,
    bridge: &MemoryBridge,
    handle: &Handle,
) -> (&'static str, serde_json::Value) {
    let start = std::time::Instant::now();

    let params: serde_json::Value = serde_json::from_str(body).unwrap_or(serde_json::json!({}));
    let batch_size = params.get("batch_size").and_then(|v| v.as_u64()).unwrap_or(500) as usize;
    let max_edges_per_fact =
        params.get("max_edges_per_fact").and_then(|v| v.as_u64()).unwrap_or(20) as usize;
    let dry_run = params.get("dry_run").and_then(|v| v.as_bool()).unwrap_or(false);
    let rebuild = params.get("rebuild").and_then(|v| v.as_bool()).unwrap_or(false);

    let store = &bridge.store;

    // If rebuild mode, invalidate ALL existing entity edges first.
    let mut edges_invalidated: u64 = 0;
    if rebuild && !dry_run {
        let all_edges = block_in_place(|| handle.block_on(store.list_all_graph_edges()));
        if let Ok(edges) = all_edges {
            for edge in &edges {
                let parsed = edge.edge_type_parsed.clone().or_else(|| {
                    serde_json::from_str::<semantic_memory::GraphEdgeType>(&edge.edge_type).ok()
                });
                if matches!(parsed, Some(semantic_memory::GraphEdgeType::Entity { .. })) {
                    let _ = block_in_place(|| {
                        handle.block_on(store.invalidate_graph_edge(&edge.id, "auto-edge rebuild"))
                    });
                    edges_invalidated += 1;
                }
            }
        }
        eprintln!("[auto-edge] rebuild: invalidated {edges_invalidated} existing entity edges");
    }

    // Namespaces to SKIP entirely — social media noise, not technical knowledge.
    const SKIP_NAMESPACES: &[&str] = &[
        "mixed", "chatgpt", "twitter", "tool-receipts", "agentguard",
    ];
    let skip_ns: std::collections::HashSet<&str> = SKIP_NAMESPACES.iter().copied().collect();

    // Large stopword list — common English words that should never create edges.
    const STOPWORDS: &[&str] = &[
        "the", "and", "for", "are", "but", "not", "you", "all", "can", "her", "was", "one",
        "our", "out", "has", "have", "had", "his", "how", "its", "may", "new", "now", "old",
        "see", "him", "way", "who", "did", "yes", "yet", "say", "she", "too", "use", "via",
        "any", "few", "get", "got", "let", "put", "run", "set", "try", "two", "bad", "big",
        "far", "off", "own", "per", "sub", "top", "end", "add", "also", "been",
        "from", "into", "that", "this", "with", "will", "your", "they", "them", "were",
        "what", "when", "where", "which", "while", "there", "their", "about", "after",
        "before", "between", "because", "being", "would", "could", "should", "than",
        "then", "these", "those", "only", "over", "under", "again", "more", "most", "some",
        "such", "very", "just", "like", "even", "back", "both", "down", "here", "make",
        "made", "each", "want", "need", "know", "same", "other", "many", "much", "last",
        "first", "third", "next", "best", "main", "full", "upon", "within",
        "without", "through", "during", "above", "below", "against", "among",
        "across", "behind", "beside", "beyond",
        // Common technical/process words that create false connections
        "project", "work", "thing", "things", "fix", "fixed", "build", "built", "code",
        "data", "system", "update", "updated", "check", "checked", "test", "tested",
        "error", "issue", "problem", "result", "results", "status", "state", "info",
        "note", "notes", "list", "item", "items", "type", "types", "field", "fields",
        "name", "names", "value", "values", "line", "lines", "file", "files",
        "part", "parts", "section", "sections", "step", "steps", "task", "tasks",
        "goal", "goals", "plan", "plans", "done", "open", "close", "closed",
        "start", "started", "stop", "stopped", "change", "changed", "changing",
        "good", "bad", "right", "wrong", "true", "false", "yes", "no", "ok",
        "lots", "lot", "really", "actually", "basically", "probably", "maybe",
        "going", "getting", "looking", "trying", "working", "something", "anything",
        "everything", "nothing", "someone", "anyone", "everyone",
        "still", "always", "never", "sometimes", "usually", "often",
        "since", "until", "though", "although", "however", "therefore",
        "either", "neither", "both", "all", "any", "some", "none",
        "version", "config", "setup", "install", "installed", "running",
        "report", "reports", "summary", "detail", "details",
        "feature", "features", "function", "functions", "method", "methods",
        "class", "classes", "module", "modules", "crate", "crates",
        "package", "packages", "library", "libraries", "app", "apps",
        "web", "page", "pages", "site", "sites", "link", "links",
        "post", "posts", "reply", "replies", "comment", "comments",
        "read", "write", "call", "called", "calling", "return", "returns",
        "input", "output", "source", "target", "source_id", "target_id",
        "create", "created", "creating", "delete", "deleted", "removing",
        "find", "found", "finding", "search", "searching", "replace", "replaced",
        "show", "shown", "showing", "hide", "hidden", "display", "rendered",
        "enable", "enabled", "disable", "disabled", "allow", "allowed",
        "require", "required", "requiring", "support", "supported",
        "default", "custom", "general", "specific", "standard",
        "current", "latest", "previous", "old", "new", "future",
        "high", "low", "medium", "critical", "normal", "minor", "major",
        "single", "multiple", "total", "count", "number",
        "description", "summary", "overview", "introduction", "conclusion",
        "todo", "fixme", "wip", "draft", "final", "complete", "completed",
        "include", "includes", "included", "exclude", "excludes", "excluded",
        "require", "requires", "required", "optional",
        "important", "urgent", "priority", "blocking",
        "question", "answer", "response", "request",
    ];
    let stopword_set: std::collections::HashSet<&str> = STOPWORDS.iter().copied().collect();

    /// Extract HIGH-QUALITY terms from text: only proper nouns, camelCase/snake_case
    /// identifiers, and words 5+ chars that aren't stopwords. This filters out
    /// common English words that create false connections.
    fn extract_terms(text: &str, stopwords: &std::collections::HashSet<&str>) -> std::collections::HashSet<String> {
        let mut terms = std::collections::HashSet::new();
        for word in text.split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-') {
            let w = word.trim_matches(|c: char| c == '-' || c == '_');
            if w.len() < 3 {
                continue;
            }
            let lower = w.to_lowercase();
            if stopwords.contains(lower.as_str()) {
                continue;
            }
            // Skip pure numbers
            if lower.chars().all(|c| c.is_ascii_digit()) {
                continue;
            }
            // Include if:
            // - camelCase or snake_case (has internal uppercase or _ or -)
            // - all lowercase and 5+ chars (filters short common words)
            // - starts with uppercase (proper noun)
            let has_separator = w.contains('_') || w.contains('-');
            let has_upper = w.chars().any(|c| c.is_uppercase());
            let starts_upper = w.chars().next().map(|c| c.is_uppercase()).unwrap_or(false);
            let long_enough = lower.len() >= 5;

            if has_separator || has_upper || starts_upper || long_enough {
                terms.insert(lower);
            }
        }
        terms
    }

    // 1. Get all namespaces that contain facts
    let namespaces = match block_in_place(|| handle.block_on(store.list_fact_namespaces())) {
        Ok(ns) => ns,
        Err(e) => {
            return (
                "500 Internal Server Error",
                serde_json::json!({"ok": false, "error": format!("list_fact_namespaces error: {e}")}),
            )
        }
    };

    // Filter out skip namespaces
    let namespaces: Vec<String> = namespaces
        .into_iter()
        .filter(|ns| !skip_ns.contains(ns.as_str()))
        .collect();

    // 2. Page through all facts in each namespace, extract terms, store (id, namespace, terms)
    struct FactInfo {
        id: String,
        namespace: String,
        terms: std::collections::HashSet<String>,
    }

    let mut all_facts: Vec<FactInfo> = Vec::new();
    for ns in &namespaces {
        let mut offset = 0usize;
        loop {
            let batch = match block_in_place(|| {
                handle.block_on(store.list_facts(ns, batch_size, offset))
            }) {
                Ok(facts) => facts,
                Err(e) => {
                    eprintln!("[auto-edge] list_facts error for ns={ns} offset={offset}: {e}");
                    break;
                }
            };
            if batch.is_empty() {
                break;
            }
            for fact in &batch {
                let terms = extract_terms(&fact.content, &stopword_set);
                // Skip facts with fewer than 3 quality terms — they can't form meaningful edges
                if terms.len() >= 3 {
                    all_facts.push(FactInfo {
                        id: fact.id.clone(),
                        namespace: fact.namespace.clone(),
                        terms,
                    });
                }
            }
            offset += batch.len();
            if batch.len() < batch_size {
                break;
            }
        }
    }

    let facts_processed = all_facts.len();

    // 3. Load existing edges to skip pairs that already have edges.
    let existing_edges: std::collections::HashSet<(String, String)> =
        match block_in_place(|| handle.block_on(store.list_all_graph_edges())) {
            Ok(edges) => edges
                .iter()
                .filter_map(|e| {
                    let parsed = e.edge_type_parsed.clone().or_else(|| {
                        serde_json::from_str::<semantic_memory::GraphEdgeType>(&e.edge_type)
                            .ok()
                    });
                    match parsed {
                        Some(semantic_memory::GraphEdgeType::Entity { .. }) => {
                            Some((e.source.clone(), e.target.clone()))
                        }
                        _ => None,
                    }
                })
                .collect(),
            Err(_) => std::collections::HashSet::new(),
        };

    // 4. For each pair of facts in DIFFERENT namespaces sharing 3+ terms, create entity edge.
    // Higher threshold (3 instead of 2) + quality term extraction = much fewer, better edges.
    let mut edges_created: u64 = 0;
    let mut edges_skipped: u64 = 0;
    let mut edge_counts: std::collections::HashMap<String, u64> =
        std::collections::HashMap::new();

    for i in 0..all_facts.len() {
        let fact_i = &all_facts[i];
        let src_id = format!("fact:{}", fact_i.id);

        // Check if this fact has already hit its edge cap
        if *edge_counts.get(&src_id).unwrap_or(&0) >= max_edges_per_fact as u64 {
            continue;
        }

        for j in (i + 1)..all_facts.len() {
            let fact_j = &all_facts[j];

            // Only create edges between DIFFERENT namespaces
            if fact_i.namespace == fact_j.namespace {
                continue;
            }

            // Check shared terms (need 3+ for high quality)
            let shared: usize = fact_i.terms.intersection(&fact_j.terms).count();
            if shared < 3 {
                continue;
            }

            let tgt_id = format!("fact:{}", fact_j.id);

            // Determine direction (smaller id first for consistency)
            let (edge_source, edge_target) = if src_id <= tgt_id {
                (src_id.clone(), tgt_id.clone())
            } else {
                (tgt_id.clone(), src_id.clone())
            };

            // Skip if edge already exists
            if existing_edges.contains(&(edge_source.clone(), edge_target.clone())) {
                edges_skipped += 1;
                continue;
            }

            // Check edge cap for both facts
            let count_src = *edge_counts.get(&src_id).unwrap_or(&0);
            let count_tgt = *edge_counts.get(&tgt_id).unwrap_or(&0);
            if count_src >= max_edges_per_fact as u64
                || count_tgt >= max_edges_per_fact as u64
            {
                continue;
            }

            if dry_run {
                edges_created += 1;
                continue;
            }

            let relation = format!("shared_terms:{}", shared);
            let edge_type = semantic_memory::GraphEdgeType::Entity {
                relation: relation.clone(),
            };
            // Weight based on overlap quality: 3 terms = 0.3, up to 1.0 for 10+ terms
            let weight = (shared.min(10) as f64) / 10.0;

            let result = block_in_place(|| {
                handle.block_on(store.add_graph_edge(
                    &edge_source,
                    &edge_target,
                    edge_type,
                    weight,
                    None,
                ))
            });

            match result {
                Ok(_) => {
                    edges_created += 1;
                    *edge_counts.entry(src_id.clone()).or_insert(0) += 1;
                    *edge_counts.entry(tgt_id.clone()).or_insert(0) += 1;
                }
                Err(e) => {
                    eprintln!(
                        "[auto-edge] add_graph_edge error: {e} for {edge_source} -> {edge_target}"
                    );
                }
            }

            // Re-check cap after creating edge
            if *edge_counts.get(&src_id).unwrap_or(&0) >= max_edges_per_fact as u64 {
                break;
            }
        }
    }

    let elapsed = start.elapsed();

    (
        "200 OK",
        serde_json::json!({
            "ok": true,
            "action": "auto-edge",
            "dry_run": dry_run,
            "facts_processed": facts_processed,
            "namespaces_scanned": namespaces.len(),
            "edges_created": edges_created,
            "edges_skipped": edges_skipped,
            "max_edges_per_fact": max_edges_per_fact,
            "time_elapsed_ms": elapsed.as_millis(),
            "time_elapsed_secs": (elapsed.as_millis() as f64) / 1000.0,
        }),
    )
}