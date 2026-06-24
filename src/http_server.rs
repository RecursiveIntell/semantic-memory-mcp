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
                    serde_json::json!({
                        "result_id": r.source.result_id(),
                        "content": r.content,
                        "score": r.score,
                        "cosine_similarity": r.cosine_similarity,
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
            (
                "200 OK",
                serde_json::json!({
                    "ok": true,
                    "results": final_results,
                    "count": count,
                    "reranked": do_rerank,
                }),
            )
        }
        Err(e) => (
            "500 Internal Server Error",
            serde_json::json!({"ok": false, "error": format!("search error: {e}")}),
        ),
    }
}

/// Handle /search-routed: routing-aware search for complex queries.
///
/// Accepts a `query_class` field (A/B/C/D/E) from the Python classifier:
/// - D (SYNTHESIS): increases top_k to gather more candidates
/// - C (CONTRADICTION): uses exact search profile
/// - A/B/E: identical to /search (early return, no overhead)
fn handle_search_routed(
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
    let base_top_k = params.get("top_k").and_then(|v| v.as_u64()).unwrap_or(12) as usize;
    let query_class = params.get("query_class").and_then(|v| v.as_str()).unwrap_or("A");
    let namespaces: Option<Vec<String>> = params
        .get("namespaces")
        .and_then(|v| serde_json::from_value(v.clone()).ok());

    if query.is_empty() {
        return (
            "400 Bad Request",
            serde_json::json!({"ok": false, "error": "missing 'query' field"}),
        );
    }

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
                    serde_json::json!({
                        "result_id": r.source.result_id(),
                        "content": r.content,
                        "score": r.score,
                        "cosine_similarity": r.cosine_similarity,
                    })
                })
                .collect();
            let count = json_results.len();
            (
                "200 OK",
                serde_json::json!({
                    "ok": true,
                    "results": json_results,
                    "count": count,
                    "query_class": query_class,
                    "routed": true,
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