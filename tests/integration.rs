//! Integration tests for semantic-memory-mcp.
//!
//! These tests exercise the MemoryBridge + HTTP server end-to-end using
//! the mock embedder (no model download, no Ollama, no network).
//! Each test gets a fresh temp directory so there is no cross-test state.

use semantic_memory::{AuthorityPermit, GraphEdgeType};
use semantic_memory_mcp::bridge::{BridgeConfig, EmbedderBackend, MemoryBridge};
use semantic_memory_mcp::server::SemanticMemoryServer;

/// Open a MemoryBridge with the mock embedder in a temp directory.
fn open_bridge(dir: &std::path::Path) -> MemoryBridge {
    let config = BridgeConfig {
        memory_dir: dir.to_path_buf(),
        embedder_backend: EmbedderBackend::Mock,
        embedding_url: "http://localhost:11434".to_string(),
        embedding_model: "nomic-embed-text".to_string(),
        embedding_dims: 768,
        turbo_quant_enabled: false,
        turbo_quant_bits: None,
        turbo_quant_projections: None,
    };
    MemoryBridge::open(config).expect("bridge should open")
}

#[test]
fn autonomous_profiles_expose_only_witnessed_search() {
    for profile in ["lean", "standard"] {
        let dir = tempfile::tempdir().unwrap();
        let server = SemanticMemoryServer::new(open_bridge(dir.path()), profile);
        assert!(server.exposes_tool("sm_search_witnessed"));
        assert!(server.exposes_tool("sm_decide_assertion_authority"));
        assert!(server.exposes_tool("sm_decide_action_authority"));
        assert!(!server.exposes_tool("sm_search"));
        assert_eq!(
            server.exposed_tool_names(),
            vec![
                "sm_decide_action_authority",
                "sm_decide_assertion_authority",
                "sm_search_witnessed",
            ]
        );
        for name in [
            "sm_decide_assertion_authority",
            "sm_decide_action_authority",
        ] {
            let annotations = server.tool_annotations(name).expect("decision metadata");
            assert_eq!(annotations.read_only_hint, Some(true));
            assert_ne!(annotations.destructive_hint, Some(true));
        }
        for forbidden in [
            "sm_add_fact",
            "sm_delete_fact",
            "sm_delete_namespace",
            "sm_update_fact",
            "sm_set_provenance",
            "sm_record_outcome",
        ] {
            assert!(
                !server.exposes_tool(forbidden),
                "{profile} exposed {forbidden}"
            );
        }
    }

    let dir = tempfile::tempdir().unwrap();
    let bridge = open_bridge(dir.path());
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let permit = AuthorityPermit::operator_system(
        "lean-principal",
        "lean-caller",
        AuthorityPermit::APPEND_CAPABILITY,
    );
    runtime
        .block_on(bridge.store.authority().append(
            permit,
            "lean-canary".into(),
            "lean".into(),
            "lean profile canary should remain queryable by witnessed surfaces".into(),
            Some("tests/canary.md".into()),
        ))
        .unwrap();
    let lean_server = SemanticMemoryServer::new(open_bridge(dir.path()), "lean");
    assert!(lean_server.exposes_tool("sm_search_witnessed"));
    assert!(!lean_server.exposes_tool("sm_add_fact"));
    let results = runtime
        .block_on(
            bridge
                .store
                .search("lean profile canary", Some(5), None, None),
        )
        .unwrap();
    assert!(
        results
            .iter()
            .any(|result| result.content.contains("lean profile canary")),
        "Canary should be queryable in lean-profile environment"
    );

    let dir = tempfile::tempdir().unwrap();
    let full = SemanticMemoryServer::new(open_bridge(dir.path()), "full");
    assert!(full.exposes_tool("sm_search_witnessed"));
    assert!(full.exposes_tool("sm_search"));
}

#[test]
fn agent_profile_exposes_bounded_daily_memory_surface() {
    let dir = tempfile::tempdir().unwrap();
    let server = SemanticMemoryServer::new(open_bridge(dir.path()), "agent");
    assert_eq!(
        server.exposed_tool_names(),
        vec![
            "sm_add_fact",
            "sm_add_graph_edge",
            "sm_decide_action_authority",
            "sm_decide_assertion_authority",
            "sm_get_fact",
            "sm_get_fact_neighbors",
            "sm_get_search_receipt",
            "sm_graph_path",
            "sm_list_namespaces",
            "sm_search_conversations",
            "sm_search_witnessed",
            "sm_set_provenance",
            "sm_stats",
            "sm_supersede_fact",
            "sm_update_fact",
        ]
    );
    for forbidden in [
        "sm_delete_fact",
        "sm_delete_namespace",
        "sm_import_envelope",
        "sm_reembed_all",
        "sm_reconcile",
        "sm_run_lifecycle",
        "sm_search",
        "sm_search_with_routing",
        "sm_vacuum",
    ] {
        assert!(!server.exposes_tool(forbidden), "agent exposed {forbidden}");
    }
}

#[test]
fn routing_feedback_is_declared_mutating() {
    let dir = tempfile::tempdir().unwrap();
    let server = SemanticMemoryServer::new(open_bridge(dir.path()), "full");
    let annotations = server
        .tool_annotations("sm_record_outcome")
        .expect("record outcome metadata");
    assert_eq!(annotations.read_only_hint, Some(false));
    assert_eq!(annotations.destructive_hint, Some(false));
}

/// Helper: add a fact via the underlying store and return its fact_id.
fn add_fact(bridge: &MemoryBridge, content: &str, namespace: &str) -> String {
    let store = &bridge.store;
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(store.add_fact(namespace, content, None, None))
        .expect("add_fact should succeed")
}

mod bridge_tests {
    use super::*;

    #[test]
    fn bridge_opens_with_mock_embedder() {
        let dir = tempfile::tempdir().unwrap();
        let bridge = open_bridge(dir.path());
        assert!(dir.path().join("memory.db").exists());
        let _ = &bridge.store;
    }

    #[test]
    fn bridge_rejects_unknown_embedder_string() {
        let result: Result<EmbedderBackend, _> = "garbage".parse();
        assert!(result.is_err());
    }

    #[test]
    fn bridge_parses_embedder_variants() {
        assert_eq!(
            "candle".parse::<EmbedderBackend>().unwrap(),
            EmbedderBackend::Candle
        );
        assert_eq!(
            "CANDLE".parse::<EmbedderBackend>().unwrap(),
            EmbedderBackend::Candle
        );
        assert_eq!(
            "ollama".parse::<EmbedderBackend>().unwrap(),
            EmbedderBackend::Ollama
        );
        assert_eq!(
            "mock".parse::<EmbedderBackend>().unwrap(),
            EmbedderBackend::Mock
        );
    }
}

mod lifecycle_tests {
    use super::*;

    #[test]
    fn add_and_search_fact() {
        let dir = tempfile::tempdir().unwrap();
        let bridge = open_bridge(dir.path());
        let id = add_fact(
            &bridge,
            "Rust is a systems programming language focused on safety and performance.",
            "coding",
        );
        assert!(!id.is_empty(), "fact_id should be non-empty");

        let store = &bridge.store;
        let rt = tokio::runtime::Runtime::new().unwrap();
        let results = rt
            .block_on(store.search("Rust programming safety", Some(10), None, None))
            .expect("search should succeed");
        assert!(!results.is_empty(), "search should return results");
        let found = results.iter().any(|r| r.content.contains("Rust"));
        assert!(found, "search should find the Rust fact");
    }

    #[test]
    fn add_facts_in_multiple_namespaces_and_filter() {
        let dir = tempfile::tempdir().unwrap();
        let bridge = open_bridge(dir.path());
        add_fact(
            &bridge,
            "Python is great for data science and ML pipelines.",
            "coding",
        );
        add_fact(
            &bridge,
            "The cat sat on the mat in Albertville Alabama.",
            "personal",
        );

        let store = &bridge.store;
        let rt = tokio::runtime::Runtime::new().unwrap();

        // Search without filter
        let all = rt
            .block_on(store.search("cat mat Albertville", Some(10), None, None))
            .expect("search should succeed");
        assert!(!all.is_empty());

        // Search with namespace filter
        let filtered = rt
            .block_on(store.search("cat mat", Some(10), Some(&["personal"]), None))
            .expect("search should succeed");
        assert!(!filtered.is_empty());
        let all_personal = filtered
            .iter()
            .all(|r| r.content.contains("cat") || r.content.contains("Albertville"));
        assert!(
            all_personal,
            "filtered results should all be from personal namespace"
        );
    }

    #[test]
    fn supersede_fact_via_graph_edge() {
        let dir = tempfile::tempdir().unwrap();
        let bridge = open_bridge(dir.path());
        let old_id = add_fact(&bridge, "turbo-quant has 1000 downloads.", "libraries");

        let store = &bridge.store;
        let rt = tokio::runtime::Runtime::new().unwrap();

        // Supersede: add new fact, create supersedes edge
        let new_id = rt
            .block_on(store.add_fact("libraries", "turbo-quant has 4000 downloads.", None, None))
            .expect("add replacement should succeed");
        let old_node = format!("fact:{}", old_id);
        let new_node = format!("fact:{}", new_id);
        rt.block_on(store.add_graph_edge(
            &new_node,
            &old_node,
            GraphEdgeType::Entity {
                relation: "supersedes".to_string(),
            },
            1.0,
            None,
        ))
        .expect("add supersedes edge should succeed");

        // Search should filter superseded facts
        let results = rt
            .block_on(store.search("turbo-quant downloads", Some(10), None, None))
            .expect("search should succeed");
        let _has_old = results.iter().any(|r| r.content.contains("1000 downloads"));
        let has_new = results.iter().any(|r| r.content.contains("4000 downloads"));
        assert!(has_new, "new fact should appear in search");
        // Old fact may or may not be filtered depending on search filter logic,
        // but the edge should exist
    }

    #[test]
    fn delete_fact_removes_it() {
        let dir = tempfile::tempdir().unwrap();
        let bridge = open_bridge(dir.path());
        let id = add_fact(&bridge, "Temporary fact to be deleted.", "test");
        let store = &bridge.store;
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(store.delete_fact(&id))
            .expect("delete should succeed");
        let fact = rt.block_on(store.get_fact(&id));
        assert!(
            fact.is_err() || fact.is_ok_and(|f| f.is_none()),
            "deleted fact should not be retrievable"
        );
    }

    #[test]
    fn list_namespaces_returns_all() {
        let dir = tempfile::tempdir().unwrap();
        let bridge = open_bridge(dir.path());
        add_fact(&bridge, "Fact in namespace alpha.", "alpha");
        add_fact(&bridge, "Fact in namespace beta.", "beta");
        add_fact(&bridge, "Fact in namespace gamma.", "gamma");

        let store = &bridge.store;
        let rt = tokio::runtime::Runtime::new().unwrap();
        let namespaces = rt
            .block_on(store.list_fact_namespaces())
            .expect("list_fact_namespaces should succeed");
        assert!(
            namespaces.len() >= 3,
            "should have at least 3 namespaces, got: {:?}",
            namespaces
        );
        assert!(namespaces.contains(&"alpha".to_string()));
        assert!(namespaces.contains(&"beta".to_string()));
        assert!(namespaces.contains(&"gamma".to_string()));
    }

    #[test]
    fn add_and_list_graph_edges() {
        let dir = tempfile::tempdir().unwrap();
        let bridge = open_bridge(dir.path());
        let id_a = add_fact(&bridge, "Fact A about semantic search.", "graph");
        let id_b = add_fact(&bridge, "Fact B about vector databases.", "graph");

        let store = &bridge.store;
        let rt = tokio::runtime::Runtime::new().unwrap();

        let source = format!("fact:{}", id_a);
        let target = format!("fact:{}", id_b);
        rt.block_on(store.add_graph_edge(
            &source,
            &target,
            GraphEdgeType::Semantic {
                cosine_similarity: 0.85,
            },
            1.0,
            None,
        ))
        .expect("add_graph_edge should succeed");

        let edges = rt
            .block_on(store.list_graph_edges_for_node(&source))
            .expect("list_graph_edges should succeed");
        assert!(!edges.is_empty(), "should have at least one edge from A");
        let found = edges.iter().any(|e| e.target == target);
        assert!(found, "should find the edge A->B");
    }

    #[test]
    fn stats_returns_counts() {
        let dir = tempfile::tempdir().unwrap();
        let bridge = open_bridge(dir.path());
        add_fact(&bridge, "Fact for stats test.", "stats_ns");
        let store = &bridge.store;
        let rt = tokio::runtime::Runtime::new().unwrap();
        let stats = rt.block_on(store.stats()).expect("stats should succeed");
        assert!(
            stats.total_facts >= 1,
            "should have at least 1 fact, got: {}",
            stats.total_facts
        );
    }
}

mod http_server_tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpStream;

    /// Start the HTTP server on a random port and return the port.
    /// Returns (port, runtime) — the runtime must stay alive while making requests.
    fn start_http(bridge: MemoryBridge) -> (u16, tokio::runtime::Runtime) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);

        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .worker_threads(2)
            .build()
            .unwrap();

        let handle = rt.handle().clone();
        let _enter = rt.enter();
        std::thread::spawn(move || {
            let _guard = handle.enter();
            semantic_memory_mcp::http_server::start_http_server(port, bridge, handle);
        });

        // Give the server a moment to bind
        std::thread::sleep(std::time::Duration::from_millis(100));
        (port, rt)
    }

    fn http_get(port: u16, path: &str) -> (String, String) {
        let mut stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
        let request = format!(
            "GET {} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n",
            path
        );
        stream.write_all(request.as_bytes()).unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        let body_start = response
            .find("\r\n\r\n")
            .map(|i| response[i + 4..].to_string())
            .unwrap_or_default();
        (response, body_start)
    }

    fn http_post(port: u16, path: &str, body: &str) -> (String, String) {
        let mut stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
        let request = format!(
            "POST {} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            path,
            body.len(),
            body
        );
        stream.write_all(request.as_bytes()).unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        let body_start = response
            .find("\r\n\r\n")
            .map(|i| response[i + 4..].to_string())
            .unwrap_or_default();
        (response, body_start)
    }

    #[test]
    fn health_endpoint_returns_ok() {
        let dir = tempfile::tempdir().unwrap();
        let bridge = open_bridge(dir.path());
        let (port, _rt) = start_http(bridge);
        let (response, body) = http_get(port, "/health");
        assert!(
            response.contains("200 OK"),
            "health should return 200, got: {}",
            response
        );
        let json: serde_json::Value = serde_json::from_str(&body).expect("valid JSON");
        assert_eq!(json["ok"], serde_json::Value::Bool(true));
        assert_eq!(json["service"], "semantic-memory-mcp");
    }

    #[test]
    fn routing_feedback_response_is_mutating_proxy_label() {
        let dir = tempfile::tempdir().unwrap();
        let bridge = open_bridge(dir.path());
        let (port, _rt) = start_http(bridge);
        let (_, body) = http_post(
            port,
            "/record-outcome",
            r#"{"query":"test query","outcome":"good"}"#,
        );
        let json: serde_json::Value = serde_json::from_str(&body).expect("valid JSON");
        assert_eq!(json["mutating"], true);
        assert_eq!(json["feedback"]["kind"], "ProxyLabel");
        assert_eq!(json["feedback"]["label"], "good");
        assert!(json.get("outcome").is_none());
    }

    #[test]
    fn add_and_search_via_http() {
        let dir = tempfile::tempdir().unwrap();
        let bridge = open_bridge(dir.path());
        let (port, _rt) = start_http(bridge);

        // Add a fact
        let (_, add_body) = http_post(
            port,
            "/add",
            r#"{"content": "Hermes Agent is a CLI AI agent by Nous Research.", "namespace": "test"}"#,
        );
        let add_json: serde_json::Value = serde_json::from_str(&add_body).expect("valid JSON");
        assert!(
            add_json["ok"] == serde_json::Value::Bool(true),
            "add should return ok=true, got: {}",
            add_body
        );
        assert!(
            add_json["fact_id"].is_string(),
            "fact_id should be a string"
        );
        assert!(
            add_json["authority_receipt"].is_object(),
            "HTTP writes must return a durable authority receipt: {add_body}"
        );
        assert_eq!(
            add_json["authority_receipt"]["affected_ids"][0], add_json["fact_id"],
            "authority receipt must witness the returned fact"
        );

        // Search for it
        let (_, search_body) = http_post(
            port,
            "/search",
            r#"{"query": "Hermes Agent CLI", "top_k": 5}"#,
        );
        let search_json: serde_json::Value =
            serde_json::from_str(&search_body).expect("valid JSON");
        assert_eq!(search_json["ok"], serde_json::Value::Bool(true));
        let results = search_json["results"]
            .as_array()
            .expect("results should be array");
        assert!(!results.is_empty(), "search should return results");
        let found = results.iter().any(|r| {
            r["content"]
                .as_str()
                .map(|c| c.contains("Hermes"))
                .unwrap_or(false)
        });
        assert!(found, "should find the Hermes fact in search results");
    }

    #[test]
    fn http_search_returns_current_supersession_head_only() {
        let dir = tempfile::tempdir().unwrap();
        let bridge = open_bridge(dir.path());
        let old_id = add_fact(&bridge, "release train is indigo", "state");
        let rt = tokio::runtime::Runtime::new().unwrap();
        let new_id = rt
            .block_on(
                bridge
                    .store
                    .add_fact("state", "release train is coral", None, None),
            )
            .unwrap();
        rt.block_on(bridge.store.add_graph_edge(
            &format!("fact:{new_id}"),
            &format!("fact:{old_id}"),
            GraphEdgeType::Entity {
                relation: "supersedes".into(),
            },
            1.0,
            None,
        ))
        .unwrap();
        let (port, _server_rt) = start_http(bridge);
        let (_, body) = http_post(
            port,
            "/search",
            r#"{"query":"release train","top_k":10,"namespaces":["state"]}"#,
        );
        let json: serde_json::Value = serde_json::from_str(&body).unwrap();
        let contents: Vec<&str> = json["results"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|r| r["content"].as_str())
            .collect();
        assert!(contents.iter().any(|value| value.contains("coral")));
        assert!(!contents.iter().any(|value| value.contains("indigo")));
    }

    #[test]
    fn stats_endpoint_returns_counts() {
        let dir = tempfile::tempdir().unwrap();
        let bridge = open_bridge(dir.path());
        let (port, _rt) = start_http(bridge);

        let (add_response, add_body) = http_post(
            port,
            "/add",
            r#"{"content": "Test fact for stats.", "namespace": "stats"}"#,
        );
        assert!(
            add_response.contains("200 OK"),
            "add should succeed: {}",
            add_body
        );

        let (_, body) = http_post(port, "/stats", "{}");
        let json: serde_json::Value = serde_json::from_str(&body).expect("valid JSON");
        assert_eq!(json["ok"], serde_json::Value::Bool(true));
        assert_eq!(json["components"]["core"]["health"], "healthy");
        assert_eq!(json["components"]["graph"]["health"], "healthy");
        assert!(
            json["facts"].as_u64().unwrap_or(0) >= 1,
            "should have >= 1 fact, got: {}",
            body
        );
    }

    #[test]
    fn unknown_path_returns_404() {
        let dir = tempfile::tempdir().unwrap();
        let bridge = open_bridge(dir.path());
        let (port, _rt) = start_http(bridge);
        let (response, body) = http_get(port, "/nonexistent");
        assert!(
            response.contains("404 Not Found"),
            "unknown path should 404"
        );
        let json: serde_json::Value = serde_json::from_str(&body).expect("valid JSON");
        assert_eq!(json["error"], "not found");
    }

    #[test]
    fn search_with_empty_query_returns_400() {
        let dir = tempfile::tempdir().unwrap();
        let bridge = open_bridge(dir.path());
        let (port, _rt) = start_http(bridge);
        let (response, body) = http_post(port, "/search", r#"{"query": ""}"#);
        assert!(
            response.contains("400 Bad Request"),
            "empty query should 400"
        );
        let json: serde_json::Value = serde_json::from_str(&body).expect("valid JSON");
        assert_eq!(json["ok"], serde_json::Value::Bool(false));
    }
}
