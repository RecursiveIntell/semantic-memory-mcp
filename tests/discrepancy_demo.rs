#![cfg(feature = "orchestration")]
//! Discrepancy demo: the traced route surfaces a degradation that raw
//! `sm_search` has no concept of. This is the Wave-5 kill/keep gate.

use semantic_memory::{MemoryConfig, MemoryStore, MockEmbedder};
use semantic_memory_mcp::query_orchestration::query_with_warnings;

fn store(dir: &std::path::Path) -> MemoryStore {
    let mut cfg = MemoryConfig::default();
    cfg.base_dir = dir.join("mem");
    let dims = cfg.embedding.dimensions;
    MemoryStore::open_with_embedder(cfg, Box::new(MockEmbedder::new(dims))).unwrap()
}

#[tokio::test]
async fn traced_route_reports_degradation_raw_search_hides() {
    let dir = tempfile::tempdir().unwrap();
    let s = store(dir.path());
    s.add_fact(
        "scope-a",
        "the deploy happened in March 2019",
        Some("test"),
        None,
    )
    .await
    .unwrap();

    // Raw search: returns hits only, no degradation/warning channel exists.
    let raw = s
        .search(
            "what changed before 2020",
            Some(5),
            Some(&["scope-a"]),
            None,
        )
        .await
        .expect("raw search should succeed");

    // Traced route: classify -> temporal range -> explicit downgrade warning.
    let traced = query_with_warnings(s, "what changed before 2020", "scope-a")
        .await
        .expect("orchestrated query should succeed");

    eprintln!(
        "DISCREPANCY raw_hits={} traced_hits={} traced_warnings={:?}",
        raw.len(),
        traced.results.len(),
        traced.warnings
    );

    assert!(
        !traced.warnings.is_empty(),
        "traced route must surface a degradation warning the raw path cannot; got none"
    );
}
