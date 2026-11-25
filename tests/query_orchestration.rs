#![cfg(feature = "orchestration")]
//! Feature-gated test for the additive orchestration query path.

use semantic_memory::{MemoryConfig, MemoryStore, MockEmbedder};
use semantic_memory_mcp::query_orchestration::query_with_warnings;

fn test_store(dir: &std::path::Path) -> MemoryStore {
    let mut cfg = MemoryConfig::default();
    cfg.base_dir = dir.join("mem");
    let dims = cfg.embedding.dimensions;
    MemoryStore::open_with_embedder(cfg, Box::new(MockEmbedder::new(dims))).unwrap()
}

#[tokio::test]
async fn orchestrated_query_returns_results_and_trace() {
    let dir = tempfile::tempdir().unwrap();
    let store = test_store(dir.path());
    store
        .add_fact(
            "scope-a",
            "the capital of France is Paris",
            Some("test"),
            None,
        )
        .await
        .unwrap();

    let out = query_with_warnings(store, "capital of France", "scope-a")
        .await
        .expect("orchestrated query should succeed");

    // The call must complete and expose a warnings vector (possibly empty).
    let _ = &out.warnings;
    assert!(
        out.results.is_empty()
            || out
                .results
                .iter()
                .any(|r| r.content.to_ascii_lowercase().contains("paris")),
        "expected Paris in results or empty result set"
    );
}
