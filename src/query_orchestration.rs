//! Additive orchestration path: route a query through `knowledge-runtime` and
//! surface its structured degradation warnings.
//!
//! This module is gated behind the `orchestration` feature (already declared in
//! `Cargo.toml` and included in the `search` feature set). It is additive: it
//! does not change the existing server query path.

use knowledge_runtime::adapters::semantic_memory::SemanticMemoryAdapter;
use knowledge_runtime::{KnowledgeRuntime, RuntimeConfig, Scope};
use semantic_memory::{MemoryStore, SearchResult};

/// Result of an orchestrated query: merged results plus human-readable
/// degradation warnings from the runtime's `QueryTrace`.
#[derive(Debug, Clone)]
pub struct OrchestratedQuery {
    pub results: Vec<SearchResult>,
    pub warnings: Vec<String>,
}

/// Run a query through `KnowledgeRuntime::query_with_trace` and return results
/// with the trace's degradation warnings rendered as strings.
///
/// `namespace` is used as the default scope. Strict modes are off, so temporal
/// downgrades and partial scope are surfaced as warnings rather than errors.
pub async fn query_with_warnings(
    store: MemoryStore,
    query: &str,
    namespace: &str,
) -> Result<OrchestratedQuery, String> {
    let config = RuntimeConfig {
        default_scope: Scope::new(namespace),
        query: Default::default(),
        entity: Default::default(),
        projection: Default::default(),
        strict_temporal: false,
        strict_scope: false,
    };
    let adapter = SemanticMemoryAdapter::new(store);
    let runtime = KnowledgeRuntime::new(config, adapter).map_err(|e| format!("{e}"))?;
    let (results, trace) = runtime
        .query_with_trace(query, None, None)
        .await
        .map_err(|e| format!("{e}"))?;
    let warnings = trace
        .warnings
        .iter()
        .map(|w| format!("{w:?}"))
        .collect::<Vec<_>>();
    Ok(OrchestratedQuery { results, warnings })
}
