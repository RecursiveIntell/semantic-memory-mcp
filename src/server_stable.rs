//! Compile-time stable MCP router.
//!
//! Selected only by `--no-default-features --features stable`. Advanced,
//! preview, parser, claim-ledger, routing, decoder, and admin tools are absent
//! from this compilation unit and cannot be re-enabled at runtime.

use crate::bridge::MemoryBridge;
use crate::tools::*;
use rmcp::{
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    tool, tool_handler, tool_router, ErrorData, Json, ServerHandler,
};
use schemars::JsonSchema;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::runtime::Handle;

static WITNESS_REQUEST_SEQUENCE: AtomicU64 = AtomicU64::new(1);

/// Schema-backed structured output shared by heterogeneous MCP tools.
///
/// Existing object responses retain their top-level wire shape through
/// `flatten`. Scalar and array results are wrapped under `value`, because MCP
/// requires structured tool outputs to have an object root schema.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema)]
pub struct StructuredOutput {
    #[serde(flatten)]
    pub fields: std::collections::BTreeMap<String, serde_json::Value>,
}

impl StructuredOutput {
    fn from_value(value: serde_json::Value) -> Self {
        let fields = match value {
            serde_json::Value::Object(map) => map.into_iter().collect(),
            value => std::collections::BTreeMap::from([("value".to_string(), value)]),
        };
        Self { fields }
    }
}

fn structured_output(value: serde_json::Value) -> Json<StructuredOutput> {
    Json(StructuredOutput::from_value(value))
}

/// Typed output for sm_stats — provides outputSchema with type: "object" for MCP.
#[derive(Debug, serde::Serialize, serde::Deserialize, JsonSchema)]
pub struct StatsOutput {
    pub ok: bool,
    pub components: serde_json::Value,
    pub facts: Option<u64>,
    pub chunks: Option<u64>,
    pub documents: Option<u64>,
    pub sessions: Option<u64>,
    pub messages: Option<u64>,
    pub graph_edges: Option<usize>,
    pub db_size_bytes: Option<u64>,
    pub db_size_mb: Option<f64>,
    pub embedding_model: Option<String>,
    pub embedding_dimensions: Option<usize>,
}

pub struct SemanticMemoryServer {
    bridge: Arc<MemoryBridge>,
    tool_router: ToolRouter<Self>,
}

impl SemanticMemoryServer {
    pub fn new(bridge: MemoryBridge, tool_profile: &str) -> Self {
        if tool_profile != "stable" {
            panic!("compile-time stable build accepts only --tool-profile stable; got '{tool_profile}'");
        }
        let server = Self {
            bridge: Arc::new(bridge),
            tool_router: Self::tool_router(),
        };
        debug_assert_eq!(server.tool_router.list_all().len(), 13);
        eprintln!("Tool profile: stable (13 compile-time tools visible)");
        server
    }

    pub fn exposes_tool(&self, name: &str) -> bool {
        self.tool_router
            .list_all()
            .iter()
            .any(|tool| tool.name == name)
    }

    pub fn exposed_tool_names(&self) -> Vec<String> {
        let mut names: Vec<_> = self
            .tool_router
            .list_all()
            .into_iter()
            .map(|tool| tool.name.into_owned())
            .collect();
        names.sort();
        names
    }

    pub fn tool_annotations(&self, name: &str) -> Option<rmcp::model::ToolAnnotations> {
        self.tool_router
            .list_all()
            .into_iter()
            .find(|tool| tool.name == name)
            .and_then(|tool| tool.annotations)
    }

    fn decide_governed_authority(
        &self,
        params: GovernedDecisionParams,
        purpose: semantic_memory::GovernedAccessPurposeV1,
    ) -> Result<Json<StructuredOutput>, ErrorData> {
        use semantic_memory::{
            AudienceV1, CallerPrincipalV1, DelegationElevationLeaseV1, GovernedAccessPurposeV1,
            GovernedAccessRequestV1, NamespaceScopeV1, SubjectPrincipalV1,
        };

        let GovernedDecisionParams {
            fact_id,
            caller,
            subject,
            audiences,
            scope,
            delegation_or_elevation,
        } = params;
        let caller = CallerPrincipalV1::new(caller)
            .map_err(|error| ErrorData::invalid_params(error, None))?;
        let subject = SubjectPrincipalV1::new(subject)
            .map_err(|error| ErrorData::invalid_params(error, None))?;
        let scope = NamespaceScopeV1 {
            namespace: scope.namespace,
            domain: scope.domain,
            workspace_id: scope.workspace_id,
            repo_id: scope.repo_id,
        };
        let mut request =
            GovernedAccessRequestV1::for_principals(caller, subject, audiences, purpose, scope);
        if let Some(lease) = delegation_or_elevation {
            let lease_scope = NamespaceScopeV1 {
                namespace: lease.scope.namespace,
                domain: lease.scope.domain,
                workspace_id: lease.scope.workspace_id,
                repo_id: lease.scope.repo_id,
            };
            let purposes = lease
                .purposes
                .into_iter()
                .map(|purpose| match purpose {
                    GovernedAccessPurposeParam::Recall => GovernedAccessPurposeV1::Recall,
                    GovernedAccessPurposeParam::Assertion => GovernedAccessPurposeV1::Assertion,
                    GovernedAccessPurposeParam::Action => GovernedAccessPurposeV1::Action,
                    GovernedAccessPurposeParam::Export => GovernedAccessPurposeV1::Export,
                    GovernedAccessPurposeParam::Replay => GovernedAccessPurposeV1::Replay,
                    GovernedAccessPurposeParam::Admin => GovernedAccessPurposeV1::Admin,
                })
                .collect();
            request = request.with_delegation_or_elevation(DelegationElevationLeaseV1 {
                lease_id: lease.lease_id,
                delegator: SubjectPrincipalV1::new(lease.delegator)
                    .map_err(|error| ErrorData::invalid_params(error, None))?,
                delegatee: CallerPrincipalV1::new(lease.delegatee)
                    .map_err(|error| ErrorData::invalid_params(error, None))?,
                purposes,
                scope: lease_scope,
                audience: AudienceV1::new(lease.audiences),
                expires_at: lease.expires_at,
                revoked: lease.revoked,
                elevation: lease.elevation,
            });
        }

        let fact_id = fact_id
            .strip_prefix("fact:")
            .unwrap_or(&fact_id)
            .to_string();
        let access = tokio::task::block_in_place(|| {
            Handle::current().block_on(
                self.bridge
                    .store
                    .authority()
                    .get_fact_governed(&fact_id, request),
            )
        })
        .map_err(|error| {
            ErrorData::internal_error(format!("governed authority decision error: {error}"), None)
        })?;

        // Deliberately serialize only the canonical typed receipt. `access.fact`
        // and `access.origin` are never part of this MCP decision surface.
        let value = serde_json::to_value(&access.decision).map_err(|error| {
            ErrorData::internal_error(
                format!("decision receipt serialization error: {error}"),
                None,
            )
        })?;
        Ok(structured_output(value))
    }

    #[cfg(feature = "claim-integration")]
    fn trust_for_fact(&self, bare_fact_id: &str) -> String {
        self.claim_trust
            .lock()
            .unwrap()
            .trust_for_fact(bare_fact_id)
    }

    #[cfg(not(feature = "claim-integration"))]
    fn trust_for_fact(&self, _bare_fact_id: &str) -> String {
        "persisted_unjudged".to_string()
    }

    #[cfg(feature = "claim-integration")]
    fn auto_link_fact_to_claims(&self, bare_fact_id: &str, content: &str) {
        self.claim_trust
            .lock()
            .unwrap()
            .auto_link_content(bare_fact_id, content);
    }

    #[cfg(not(feature = "claim-integration"))]
    fn auto_link_fact_to_claims(&self, _bare_fact_id: &str, _content: &str) {}

    fn enrich_results_with_trust(&self, results: &mut [serde_json::Value]) {
        for result in results.iter_mut() {
            let bare_fact_id = result
                .get("memory_id")
                .and_then(|v| v.as_str())
                .map(|s| s.strip_prefix("fact:").unwrap_or(s).to_string());
            let Some(bare_fact_id) = bare_fact_id else {
                continue;
            };
            if let Some(content) = result.get("content").and_then(|v| v.as_str()) {
                self.auto_link_fact_to_claims(&bare_fact_id, content);
            }
            if let Some(obj) = result.as_object_mut() {
                obj.insert(
                    "trust".to_string(),
                    serde_json::Value::String(self.trust_for_fact(&bare_fact_id)),
                );
            }
        }
    }
}

fn load_superseded_targets(
    store: &semantic_memory::MemoryStore,
) -> Result<HashSet<String>, ErrorData> {
    let edges =
        tokio::task::block_in_place(|| Handle::current().block_on(store.list_all_graph_edges()))
            .map_err(|e| {
                ErrorData::internal_error(format!("Failed to load graph edges: {e}"), None)
            })?;
    let mut targets = HashSet::new();
    for edge in edges {
        let parsed_type = edge
            .edge_type_parsed
            .clone()
            .or_else(|| serde_json::from_str(&edge.edge_type).ok());
        if let Some(semantic_memory::GraphEdgeType::Entity { relation }) = parsed_type {
            if relation == "supersedes" {
                targets.insert(edge.target);
            }
        }
    }
    Ok(targets)
}

fn json_to_output(value: &serde_json::Value) -> Result<Json<StructuredOutput>, ErrorData> {
    Ok(structured_output(value.clone()))
}

fn mcp_receipt_id(tool_name: &str) -> String {
    format!("mcp-receipt:{tool_name}:{}", uuid::Uuid::new_v4())
}

fn mcp_now_iso() -> String {
    chrono::Utc::now().to_rfc3339()
}

fn mcp_receipt(tool_name: &str) -> serde_json::Value {
    serde_json::json!({
        "receipt_id": mcp_receipt_id(tool_name),
        "recorded_at": mcp_now_iso(),
        "tool": tool_name,
    })
}

fn witnessed_injectible_fact(
    store: &semantic_memory::MemoryStore,
    result: semantic_memory::SearchResult,
    receipt_ref: &str,
) -> Result<Option<serde_json::Value>, ErrorData> {
    let semantic_memory::SearchSource::Fact { fact_id, .. } = &result.source else {
        return Ok(None);
    };
    let fact = tokio::task::block_in_place(|| Handle::current().block_on(store.get_fact(fact_id)))
        .map_err(|e| {
            ErrorData::internal_error(
                format!("witnessed fact provenance hydration failed: {e}"),
                None,
            )
        })?;
    let Some(fact) = fact else {
        return Ok(None);
    };
    let Some(source) = fact.source.filter(|source| !source.trim().is_empty()) else {
        return Ok(None);
    };
    let memory_id = format!("fact:{}", fact.id);
    Ok(Some(serde_json::json!({
        "memory_id": memory_id,
        "result_id": result.source.result_id(),
        "content": fact.content,
        "namespace": fact.namespace,
        "source": source,
        "trust": "persisted_unjudged",
        "state": "current",
        "retrieval_receipt_ref": receipt_ref,
        "score": result.score,
        "bm25_rank": result.bm25_rank,
        "vector_rank": result.vector_rank,
        "cosine_similarity": result.cosine_similarity,
    })))
}

pub enum GraphPathOutcome {
    Found(Vec<String>),
    NoPathWithinCompleteSearch,
    BudgetExceeded,
    InvalidEndpoint(String),
}

fn typed_graph_path(
    graph: &dyn semantic_memory::GraphView,
    from: &str,
    to: &str,
    max_depth: usize,
) -> Result<GraphPathOutcome, semantic_memory::MemoryError> {
    use semantic_memory::GraphDirection;
    let from_edges = graph.neighbors(from, GraphDirection::Both, 1)?;
    if from_edges.is_empty() {
        return Ok(GraphPathOutcome::InvalidEndpoint(from.to_string()));
    }
    let to_edges = graph.neighbors(to, GraphDirection::Both, 1)?;
    if to_edges.is_empty() {
        return Ok(GraphPathOutcome::InvalidEndpoint(to.to_string()));
    }
    if from == to {
        return Ok(GraphPathOutcome::Found(vec![from.to_string()]));
    }

    let mut visited = HashSet::from([from.to_string()]);
    let mut parents = HashMap::<String, String>::new();
    let mut queue = VecDeque::from([(from.to_string(), 0usize)]);
    let mut hit_depth_budget = false;
    while let Some((node, depth)) = queue.pop_front() {
        let edges = graph.neighbors(&node, GraphDirection::Both, 1)?;
        for edge in edges {
            let next = if edge.source == node {
                edge.target
            } else {
                edge.source
            };
            if visited.contains(&next) {
                continue;
            }
            if depth >= max_depth {
                hit_depth_budget = true;
                continue;
            }
            visited.insert(next.clone());
            parents.insert(next.clone(), node.clone());
            if next == to {
                let mut path = vec![to.to_string()];
                let mut cursor = to.to_string();
                while let Some(parent) = parents.get(&cursor) {
                    path.push(parent.clone());
                    if parent == from {
                        break;
                    }
                    cursor = parent.clone();
                }
                path.reverse();
                return Ok(GraphPathOutcome::Found(path));
            }
            if visited.len() >= 500 {
                return Ok(GraphPathOutcome::BudgetExceeded);
            }
            queue.push_back((next, depth + 1));
        }
    }
    Ok(if hit_depth_budget {
        GraphPathOutcome::BudgetExceeded
    } else {
        GraphPathOutcome::NoPathWithinCompleteSearch
    })
}

fn build_path_segments(
    store: &semantic_memory::MemoryStore,
    path: &[String],
) -> Vec<serde_json::Value> {
    let mut segments = Vec::new();
    if path.len() < 2 {
        return segments;
    }

    for i in 0..path.len() - 1 {
        let from = &path[i];
        let to = &path[i + 1];

        // Get neighbors of the current node to find the edge to the next node.
        let g = store.graph_view();
        match g.neighbors(from, semantic_memory::GraphDirection::Both, 1) {
            Ok(edges) => {
                // Find the edge that connects from -> to.
                let connecting = edges.iter().find(|e| {
                    (e.source == *from && e.target == *to) || (e.source == *to && e.target == *from)
                });

                if let Some(edge) = connecting {
                    let edge_type_str = match &edge.edge_type {
                        semantic_memory::GraphEdgeType::Semantic { cosine_similarity } => {
                            serde_json::json!({
                                "type": "semantic",
                                "cosine_similarity": cosine_similarity,
                            })
                        }
                        semantic_memory::GraphEdgeType::Temporal { delta_secs } => {
                            serde_json::json!({
                                "type": "temporal",
                                "delta_secs": delta_secs,
                            })
                        }
                        semantic_memory::GraphEdgeType::Causal {
                            confidence,
                            evidence_ids,
                        } => {
                            serde_json::json!({
                                "type": "causal",
                                "confidence": confidence,
                                "evidence_ids": evidence_ids,
                            })
                        }
                        semantic_memory::GraphEdgeType::Entity { relation } => {
                            serde_json::json!({
                                "type": "entity",
                                "relation": relation,
                            })
                        }
                    };

                    segments.push(serde_json::json!({
                        "source": from,
                        "target": to,
                        "edge_type": edge_type_str,
                        "weight": edge.weight,
                        "metadata": edge.metadata,
                    }));
                } else {
                    // No edge found between consecutive path nodes — shouldn't
                    // happen but handle gracefully.
                    segments.push(serde_json::json!({
                        "source": from,
                        "target": to,
                        "edge_type": null,
                        "weight": null,
                        "metadata": null,
                    }));
                }
            }
            Err(_) => {
                segments.push(serde_json::json!({
                    "source": from,
                    "target": to,
                    "edge_type": null,
                    "weight": null,
                    "metadata": null,
                }));
            }
        }
    }

    segments
}

#[tool_router]
impl SemanticMemoryServer {
    #[tool(
        description = "Semantic hybrid search (BM25 + vector + RRF). Returns ranked results with content, scores, and stable result IDs.",
        annotations(read_only_hint = true)
    )]
    fn sm_search(
        &self,
        Parameters(SearchParams {
            query,
            top_k,
            namespaces,
        }): Parameters<SearchParams>,
    ) -> Result<Json<StructuredOutput>, ErrorData> {
        let requested_k = top_k.map(|v| v as usize).unwrap_or(5);
        let allow_superseded = false;
        let search_k = if allow_superseded {
            requested_k
        } else {
            (requested_k * 4).max(20)
        };
        let ns: Option<Vec<&str>> = namespaces
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect());

        let store = &self.bridge.store;
        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.search(&query, Some(search_k), ns.as_deref(), None))
        });

        match result {
            Ok(results) => {
                let superseded_targets = if allow_superseded {
                    HashSet::new()
                } else {
                    load_superseded_targets(store)?
                };
                let fresh_results: Vec<_> = results
                    .iter()
                    .filter(|r| !superseded_targets.contains(&r.source.result_id()))
                    .collect();
                let result_refs: Vec<_> = if superseded_targets.is_empty() {
                    results.iter().collect()
                } else {
                    fresh_results
                };
                let superseded_filtered_count = results.len().saturating_sub(result_refs.len());
                let json_results: Vec<serde_json::Value> = result_refs
                    .iter()
                    .take(requested_k)
                    .map(|r| {
                        serde_json::json!({
                            "result_id": r.source.result_id(),
                            "content": r.content,
                            "source": format!("{:?}", r.source),
                            "score": r.score,
                            "bm25_rank": r.bm25_rank,
                            "vector_rank": r.vector_rank,
                            "cosine_similarity": r.cosine_similarity,
                        })
                    })
                    .collect();
                json_to_output(&serde_json::json!({
                    "ok": true,
                    "results": json_results,
                    "count": json_results.len(),
                    "superseded_filtered_count": superseded_filtered_count,
                }))
            }
            Err(e) => Err(ErrorData::internal_error(
                format!("Search error: {e}"),
                None,
            )),
        }
    }

    #[tool(
        description = "Mandatory witnessed retrieval. Bypasses cache, verifies durable receipt persistence, defaults to Current state, and supports privacy-preserving opt-in storage for complete replay.",
        annotations(read_only_hint = true)
    )]
    fn sm_search_witnessed(
        &self,
        Parameters(SearchWitnessedParams {
            query,
            top_k,
            namespaces,
            request_id,
            retrieval_mode,
            replay_mode,
        }): Parameters<SearchWitnessedParams>,
    ) -> Result<Json<StructuredOutput>, ErrorData> {
        use semantic_memory::{ExactnessProfile, ReceiptMode, ReplayMode, SearchContext};
        let k = top_k.map(|v| v as usize).unwrap_or(5);
        let request_id = request_id.unwrap_or_else(|| {
            format!(
                "mcp-witness-{}-{}",
                chrono::Utc::now().timestamp_micros(),
                WITNESS_REQUEST_SEQUENCE.fetch_add(1, Ordering::Relaxed)
            )
        });
        let digest = |s: &str| format!("blake3:{}", blake3::hash(s.as_bytes()).to_hex());
        let filters = serde_json::json!({"namespaces": namespaces});
        let query_digest = digest(&query);
        let input_digest = digest(
            &serde_json::json!({"query": query, "top_k": k, "filters": filters}).to_string(),
        );
        let filter_digest = digest(&filters.to_string());
        let retrieval_mode = retrieval_mode.unwrap_or(RetrievalModeParam::Hybrid);
        let retrieval_mode_name = match retrieval_mode {
            RetrievalModeParam::Hybrid => "hybrid",
            RetrievalModeParam::FtsOnly => "fts_only",
            RetrievalModeParam::VectorOnly => "vector_only",
        };
        let config_digest = digest(&format!(
            "retrieval_mode={retrieval_mode_name};top_k={k};state=current;cache=bypass;exactness=prefer_exact"
        ));
        let ns: Option<Vec<&str>> = namespaces
            .as_ref()
            .map(|v| v.iter().map(String::as_str).collect());
        let mut context = SearchContext::default_now();
        context.receipt_mode = ReceiptMode::ReturnReceipt;
        context.replay_mode = match replay_mode.unwrap_or(ReplayModeParam::NoReplay) {
            ReplayModeParam::NoReplay => ReplayMode::NoReplay,
            ReplayModeParam::StoreInputs => ReplayMode::StoreInputs,
        };
        context.exactness_profile = ExactnessProfile::PreferExact;
        context.request_id = Some(request_id.clone());
        context.query_text_digest = Some(query_digest.clone());
        context.query_input_digest = Some(input_digest.clone());
        context.filter_digest = Some(filter_digest.clone());
        // ReturnReceipt bypasses semantic-memory's cache and propagates persistence failure.
        let response = tokio::task::block_in_place(|| {
            Handle::current().block_on(async {
                match retrieval_mode {
                    RetrievalModeParam::Hybrid => {
                        self.bridge
                            .store
                            .search_with_context(&query, Some(k), ns.as_deref(), None, context)
                            .await
                    }
                    RetrievalModeParam::FtsOnly => {
                        self.bridge
                            .store
                            .search_fts_only_with_context(
                                &query,
                                Some(k),
                                ns.as_deref(),
                                None,
                                context,
                            )
                            .await
                    }
                    RetrievalModeParam::VectorOnly => {
                        self.bridge
                            .store
                            .search_vector_only_with_context(
                                &query,
                                Some(k),
                                ns.as_deref(),
                                None,
                                context,
                            )
                            .await
                    }
                }
            })
        })
        .map_err(|e| {
            ErrorData::internal_error(
                format!("witnessed search/receipt persistence failed: {e}"),
                None,
            )
        })?;
        let receipt = response.receipt.ok_or_else(|| {
            ErrorData::internal_error("witness missing; operation contained".to_string(), None)
        })?;
        let authority_state = tokio::task::block_in_place(|| {
            Handle::current().block_on(self.bridge.store.authority().current_state())
        })
        .map_err(|error| {
            ErrorData::internal_error(format!("authority state lookup failed: {error}"), None)
        })?;
        let durable = tokio::task::block_in_place(|| {
            Handle::current().block_on(self.bridge.store.get_search_receipt(&receipt.receipt_id))
        })
        .map_err(|e| {
            ErrorData::internal_error(format!("receipt verification failed: {e}"), None)
        })?;
        if durable.is_none() {
            return Err(ErrorData::internal_error(
                "receipt not durable; operation contained".to_string(),
                None,
            ));
        }
        let complete_replay_available = tokio::task::block_in_place(|| {
            Handle::current().block_on(
                self.bridge
                    .store
                    .search_replay_inputs_available(&receipt.receipt_id),
            )
        })
        .map_err(|e| {
            ErrorData::internal_error(format!("replay input verification failed: {e}"), None)
        })?;
        let stats =
            tokio::task::block_in_place(|| Handle::current().block_on(self.bridge.store.stats()))
                .map_err(|e| {
                ErrorData::internal_error(format!("model identity unavailable: {e}"), None)
            })?;
        let model_digest = digest(&serde_json::json!({"model": stats.embedding_model, "dimensions": stats.embedding_dimensions}).to_string());
        let receipt_ref = format!("receipt:{}", receipt.receipt_id);
        let mut results = Vec::new();
        for result in response.results {
            if let Some(hit) = witnessed_injectible_fact(&self.bridge.store, result, &receipt_ref)?
            {
                results.push(hit);
            }
        }
        // T2.6: Enrich search results with claim-ledger support state.
        // Best-effort: falls back to "persisted_unjudged" when no claim exists.
        self.enrich_results_with_trust(&mut results);

        // P1.3: Factor graph reranking (opt-in via integration feature).
        // When graph edges exist in the store, build a factor graph with
        // search scores as initial beliefs, run belief propagation, and
        // rerank results by refined beliefs. Items connected by multiple
        // relationship types get compounded confidence.
        #[cfg(feature = "integration")]
        {
            use semantic_memory::factor_graph::{
                factors_from_edges, FactorGraph, FactorGraphConfig,
            };
            let result_nodes: Vec<(String, f64)> = results
                .iter()
                .filter_map(|result| {
                    let id = result
                        .get("memory_id")
                        .and_then(|value| value.as_str())?
                        .to_string();
                    let score = result
                        .get("score")
                        .and_then(|value| value.as_f64())
                        .unwrap_or(0.5);
                    Some((id, score))
                })
                .collect();
            if !result_nodes.is_empty() {
                let seed_ids: Vec<String> = result_nodes
                    .iter()
                    .map(|(item_id, _)| item_id.clone())
                    .collect();
                let edge_tuples = load_neighborhood_factor_edges(&self.bridge.store, &seed_ids)?;
                if !edge_tuples.is_empty() {
                    let factors = factors_from_edges(&edge_tuples);
                    let factor_graph =
                        FactorGraph::new(&result_nodes, factors, FactorGraphConfig::default());
                    let result_beliefs = factor_graph.propagate();
                    let reranked = result_beliefs.top_k(result_nodes.len());
                    // Reorder results by factor graph beliefs (higher = better).
                    results.sort_by(|a, b| {
                        let a_id = a.get("memory_id").and_then(|v| v.as_str()).unwrap_or("");
                        let b_id = b.get("memory_id").and_then(|v| v.as_str()).unwrap_or("");
                        let a_belief = reranked
                            .iter()
                            .find(|(id, _)| id == a_id)
                            .map(|(_, b)| *b)
                            .unwrap_or(0.0);
                        let b_belief = reranked
                            .iter()
                            .find(|(id, _)| id == b_id)
                            .map(|(_, b)| *b)
                            .unwrap_or(0.0);
                        b_belief
                            .partial_cmp(&a_belief)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                }
            }
        }

        let ordered_results: Vec<_> = results.iter().map(|r| serde_json::json!({"result_id": r["result_id"], "result_digest": digest(&r.to_string())})).collect();
        let exactness = if receipt.approximate {
            "approximate_candidates"
        } else if receipt.exact_rerank {
            "exact_f32_rerank"
        } else {
            "backend_reported_non_approximate"
        };
        json_to_output(&serde_json::json!({
            "schema_version": "retrieval_response_v1", "ok": true, "request_id": request_id, "receipt_id": receipt.receipt_id, "retrieval_mode": retrieval_mode_name,
            "state_view": {"kind": "Current"}, "current_snapshot_id": authority_state.snapshot_id.0,
            "retrieval_epoch": authority_state.retrieval_epoch.0,
            "evaluation_time": receipt.evaluation_time,
            "authority": {
                "snapshot_id": authority_state.snapshot_id.0,
                "retrieval_epoch": authority_state.retrieval_epoch.0,
                "status": "Applied",
                "degradation": null
            },
            "digests": {"query_text": query_digest, "input": input_digest, "filter": filter_digest, "config": config_digest, "model": model_digest},
            "execution": {"cache": "bypassed", "candidate_backend": receipt.candidate_backend, "exactness": exactness, "artifact_generation_id": receipt.artifact_generation_id},
            "ordered_results": ordered_results, "results": results,
            "stage_outcomes": {
                "authority_snapshot": {"outcome": "Applied", "degradation": null},
                "hybrid_retrieval": {"outcome": if matches!(retrieval_mode, RetrievalModeParam::Hybrid) { "Applied" } else { "Skipped" }, "degradation": null},
                "selected_retrieval": {"outcome": "Applied", "degradation": null, "mode": retrieval_mode_name},
                "receipt_persistence": {"outcome": "Applied", "degradation": null},
                "cache": {"outcome": "Skipped", "degradation": "witnessed retrieval bypasses cache"},
                "replay": if complete_replay_available {
                    serde_json::json!({"outcome": "Applied", "degradation": null})
                } else {
                    serde_json::json!({"outcome": "AnalysisOnly", "degradation": "complete replay inputs are not available"})
                }
            },
            "degradations": receipt.degradations,
            "complete_replay_available": complete_replay_available
        }))
    }

    #[tool(
        description = "Get knowledge base statistics: fact/chunk/document/session counts, DB size, embedding model, and graph edge count.",
        annotations(read_only_hint = true)
    )]
    fn sm_stats(&self) -> Result<Json<StatsOutput>, ErrorData> {
        let store = &self.bridge.store;
        let core = tokio::task::block_in_place(|| Handle::current().block_on(store.stats()));
        let graph = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.list_all_graph_edges())
        });
        let core_health = match &core {
            Ok(_) => serde_json::json!({"health": "healthy", "error": null}),
            Err(e) => serde_json::json!({"health": "error", "error": e.to_string()}),
        };
        let graph_health = match &graph {
            Ok(_) => serde_json::json!({"health": "healthy", "error": null}),
            Err(e) => serde_json::json!({"health": "error", "error": e.to_string()}),
        };
        let core_value = core.ok();
        let graph_count = graph.ok().map(|edges| edges.len());
        Ok(Json(StatsOutput {
            ok: core_value.is_some() && graph_count.is_some(),
            components: serde_json::json!({"core": core_health, "graph": graph_health}),
            facts: core_value.as_ref().map(|s| s.total_facts),
            chunks: core_value.as_ref().map(|s| s.total_chunks),
            documents: core_value.as_ref().map(|s| s.total_documents),
            sessions: core_value.as_ref().map(|s| s.total_sessions),
            messages: core_value.as_ref().map(|s| s.total_messages),
            graph_edges: graph_count,
            db_size_bytes: core_value.as_ref().map(|s| s.database_size_bytes),
            db_size_mb: core_value
                .as_ref()
                .map(|s| (s.database_size_bytes as f64 / 1_048_576.0 * 100.0).round() / 100.0),
            embedding_model: core_value.as_ref().and_then(|s| s.embedding_model.clone()),
            embedding_dimensions: core_value.as_ref().and_then(|s| s.embedding_dimensions),
        }))
    }

    #[tool(
        description = "List namespaces that currently contain facts. Use before sm_list_facts to discover what is stored.",
        annotations(read_only_hint = true)
    )]
    fn sm_list_namespaces(&self) -> Result<Json<StructuredOutput>, ErrorData> {
        let store = &self.bridge.store;
        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.list_fact_namespaces())
        });
        match result {
            Ok(ns) => json_to_output(&serde_json::json!({
                "ok": true,
                "count": ns.len(),
                "namespaces": ns,
            })),
            Err(e) => Err(ErrorData::internal_error(
                format!("list_namespaces error: {e}"),
                None,
            )),
        }
    }

    #[tool(
        description = "Fetch one fact by id (bare UUID or prefixed 'fact:<uuid>'). Returns full content, namespace, source, timestamps, and metadata.",
        annotations(read_only_hint = true)
    )]
    fn sm_get_fact(
        &self,
        Parameters(GetFactParams { fact_id }): Parameters<GetFactParams>,
    ) -> Result<Json<StructuredOutput>, ErrorData> {
        let bare = fact_id
            .strip_prefix("fact:")
            .unwrap_or(&fact_id)
            .to_string();
        let store = &self.bridge.store;
        let result =
            tokio::task::block_in_place(|| Handle::current().block_on(store.get_fact(&bare)));
        match result {
            Ok(Some(f)) => json_to_output(&serde_json::json!({
                "ok": true,
                "found": true,
                "fact": {
                    "result_id": format!("fact:{}", f.id),
                    "id": f.id,
                    "namespace": f.namespace,
                    "content": f.content,
                    "source": f.source,
                    "created_at": f.created_at,
                    "updated_at": f.updated_at,
                    "metadata": f.metadata,
                },
            })),
            Ok(None) => json_to_output(&serde_json::json!({
                "ok": true,
                "found": false,
                "message": format!("No fact with id '{fact_id}'"),
            })),
            Err(e) => Err(ErrorData::internal_error(
                format!("get_fact error: {e}"),
                None,
            )),
        }
    }

    #[tool(
        description = "Fetch a fact plus its graph neighbors WITH their content in one call. Hydrates neighbor facts for ids returned by graph tools.",
        annotations(read_only_hint = true)
    )]
    fn sm_get_fact_neighbors(
        &self,
        Parameters(GetFactNeighborsParams { item_id }): Parameters<GetFactNeighborsParams>,
    ) -> Result<Json<StructuredOutput>, ErrorData> {
        let node_id = if item_id.contains(':') {
            item_id.clone()
        } else {
            format!("fact:{item_id}")
        };
        let bare = node_id
            .strip_prefix("fact:")
            .unwrap_or(&node_id)
            .to_string();
        let store = &self.bridge.store;

        let center =
            tokio::task::block_in_place(|| Handle::current().block_on(store.get_fact(&bare)))
                .map_err(|e| ErrorData::internal_error(format!("get_fact error: {e}"), None))?;
        let edges = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.list_graph_edges_for_node(&node_id))
        })
        .map_err(|e| ErrorData::internal_error(format!("list edges error: {e}"), None))?;

        let mut neighbors: Vec<serde_json::Value> = Vec::new();
        for e in &edges {
            let outgoing = e.source == node_id;
            let other = if outgoing { &e.target } else { &e.source };
            let other_bare = other.strip_prefix("fact:").unwrap_or(other).to_string();
            let content = tokio::task::block_in_place(|| {
                Handle::current().block_on(store.get_fact(&other_bare))
            })
            .ok()
            .flatten()
            .map(|f| f.content);
            neighbors.push(serde_json::json!({
                "neighbor_id": other,
                "direction": if outgoing { "out" } else { "in" },
                "edge_type": e.edge_type,
                "weight": e.weight,
                "content": content,
            }));
        }
        json_to_output(&serde_json::json!({
            "ok": true,
            "item_id": node_id,
            "center_content": center.map(|f| f.content),
            "neighbor_count": neighbors.len(),
            "neighbors": neighbors,
        }))
    }

    #[tool(
        description = "Find shortest path between two items in the knowledge graph. Traverses all edge types. Returns node IDs with edge evidence per hop.",
        annotations(read_only_hint = true)
    )]
    fn sm_graph_path(
        &self,
        Parameters(GraphPathParams {
            from_id,
            to_id,
            max_depth,
        }): Parameters<GraphPathParams>,
    ) -> Result<Json<StructuredOutput>, ErrorData> {
        let depth = max_depth.map(|v| v as usize).unwrap_or(5);
        let store = &self.bridge.store;
        let g = store.graph_view();

        match typed_graph_path(g.as_ref(), &from_id, &to_id, depth) {
            Ok(GraphPathOutcome::Found(path)) => {
                // Build edge evidence for each hop by examining neighbors.
                let path_segments = build_path_segments(store, &path);
                json_to_output(&serde_json::json!({
                    "ok": true,
                    "outcome": "Found",
                    "from": from_id,
                    "to": to_id,
                    "path": path,
                    "path_length": path.len(),
                    "segments": path_segments,
                }))
            }
            Ok(GraphPathOutcome::NoPathWithinCompleteSearch) => {
                json_to_output(&serde_json::json!({
                    "ok": true,
                    "outcome": "NoPathWithinCompleteSearch",
                    "from": from_id,
                    "to": to_id,
                    "path": null,
                    "message": format!("No path found from {from_id} to {to_id} within depth {depth}"),
                }))
            }
            Ok(GraphPathOutcome::BudgetExceeded) => json_to_output(&serde_json::json!({
                "ok": false, "outcome": "BudgetExceeded", "from": from_id, "to": to_id,
                "path": null, "budget": {"max_depth": depth}
            })),
            Ok(GraphPathOutcome::InvalidEndpoint(endpoint)) => json_to_output(&serde_json::json!({
                "ok": false, "outcome": "InvalidEndpoint", "invalid_endpoint": endpoint,
                "from": from_id, "to": to_id, "path": null
            })),
            Err(e) => Err(ErrorData::internal_error(
                format!("Graph view error: {e}"),
                None,
            )),
        }
    }

    #[tool(
        description = "Hybrid semantic search over stored conversation MESSAGES (not facts). Recall what was discussed in past sessions. Returns ranked messages.",
        annotations(read_only_hint = true)
    )]
    fn sm_search_conversations(
        &self,
        Parameters(SearchConversationsParams { query, top_k }): Parameters<
            SearchConversationsParams,
        >,
    ) -> Result<Json<StructuredOutput>, ErrorData> {
        let k = top_k.map(|v| v as usize);
        let store = &self.bridge.store;
        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.search_conversations(&query, k, None))
        });
        match result {
            Ok(results) => json_to_output(&serde_json::json!({
                "ok": true,
                "count": results.len(),
                "results": results.iter().map(|r| serde_json::json!({
                    "result_id": r.source.result_id(),
                    "content": r.content,
                    "score": r.score,
                    "cosine_similarity": r.cosine_similarity,
                })).collect::<Vec<_>>(),
            })),
            Err(e) => Err(ErrorData::internal_error(
                format!("search_conversations error: {e}"),
                None,
            )),
        }
    }

    #[tool(
        description = "Add a fact to the knowledge base. Embedded and indexed for semantic search. Returns fact ID and content digest."
    )]
    fn sm_add_fact(
        &self,
        Parameters(AddFactParams {
            content,
            namespace,
            source,
            extract_entities,
            memory_kind,
            sensitivity,
            evidence_refs,
            idempotency_key,
        }): Parameters<AddFactParams>,
    ) -> Result<Json<StructuredOutput>, ErrorData> {
        let store = &self.bridge.store;

        // Admission gate: classify sensitivity
        let sens = sensitivity.unwrap_or_else(|| "internal".to_string());
        let kind = memory_kind.unwrap_or_else(|| "durable_fact".to_string());

        // Block confidential/restricted content from autocapture
        if sens == "confidential" || sens == "restricted" {
            return Err(ErrorData::invalid_params(
                format!("Admission gate BLOCKED: sensitivity='{sens}' content cannot be stored without explicit user request"),
                None,
            ));
        }

        // Block ephemeral_inference from becoming durable without evidence
        let explicit_evidence: Vec<String> = evidence_refs
            .as_deref()
            .unwrap_or_default()
            .iter()
            .filter(|reference| !reference.trim().is_empty())
            .cloned()
            .collect();
        if kind == "ephemeral_inference" && explicit_evidence.is_empty() {
            return Err(ErrorData::invalid_params(
                "Admission gate BLOCKED: ephemeral_inference requires evidence_refs to promote to durable".to_string(),
                None,
            ));
        }

        let mut authority_evidence = explicit_evidence;
        if let Some(source_ref) = source.as_ref().filter(|value| !value.trim().is_empty()) {
            if !authority_evidence.contains(source_ref) {
                authority_evidence.push(source_ref.clone());
            }
        }

        // Build metadata JSON with typed memory fields
        let mut meta = serde_json::Map::new();
        meta.insert("memory_kind".to_string(), serde_json::json!(kind));
        meta.insert("sensitivity".to_string(), serde_json::json!(sens));
        if let Some(refs) = evidence_refs {
            meta.insert("evidence_refs".to_string(), serde_json::json!(refs));
        }
        let metadata = serde_json::Value::Object(meta);

        let caller_idempotency_key = match idempotency_key {
            Some(key) if !key.trim().is_empty() => key,
            Some(_) => {
                return Err(ErrorData::invalid_params(
                    "idempotency_key must not be blank".to_string(),
                    None,
                ))
            }
            None => format!("mcp-sm-add-fact:{}", uuid::Uuid::new_v4()),
        };
        let origin = if authority_evidence.is_empty() {
            semantic_memory::OriginAuthorityLabelV1::operator_system(
                "principal:semantic-memory-mcp",
                "caller:sm_add_fact",
            )
        } else {
            semantic_memory::OriginAuthorityLabelV1::new(
                semantic_memory::OriginClassV1::ExternalEvidence,
                "principal:semantic-memory-mcp",
                "caller:sm_add_fact",
                format!(
                    "blake3:{}",
                    blake3::hash(authority_evidence.join("\n").as_bytes()).to_hex()
                ),
                semantic_memory::OriginRiskV1::Medium,
                semantic_memory::AuthorityScopesV1 {
                    recall: semantic_memory::AuthorityScopeV1::Universal,
                    assertion: semantic_memory::AuthorityScopeV1::Denied,
                    action: semantic_memory::AuthorityScopeV1::Denied,
                },
                semantic_memory::ElevationRequirementV1::ExplicitOperatorApproval,
                None,
                semantic_memory::RevocationStatusV1::Active,
                vec!["principal:semantic-memory-mcp".into()],
            )
            .map_err(|error| {
                ErrorData::internal_error(format!("invalid origin label: {error}"), None)
            })?
        };
        let permit = if authority_evidence.is_empty() {
            semantic_memory::AuthorityPermit::operator_system(
                "principal:semantic-memory-mcp",
                "caller:sm_add_fact",
                semantic_memory::AuthorityPermit::APPEND_CAPABILITY,
            )
        } else {
            semantic_memory::AuthorityPermit::with_evidence(
                "principal:semantic-memory-mcp",
                "caller:sm_add_fact",
                semantic_memory::AuthorityPermit::APPEND_CAPABILITY,
                authority_evidence,
            )
        }
        .with_origin(origin);

        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.authority().append_with_metadata(
                permit,
                caller_idempotency_key,
                namespace.clone(),
                content.clone(),
                source.clone(),
                Some(metadata),
            ))
        });

        match result {
            Ok(receipt) => {
                let id = receipt.affected_ids.first().cloned().ok_or_else(|| {
                    ErrorData::internal_error(
                        "authority append returned no affected fact id".to_string(),
                        None,
                    )
                })?;
                // D4: best-effort auto-link to an existing claim with matching
                // normalized content. Never fails the whole operation.
                self.auto_link_fact_to_claims(&id, &content);
                // Optional entity extraction — best-effort, never fails the whole operation.
                if extract_entities == Some(true) {
                    let prompt = format!(
                        "Extract entities from this text as JSON. Format: {{\"entities\": [{{\"name\": \"...\", \"type\": \"person|project|concept|tool|version|path\"}}]}}\nText: {content}\nJSON:"
                    );
                    let body = serde_json::json!({
                        "model": "granite4.1:3b",
                        "prompt": prompt,
                        "stream": false,
                        "options": {"temperature": 0, "num_predict": 200}
                    });
                    if let Ok(resp) = reqwest::blocking::Client::new()
                        .post("http://127.0.0.1:11434/api/generate")
                        .json(&body)
                        .send()
                    {
                        if let Ok(v) = resp.json::<serde_json::Value>() {
                            if let Some(response_str) = v.get("response").and_then(|r| r.as_str()) {
                                // Use boundary compiler for robust JSON parsing with duplicate-key rejection
                                let parsed_result =
                                    boundary_compiler::parse_with_dup_check(response_str.trim());
                                if let Ok(parsed) = parsed_result {
                                    if let Some(entities) =
                                        parsed.get("entities").and_then(|e| e.as_array())
                                    {
                                        let fact_node = format!("fact:{id}");
                                        for entity in entities {
                                            if let Some(name) =
                                                entity.get("name").and_then(|n| n.as_str())
                                            {
                                                let entity_node = format!("entity:{name}");
                                                let _ = tokio::task::block_in_place(|| {
                                                    Handle::current()
                                                        .block_on(store.add_graph_edge(
                                                        &fact_node,
                                                        &entity_node,
                                                        semantic_memory::GraphEdgeType::Entity {
                                                            relation: "mentions".to_string(),
                                                        },
                                                        1.0,
                                                        None,
                                                    ))
                                                });
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                json_to_output(&serde_json::json!({
                    "ok": true,
                    "fact_id": id,
                    "namespace": namespace,
                    "receipt": mcp_receipt("sm_add_fact"),
                    "message": "Fact added successfully",
                }))
            }
            Err(e) => Err(ErrorData::internal_error(
                format!("Error adding fact: {e}"),
                None,
            )),
        }
    }

    #[tool(
        description = "Create a replacement fact and link it to a stale fact via 'supersedes' edge. Use instead of deleting outdated facts. Returns new fact id and edge id.",
        annotations(idempotent_hint = true)
    )]
    fn sm_supersede_fact(
        &self,
        Parameters(SupersedeFactParams {
            old_fact_id,
            content,
            namespace,
            source,
            reason,
        }): Parameters<SupersedeFactParams>,
    ) -> Result<Json<StructuredOutput>, ErrorData> {
        use semantic_memory::GraphEdgeType;

        let old_bare = old_fact_id
            .strip_prefix("fact:")
            .unwrap_or(&old_fact_id)
            .to_string();
        let old_node = format!("fact:{old_bare}");
        let store = &self.bridge.store;
        let old =
            tokio::task::block_in_place(|| Handle::current().block_on(store.get_fact(&old_bare)))
                .map_err(|e| ErrorData::internal_error(format!("get old fact error: {e}"), None))?;
        let Some(old_fact) = old else {
            return Err(ErrorData::invalid_params(
                format!("No fact with id '{old_fact_id}'"),
                None,
            ));
        };

        let ns = namespace.unwrap_or_else(|| old_fact.namespace.clone());
        let new_id = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.add_fact(&ns, &content, source.as_deref(), None))
        })
        .map_err(|e| ErrorData::internal_error(format!("add replacement fact error: {e}"), None))?;
        let new_node = format!("fact:{new_id}");
        let metadata = serde_json::json!({
            "reason": reason.unwrap_or_else(|| "replacement fact supersedes stale fact".to_string()),
            "old_fact_id": old_bare,
        });
        let edge = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.add_graph_edge(
                &new_node,
                &old_node,
                GraphEdgeType::Entity {
                    relation: "supersedes".to_string(),
                },
                1.0,
                Some(metadata),
            ))
        })
        .map_err(|e| ErrorData::internal_error(format!("add supersedes edge error: {e}"), None))?;

        json_to_output(&serde_json::json!({
            "ok": true,
            "receipt": mcp_receipt("sm_supersede_fact"),
            "new_fact_id": new_id,
            "new_result_id": new_node,
            "old_fact_id": old_bare,
            "old_result_id": old_node,
            "namespace": ns,
            "edge_id": edge.id,
            "relation": "supersedes",
        }))
    }

    #[tool(
        description = "Add a durable, typed graph edge between two nodes. Edge types: semantic, temporal, causal, entity. Idempotent — same edge returns existing ID.",
        annotations(idempotent_hint = true)
    )]
    fn sm_add_graph_edge(
        &self,
        Parameters(params): Parameters<AddGraphEdgeParams>,
    ) -> Result<Json<StructuredOutput>, ErrorData> {
        use semantic_memory::GraphEdgeType;

        // SM-AUD-015: Validate numeric params are finite and in range.
        if let Some(cs) = params.cosine_similarity {
            if !cs.is_finite() || !(0.0..=1.0).contains(&cs) {
                return Err(ErrorData::invalid_params(
                    format!("cosine_similarity must be finite and in [0.0, 1.0], got {cs}"),
                    None,
                ));
            }
        }
        if let Some(conf) = params.confidence {
            if !conf.is_finite() || !(0.0..=1.0).contains(&conf) {
                return Err(ErrorData::invalid_params(
                    format!("confidence must be finite and in [0.0, 1.0], got {conf}"),
                    None,
                ));
            }
        }

        let edge_type = match params.edge_type {
            EdgeType::Semantic => GraphEdgeType::Semantic {
                cosine_similarity: params.cosine_similarity.unwrap_or(0.5),
            },
            EdgeType::Temporal => GraphEdgeType::Temporal {
                delta_secs: params.delta_secs.unwrap_or(0),
            },
            EdgeType::Causal => GraphEdgeType::Causal {
                confidence: params.confidence.unwrap_or(0.5),
                evidence_ids: params.evidence_ids.unwrap_or_default(),
            },
            EdgeType::Entity => GraphEdgeType::Entity {
                relation: params.relation.unwrap_or_else(|| "related".to_string()),
            },
        };

        // MCP-004: Reject malformed metadata JSON instead of silently dropping it.
        let metadata = match params.metadata.as_deref() {
            None => None,
            Some(s) => match serde_json::from_str::<serde_json::Value>(s) {
                Ok(v) => Some(v),
                Err(e) => {
                    return Err(ErrorData::invalid_params(
                        format!("metadata is not valid JSON: {e}"),
                        None,
                    ))
                }
            },
        };

        let store = &self.bridge.store;
        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.add_graph_edge(
                &params.source,
                &params.target,
                edge_type,
                params.weight,
                metadata,
            ))
        });

        match result {
            Ok(edge) => json_to_output(&serde_json::json!({
                "ok": true,
                "receipt": mcp_receipt("sm_add_graph_edge"),
                "id": edge.id,
                "source": edge.source,
                "target": edge.target,
                "edge_type": edge.edge_type,
                "weight": edge.weight,
                "content_digest": edge.content_digest,
                "recorded_at": edge.recorded_at,
                "message": "Graph edge added successfully",
            })),
            Err(e) => Err(ErrorData::internal_error(
                format!("Error adding graph edge: {e}"),
                None,
            )),
        }
    }

    #[tool(
        description = "Return the canonical typed origin-authority decision receipt for asserting a fact. The purpose is fixed to assertion; recall authority is not reused. This read-only decision surface never returns memory content.",
        annotations(read_only_hint = true)
    )]
    fn sm_decide_assertion_authority(
        &self,
        Parameters(params): Parameters<GovernedDecisionParams>,
    ) -> Result<Json<StructuredOutput>, ErrorData> {
        self.decide_governed_authority(params, semantic_memory::GovernedAccessPurposeV1::Assertion)
    }

    #[tool(
        description = "Return the canonical typed origin-authority decision receipt for acting on a fact. The purpose is fixed to action; recall or assertion authority is not reused. This read-only decision surface never returns memory content or performs the action.",
        annotations(read_only_hint = true)
    )]
    fn sm_decide_action_authority(
        &self,
        Parameters(params): Parameters<GovernedDecisionParams>,
    ) -> Result<Json<StructuredOutput>, ErrorData> {
        self.decide_governed_authority(params, semantic_memory::GovernedAccessPurposeV1::Action)
    }
}

#[tool_handler(
    router = self.tool_router,
    name = "semantic-memory-mcp",
    version = "0.5.3",
    instructions = "Compile-time stable semantic memory surface. Search before asking for context. Use witnessed retrieval for durable receipts. Recall authority never implies assertion or action authority. Prefer supersession to deletion."
)]
impl ServerHandler for SemanticMemoryServer {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bridge::{BridgeConfig, EmbedderBackend};

    fn open_server() -> (tempfile::TempDir, SemanticMemoryServer) {
        let dir = tempfile::tempdir().unwrap();
        let bridge = MemoryBridge::open(BridgeConfig {
            memory_dir: dir.path().to_path_buf(),
            embedder_backend: EmbedderBackend::Mock,
            embedding_url: String::new(),
            embedding_model: "mock".into(),
            embedding_dims: 768,
            turbo_quant_enabled: false,
            turbo_quant_bits: None,
            turbo_quant_projections: None,
        })
        .unwrap();
        (dir, SemanticMemoryServer::new(bridge, "stable"))
    }

    #[test]
    fn stable_router_is_exact_and_schema_backed() {
        let (_dir, server) = open_server();
        let expected: HashSet<&str> = [
            "sm_search",
            "sm_search_witnessed",
            "sm_stats",
            "sm_list_namespaces",
            "sm_get_fact",
            "sm_get_fact_neighbors",
            "sm_graph_path",
            "sm_search_conversations",
            "sm_add_fact",
            "sm_supersede_fact",
            "sm_add_graph_edge",
            "sm_decide_assertion_authority",
            "sm_decide_action_authority",
        ]
        .into_iter()
        .collect();
        let tools = server.tool_router.list_all();
        assert_eq!(tools.len(), expected.len());
        for tool in tools {
            assert!(expected.contains(tool.name.as_ref()));
            assert_eq!(
                tool.output_schema
                    .as_ref()
                    .and_then(|s| s.get("type"))
                    .and_then(serde_json::Value::as_str),
                Some("object")
            );
        }
        for forbidden in [
            "sm_delete_fact",
            "sm_route_query",
            "sm_create_claim",
            "sm_parse_json",
        ] {
            assert!(!server.exposes_tool(forbidden));
        }
    }

    #[test]
    #[should_panic(expected = "accepts only --tool-profile stable")]
    fn stable_build_rejects_runtime_widening() {
        let (dir, _) = open_server();
        let bridge = MemoryBridge::open(BridgeConfig {
            memory_dir: dir.path().join("other"),
            embedder_backend: EmbedderBackend::Mock,
            embedding_url: String::new(),
            embedding_model: "mock".into(),
            embedding_dims: 768,
            turbo_quant_enabled: false,
            turbo_quant_bits: None,
            turbo_quant_projections: None,
        })
        .unwrap();
        let _ = SemanticMemoryServer::new(bridge, "full");
    }
}
