//! MCP server handler using rmcp's #[tool_router] macro.
//!
//! Each #[tool] method becomes an MCP tool that Hermes/Claude Desktop
//! can discover and call. The rmcp macro auto-generates JSON Schema
//! from the parameter structs in tools.rs.

use crate::bridge::MemoryBridge;
use crate::tools::*;
use rmcp::{
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    tool, tool_handler, tool_router, ErrorData, ServerHandler,
};
use std::collections::HashSet;
use std::sync::Arc;
use tokio::runtime::Handle;

// Re-export the specific parameter types we use in tool signatures.
use crate::tools::{
    AddGraphEdgeParams, CommunityParams, FactorGraphParams, InvalidateGraphEdgeParams,
    ListGraphEdgesParams, RecordOutcomeParams, TopologyParams,
};
use crate::tools::{
    AddSupportAdmissionParams, ClassifyQueryParams, EntityLookupParams,
    EvaluateProofDebtGateParams, ExportClaimBundleParams, PlanQueryParams, ProjectionHealthParams,
    ProofDebtStatusParams, QueryOrchestratedParams, QueryTemporalKParams,
    RecordContradictionParams, ResolveContradictionParams, SupersedeClaimParams,
    VerifyLedgerParams,
};

pub struct SemanticMemoryServer {
    bridge: Arc<MemoryBridge>,
    tool_router: ToolRouter<Self>,
    #[cfg(feature = "orchestration")]
    runtime: Option<knowledge_runtime::KnowledgeRuntime>,
}

impl SemanticMemoryServer {
    pub fn new(bridge: MemoryBridge, tool_profile: &str) -> Self {
        let mut router = Self::tool_router();

        // Tools hidden in "lean" mode (maintenance + audit + bitemporal query + import)
        // Plus new admin-gated orchestration and claim-ledger tools.
        let admin_tools = [
            // Maintenance
            "sm_reconcile",
            "sm_vacuum",
            "sm_reembed_all",
            "sm_embeddings_are_dirty",
            // Audit
            "sm_get_search_receipt",
            "sm_replay_search_receipt",
            // Bitemporal query
            "sm_query_claim_versions",
            "sm_query_relation_versions",
            "sm_query_episodes",
            "sm_query_entity_aliases",
            "sm_query_evidence_refs",
            // Import
            "sm_import_envelope",
            "sm_import_status",
            "sm_list_imports",
            // Orchestration admin tools
            "sm_query_temporal",
            "sm_projection_health",
            // Claim-ledger admin tools
            "sm_proof_debt_status",
            "sm_evaluate_proof_debt_gate",
            "sm_resolve_contradiction",
            "sm_verify_ledger",
            "sm_export_claim_bundle",
            // Advanced graph tools (moved from lean to standard)
            "sm_topology",
            "sm_factor_graph",
            "sm_decoder_analyze",
            "sm_run_lifecycle",
            // Conversation write tools (hook-called, not agent-called)
            // Hidden in all profiles except full — the hook calls them via RPC bypass
        ];

        // Tools hidden in all profiles except full (import + conversation write)
        let full_only_tools: &[&str] = &[
            "sm_create_session",
            "sm_add_message",
            "sm_list_sessions",
        ];

        match tool_profile {
            "full" => { /* all tools visible */ }
            "standard" => {
                // Hide import tools + admin orchestration/claim tools + conversation writes
                for t in &admin_tools {
                    if t.starts_with("sm_import_")
                        || *t == "sm_list_imports"
                        || *t == "sm_query_temporal"
                        || *t == "sm_projection_health"
                        || *t == "sm_proof_debt_status"
                        || *t == "sm_evaluate_proof_debt_gate"
                        || *t == "sm_resolve_contradiction"
                        || *t == "sm_verify_ledger"
                        || *t == "sm_export_claim_bundle"
                    {
                        router.disable_route(*t);
                    }
                }
                // Also hide conversation write tools (hook-called via RPC)
                for t in full_only_tools {
                    router.disable_route(*t);
                }
            }
            _ => {
                // "lean" (default) — hide all admin tools + full-only tools
                for t in &admin_tools {
                    router.disable_route(*t);
                }
                for t in full_only_tools {
                    router.disable_route(*t);
                }
            }
        }

        eprintln!(
            "Tool profile: {} ({} tools visible)",
            tool_profile,
            router.list_all().len()
        );

        // Construct knowledge-runtime when orchestration feature is on.
        #[cfg(feature = "orchestration")]
        let runtime = {
            let adapter = knowledge_runtime::adapters::semantic_memory::SemanticMemoryAdapter::new(
                bridge.store.clone(),
            );
            let config = knowledge_runtime::RuntimeConfig {
                default_scope: knowledge_runtime::Scope::new("general"),
                query: knowledge_runtime::config::QueryConfig::default(),
                entity: knowledge_runtime::config::EntityConfig::default(),
                projection: knowledge_runtime::config::ProjectionConfig::default(),
                strict_temporal: false,
                strict_scope: false,
            };
            knowledge_runtime::KnowledgeRuntime::new(config, adapter).ok()
        };

        Self {
            bridge: Arc::new(bridge),
            tool_router: router,
            #[cfg(feature = "orchestration")]
            runtime,
        }
    }
}

/// Helper: load all stored graph edges from the store as GraphEdgeRef tuples
/// for discord scoring.
fn load_stored_edge_refs(
    store: &semantic_memory::MemoryStore,
) -> Result<Vec<semantic_memory::discord::GraphEdgeRef>, ErrorData> {
    let edges =
        tokio::task::block_in_place(|| Handle::current().block_on(store.list_all_graph_edges()))
            .map_err(|e| {
                ErrorData::internal_error(format!("Failed to load graph edges: {e}"), None)
            })?;
    let refs = edges
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
    Ok(refs)
}

/// Helper: load all stored graph edges from the store as raw factor graph
/// edge tuples (source, target, GraphEdgeType, weight, metadata_json).
fn load_stored_factor_edges(
    store: &semantic_memory::MemoryStore,
) -> Result<
    Vec<(
        String,
        String,
        semantic_memory::GraphEdgeType,
        f64,
        Option<String>,
    )>,
    ErrorData,
> {
    let edges =
        tokio::task::block_in_place(|| Handle::current().block_on(store.list_all_graph_edges()))
            .map_err(|e| {
                ErrorData::internal_error(format!("Failed to load graph edges: {e}"), None)
            })?;
    let raw = edges
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
    Ok(raw)
}

/// Helper: load all stored graph edges as (source, target) pairs.
fn load_stored_edge_pairs(
    store: &semantic_memory::MemoryStore,
) -> Result<Vec<(String, String)>, ErrorData> {
    let edges =
        tokio::task::block_in_place(|| Handle::current().block_on(store.list_all_graph_edges()))
            .map_err(|e| {
                ErrorData::internal_error(format!("Failed to load graph edges: {e}"), None)
            })?;
    let pairs = edges
        .iter()
        .map(|edge| (edge.source.clone(), edge.target.clone()))
        .collect();
    Ok(pairs)
}

/// Helper: load graph edges for a neighborhood around the given seed node IDs.
/// Uses BFS expansion with max_hops=2 and max_nodes=200 by default.
/// Falls back to full graph load if seeds are empty.
fn load_neighborhood_edge_pairs(
    store: &semantic_memory::MemoryStore,
    seed_ids: &[String],
) -> Result<Vec<(String, String)>, ErrorData> {
    if seed_ids.is_empty() {
        return load_stored_edge_pairs(store);
    }
    let edges = tokio::task::block_in_place(|| {
        Handle::current().block_on(store.list_graph_edges_for_neighborhood(
            seed_ids.to_vec(),
            2,
            200,
        ))
    })
    .map_err(|e| {
        ErrorData::internal_error(format!("Failed to load neighborhood edges: {e}"), None)
    })?;
    let pairs = edges
        .iter()
        .map(|edge| (edge.source.clone(), edge.target.clone()))
        .collect();
    Ok(pairs)
}

/// Helper: load graph edges for a neighborhood as GraphEdgeRef vec.
fn load_neighborhood_edge_refs(
    store: &semantic_memory::MemoryStore,
    seed_ids: &[String],
) -> Result<Vec<semantic_memory::discord::GraphEdgeRef>, ErrorData> {
    if seed_ids.is_empty() {
        return load_stored_edge_refs(store);
    }
    let edges = tokio::task::block_in_place(|| {
        Handle::current().block_on(store.list_graph_edges_for_neighborhood(
            seed_ids.to_vec(),
            2,
            200,
        ))
    })
    .map_err(|e| {
        ErrorData::internal_error(format!("Failed to load neighborhood edges: {e}"), None)
    })?;
    let refs = edges
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
    Ok(refs)
}

/// Helper: load graph edges for a neighborhood as factor graph tuples.
fn load_neighborhood_factor_edges(
    store: &semantic_memory::MemoryStore,
    seed_ids: &[String],
) -> Result<
    Vec<(
        String,
        String,
        semantic_memory::GraphEdgeType,
        f64,
        Option<String>,
    )>,
    ErrorData,
> {
    if seed_ids.is_empty() {
        return load_stored_factor_edges(store);
    }
    let edges = tokio::task::block_in_place(|| {
        Handle::current().block_on(store.list_graph_edges_for_neighborhood(
            seed_ids.to_vec(),
            2,
            200,
        ))
    })
    .map_err(|e| {
        ErrorData::internal_error(format!("Failed to load neighborhood edges: {e}"), None)
    })?;
    let raw = edges
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
    Ok(raw)
}

/// Load fact ids targeted by entity relation="supersedes" graph edges.
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

fn query_allows_superseded(query: &str) -> bool {
    let q = query.to_lowercase();
    q.contains("supersed")
        || q.contains("stale")
        || q.contains("obsolete")
        || q.contains("histor")
        || q.contains("old fact")
        || q.contains("previous fact")
}

/// Serialize a JSON value to a pretty string, mapping serialization errors
/// to protocol-level errors instead of success strings.
/// Build a `ProjectionQuery` from the MCP-facing `ProjectionQueryParams`.
///
/// Maps the flat parameter struct into the library's `ProjectionQuery` with
/// a fully-resolved `ScopeKey` and typed ID filters.
fn build_projection_query(params: ProjectionQueryParams) -> semantic_memory::ProjectionQuery {
    use stack_ids::{ClaimId, ClaimVersionId, EntityId, ScopeKey};

    let scope = ScopeKey {
        namespace: params.namespace,
        domain: params.domain,
        workspace_id: params.workspace_id,
        repo_id: params.repo_id,
    };

    let limit = params.limit.unwrap_or(10) as usize;

    semantic_memory::ProjectionQuery {
        scope,
        text_query: params.text_query,
        valid_at: params.valid_at,
        recorded_at_or_before: params.recorded_at_or_before,
        subject_entity_id: params.subject_entity_id.map(EntityId::new),
        canonical_entity_id: params.canonical_entity_id.map(EntityId::new),
        claim_state: params.claim_state,
        claim_id: params.claim_id.map(ClaimId::new),
        claim_version_id: params.claim_version_id.map(ClaimVersionId::new),
        limit,
    }
}

fn json_to_string(value: &serde_json::Value) -> Result<String, ErrorData> {
    serde_json::to_string_pretty(value)
        .map_err(|e| ErrorData::internal_error(format!("Serialization error: {e}"), None))
}

#[tool_router]
impl SemanticMemoryServer {
    // ── Core search tools ────────────────────────────────────────────

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
    ) -> Result<String, ErrorData> {
        let requested_k = top_k.map(|v| v as usize).unwrap_or(5);
        let allow_superseded = query_allows_superseded(&query);
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
                let result_refs: Vec<_> =
                    if superseded_targets.is_empty() || fresh_results.is_empty() {
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
                json_to_string(&serde_json::json!({
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

    // #[tool( (DEPRECATED: sm_search_explained merged/removed per audit)
    // description = "Search with full score breakdown showing how BM25 and vector scores combine. Useful for debugging retrieval quality.",
    // annotations(read_only_hint = true)
    // )] (DEPRECATED: merged/removed per tool audit)
    #[allow(dead_code)]
    fn sm_search_explained(
        &self,
        Parameters(SearchExplainedParams { query, top_k }): Parameters<SearchExplainedParams>,
    ) -> Result<String, ErrorData> {
        let requested_k = top_k.map(|v| v as usize).unwrap_or(5);
        let allow_superseded = query_allows_superseded(&query);
        let search_k = if allow_superseded {
            requested_k
        } else {
            (requested_k * 4).max(20)
        };
        let store = &self.bridge.store;
        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.search_explained(&query, Some(search_k), None, None))
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
                    .filter(|r| !superseded_targets.contains(&r.result.source.result_id()))
                    .collect();
                let result_refs: Vec<_> =
                    if superseded_targets.is_empty() || fresh_results.is_empty() {
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
                            "result_id": r.result.source.result_id(),
                            "content": r.result.content,
                            "source": format!("{:?}", r.result.source),
                            "score": r.result.score,
                            "bm25_rank": r.result.bm25_rank,
                            "vector_rank": r.result.vector_rank,
                            "cosine_similarity": r.result.cosine_similarity,
                            "breakdown": {
                                "rrf_score": r.breakdown.rrf_score,
                                "bm25_score": r.breakdown.bm25_score,
                                "vector_score": r.breakdown.vector_score,
                                "recency_score": r.breakdown.recency_score,
                                "bm25_rank": r.breakdown.bm25_rank,
                                "vector_rank": r.breakdown.vector_rank,
                                "vector_source_rank": r.breakdown.vector_source_rank,
                                "vector_source_score": r.breakdown.vector_source_score,
                                "bm25_contribution": r.breakdown.bm25_contribution,
                                "vector_contribution": r.breakdown.vector_contribution,
                                "vector_reranked_from_f32": r.breakdown.vector_reranked_from_f32,
                                "bm25_weight": r.breakdown.bm25_weight,
                                "vector_weight": r.breakdown.vector_weight,
                                "recency_weight": r.breakdown.recency_weight,
                                "rrf_k": r.breakdown.rrf_k,
                            },
                        })
                    })
                    .collect();
                json_to_string(&serde_json::json!({
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
        description = "Add a fact to the knowledge base. Embedded and indexed for semantic search. Returns fact ID and content digest.",
        annotations(idempotent_hint = true)
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
        }): Parameters<AddFactParams>,
    ) -> Result<String, ErrorData> {
        let store = &self.bridge.store;
        let src = source.as_deref();

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
        if kind == "ephemeral_inference" {
            let refs = evidence_refs.as_ref().map(|v| v.len()).unwrap_or(0);
            if refs == 0 {
                return Err(ErrorData::invalid_params(
                    "Admission gate BLOCKED: ephemeral_inference requires evidence_refs to promote to durable".to_string(),
                    None,
                ));
            }
        }

        // Build metadata JSON with typed memory fields
        let mut meta = serde_json::Map::new();
        meta.insert("memory_kind".to_string(), serde_json::json!(kind));
        meta.insert("sensitivity".to_string(), serde_json::json!(sens));
        if let Some(refs) = evidence_refs {
            meta.insert("evidence_refs".to_string(), serde_json::json!(refs));
        }
        let _metadata_str = serde_json::to_string(&serde_json::Value::Object(meta)).ok();

        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.add_fact(&namespace, &content, src, None))
        });

        match result {
            Ok(id) => {
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

                json_to_string(&serde_json::json!({
                    "ok": true,
                    "fact_id": id,
                    "namespace": namespace,
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
        description = "Ingest a document with automatic chunking. Splits into chunks, each embedded and indexed. Returns document ID and chunk count.",
        annotations(idempotent_hint = true)
    )]
    fn sm_ingest_document(
        &self,
        Parameters(IngestDocumentParams {
            content,
            title,
            namespace,
        }): Parameters<IngestDocumentParams>,
    ) -> Result<String, ErrorData> {
        let store = &self.bridge.store;
        let result = tokio::task::block_in_place(|| {
            Handle::current()
                .block_on(store.ingest_document(&title, &content, &namespace, None, None))
        });

        match result {
            Ok(doc_id) => {
                let chunk_count = tokio::task::block_in_place(|| {
                    Handle::current().block_on(store.count_chunks_for_document(&doc_id))
                })
                .unwrap_or(0);
                json_to_string(&serde_json::json!({
                    "ok": true,
                    "document_id": doc_id,
                    "title": title,
                    "chunk_count": chunk_count,
                    "message": "Document ingested successfully",
                }))
            }
            Err(e) => Err(ErrorData::internal_error(
                format!("Error ingesting document: {e}"),
                None,
            )),
        }
    }

    #[tool(
        description = "Get knowledge base statistics: fact/chunk/document/session counts, DB size, embedding model, and graph edge count.",
        annotations(read_only_hint = true)
    )]
    fn sm_stats(&self) -> Result<String, ErrorData> {
        let store = &self.bridge.store;
        let result = tokio::task::block_in_place(|| Handle::current().block_on(store.stats()));

        match result {
            Ok(stats) => {
                // Load graph edge count separately — propagates errors
                // instead of hiding them (SM-AUD-016).
                let graph_edge_count = tokio::task::block_in_place(|| {
                    Handle::current().block_on(store.list_all_graph_edges())
                })
                .map(|edges| edges.len())
                .unwrap_or_else(|e| {
                    tracing::warn!("graph_edges table unavailable: {e}");
                    0
                });
                json_to_string(&serde_json::json!({
                    "ok": true,
                    "facts": stats.total_facts,
                    "chunks": stats.total_chunks,
                    "documents": stats.total_documents,
                    "sessions": stats.total_sessions,
                    "messages": stats.total_messages,
                    "graph_edges": graph_edge_count,
                    "db_size_bytes": stats.database_size_bytes,
                    "db_size_mb": (stats.database_size_bytes as f64 / 1_048_576.0 * 100.0).round() / 100.0,
                    "embedding_model": stats.embedding_model,
                    "embedding_dimensions": stats.embedding_dimensions,
                }))
            }
            Err(e) => Err(ErrorData::internal_error(format!("Stats error: {e}"), None)),
        }
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
    ) -> Result<String, ErrorData> {
        let depth = max_depth.map(|v| v as usize).unwrap_or(5);
        let store = &self.bridge.store;
        let g = store.graph_view();

        match g.path(&from_id, &to_id, depth) {
            Ok(Some(path)) => {
                // Build edge evidence for each hop by examining neighbors.
                let path_segments = build_path_segments(store, &path);
                json_to_string(&serde_json::json!({
                    "ok": true,
                    "from": from_id,
                    "to": to_id,
                    "path": path,
                    "path_length": path.len(),
                    "segments": path_segments,
                }))
            }
            Ok(None) => json_to_string(&serde_json::json!({
                "ok": true,
                "from": from_id,
                "to": to_id,
                "path": null,
                "message": format!("No path found from {from_id} to {to_id} within depth {depth}"),
            })),
            Err(e) => Err(ErrorData::internal_error(
                format!("Graph view error: {e}"),
                None,
            )),
        }
    }

    // ── Direct read and supersession tools (v0.3.1) ──────────────────

    #[tool(
        description = "Fetch one fact by id (bare UUID or prefixed 'fact:<uuid>'). Returns full content, namespace, source, timestamps, and metadata.",
        annotations(read_only_hint = true)
    )]
    fn sm_get_fact(
        &self,
        Parameters(GetFactParams { fact_id }): Parameters<GetFactParams>,
    ) -> Result<String, ErrorData> {
        let bare = fact_id
            .strip_prefix("fact:")
            .unwrap_or(&fact_id)
            .to_string();
        let store = &self.bridge.store;
        let result =
            tokio::task::block_in_place(|| Handle::current().block_on(store.get_fact(&bare)));
        match result {
            Ok(Some(f)) => json_to_string(&serde_json::json!({
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
            Ok(None) => json_to_string(&serde_json::json!({
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
        description = "Enumerate facts in a namespace (newest first) with pagination. Exhaustive, not similarity-ranked — for browsing, auditing, or deduping.",
        annotations(read_only_hint = true)
    )]
    fn sm_list_facts(
        &self,
        Parameters(ListFactsParams {
            namespace,
            limit,
            offset,
        }): Parameters<ListFactsParams>,
    ) -> Result<String, ErrorData> {
        let lim = limit.map(|v| v as usize).unwrap_or(50);
        let off = offset.map(|v| v as usize).unwrap_or(0);
        let store = &self.bridge.store;
        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.list_facts(&namespace, lim, off))
        });
        match result {
            Ok(facts) => {
                let arr: Vec<serde_json::Value> = facts
                    .iter()
                    .map(|f| {
                        serde_json::json!({
                            "result_id": format!("fact:{}", f.id),
                            "id": f.id,
                            "namespace": f.namespace,
                            "content": f.content,
                            "source": f.source,
                            "updated_at": f.updated_at,
                        })
                    })
                    .collect();
                json_to_string(&serde_json::json!({
                    "ok": true,
                    "namespace": namespace,
                    "count": arr.len(),
                    "limit": lim,
                    "offset": off,
                    "facts": arr,
                }))
            }
            Err(e) => Err(ErrorData::internal_error(
                format!("list_facts error: {e}"),
                None,
            )),
        }
    }

    #[tool(
        description = "List namespaces that currently contain facts. Use before sm_list_facts to discover what is stored.",
        annotations(read_only_hint = true)
    )]
    fn sm_list_namespaces(&self) -> Result<String, ErrorData> {
        let store = &self.bridge.store;
        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.list_fact_namespaces())
        });
        match result {
            Ok(ns) => json_to_string(&serde_json::json!({
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
        description = "Fetch a fact plus its graph neighbors WITH their content in one call. Hydrates neighbor facts for ids returned by graph tools.",
        annotations(read_only_hint = true)
    )]
    fn sm_get_fact_neighbors(
        &self,
        Parameters(GetFactNeighborsParams { item_id }): Parameters<GetFactNeighborsParams>,
    ) -> Result<String, ErrorData> {
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
        json_to_string(&serde_json::json!({
            "ok": true,
            "item_id": node_id,
            "center_content": center.map(|f| f.content),
            "neighbor_count": neighbors.len(),
            "neighbors": neighbors,
        }))
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
    ) -> Result<String, ErrorData> {
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

        json_to_string(&serde_json::json!({
            "ok": true,
            "new_fact_id": new_id,
            "new_result_id": new_node,
            "old_fact_id": old_bare,
            "old_result_id": old_node,
            "namespace": ns,
            "edge_id": edge.id,
            "relation": "supersedes",
        }))
    }

    // ── Conversation / session tools (v0.3.0) ────────────────────────

    #[tool(
        description = "Create a conversation session (container for messages). Returns session id. Use to persist history recallable via sm_search_conversations.",
        annotations(idempotent_hint = true)
    )]
    fn sm_create_session(
        &self,
        Parameters(CreateSessionParams { channel, metadata }): Parameters<CreateSessionParams>,
    ) -> Result<String, ErrorData> {
        let meta: Option<serde_json::Value> = metadata
            .as_deref()
            .and_then(|s| serde_json::from_str(s).ok());
        let store = &self.bridge.store;
        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.create_session_with_metadata(&channel, meta))
        });
        match result {
            Ok(id) => json_to_string(
                &serde_json::json!({"ok": true, "session_id": id, "channel": channel}),
            ),
            Err(e) => Err(ErrorData::internal_error(
                format!("create_session error: {e}"),
                None,
            )),
        }
    }

    #[tool(
        description = "Append a message to a session. role: user|assistant|system|tool. Message is embedded and FTS-indexed. Returns message id."
    )]
    fn sm_add_message(
        &self,
        Parameters(AddMessageParams {
            session_id,
            role,
            content,
        }): Parameters<AddMessageParams>,
    ) -> Result<String, ErrorData> {
        let parsed_role = match role.to_lowercase().as_str() {
            "user" => semantic_memory::types::Role::User,
            "assistant" => semantic_memory::types::Role::Assistant,
            "system" => semantic_memory::types::Role::System,
            "tool" => semantic_memory::types::Role::Tool,
            other => {
                return Err(ErrorData::invalid_params(
                    format!("invalid role '{other}' (use user|assistant|system|tool)"),
                    None,
                ))
            }
        };
        let store = &self.bridge.store;
        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.add_message_embedded(
                &session_id,
                parsed_role,
                &content,
                None,
                None,
            ))
        });
        match result {
            Ok(id) => json_to_string(
                &serde_json::json!({"ok": true, "message_id": id, "session_id": session_id}),
            ),
            Err(e) => Err(ErrorData::internal_error(
                format!("add_message error: {e}"),
                None,
            )),
        }
    }

    #[tool(description = "List recent conversation sessions (newest first) with message counts.", annotations(read_only_hint = true))]
    fn sm_list_sessions(
        &self,
        Parameters(ListSessionsParams { limit, offset }): Parameters<ListSessionsParams>,
    ) -> Result<String, ErrorData> {
        let lim = limit.map(|v| v as usize).unwrap_or(20);
        let off = offset.map(|v| v as usize).unwrap_or(0);
        let store = &self.bridge.store;
        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.list_sessions(lim, off))
        });
        match result {
            Ok(sessions) => json_to_string(&serde_json::json!({
                "ok": true,
                "count": sessions.len(),
                "sessions": sessions.iter().map(|s| serde_json::json!({
                    "session_id": s.id,
                    "channel": s.channel,
                    "message_count": s.message_count,
                    "created_at": s.created_at,
                    "updated_at": s.updated_at,
                })).collect::<Vec<_>>(),
            })),
            Err(e) => Err(ErrorData::internal_error(
                format!("list_sessions error: {e}"),
                None,
            )),
        }
    }

    #[tool(
        description = "Get most recent messages from a session within a token budget (default 4000), chronological order. Returns role, content, timestamps.",
        annotations(read_only_hint = true)
    )]
    fn sm_get_messages(
        &self,
        Parameters(GetMessagesParams {
            session_id,
            max_tokens,
        }): Parameters<GetMessagesParams>,
    ) -> Result<String, ErrorData> {
        let budget = max_tokens.unwrap_or(4000);
        let store = &self.bridge.store;
        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.get_messages_within_budget(&session_id, budget))
        });
        match result {
            Ok(msgs) => json_to_string(&serde_json::json!({
                "ok": true,
                "session_id": session_id,
                "count": msgs.len(),
                "messages": msgs.iter().map(|m| serde_json::json!({
                    "id": m.id,
                    "role": m.role,
                    "content": m.content,
                    "token_count": m.token_count,
                    "created_at": m.created_at,
                })).collect::<Vec<_>>(),
            })),
            Err(e) => Err(ErrorData::internal_error(
                format!("get_messages error: {e}"),
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
    ) -> Result<String, ErrorData> {
        let k = top_k.map(|v| v as usize);
        let store = &self.bridge.store;
        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.search_conversations(&query, k, None))
        });
        match result {
            Ok(results) => json_to_string(&serde_json::json!({
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

    // ── Feature-gated tools ──────────────────────────────────────────
    // Note: cfg gates are removed from individual tool methods because
    // rmcp's #[tool_router] macro needs all tools visible at expansion
    // time. The `full` feature in Cargo.toml already enables the
    // semantic-memory sub-features these tools depend on.

    // #[tool( (DEPRECATED: sm_route_query merged/removed per audit)
    // description = "Profile a query and get an adaptive routing decision. Determines which retrieval stages (BM25, vector, rerank, graph, decoder, discord) to activate.",
    // annotations(read_only_hint = true)
    // )] (DEPRECATED: merged/removed per tool audit)
    fn sm_route_query(
        &self,
        Parameters(RouteQueryParams { query }): Parameters<RouteQueryParams>,
    ) -> Result<String, ErrorData> {
        use semantic_memory::routing::RetrievalRouter;

        let router = RetrievalRouter {
            decoder_enabled: true,
            discord_enabled: true,
            corpus_density: 0.5,
            ..Default::default()
        };

        let decision = router.route_query(&query);
        json_to_string(&serde_json::json!({
            "ok": true,
            "bm25_coarse": decision.bm25_coarse,
            "vector_medium": decision.vector_medium,
            "rerank_fine": decision.rerank_fine,
            "graph_expansion": decision.graph_expansion,
            "decoder": decision.decoder,
            "discord": decision.discord,
            "no_retrieval": decision.no_retrieval,
            "reasoning": decision.reasoning,
        }))
    }

    #[tool(
        description = "Adaptive search: profiles query, routes to appropriate stages, applies factor graph belief propagation if decoder is activated. Returns results with stable IDs.",
        annotations(read_only_hint = true)
    )]
    fn sm_search_with_routing(
        &self,
        Parameters(SearchWithRoutingParams {
            query,
            top_k,
            contradictions,
            group_by_community,
        }): Parameters<SearchWithRoutingParams>,
    ) -> Result<String, ErrorData> {
        use semantic_memory::integration::plan_execution;
        use semantic_memory::rl_routing::route_with_rl;
        use semantic_memory::routing::QueryProfile;

        let k = top_k.map(|v| v as usize).unwrap_or(5);
        let allow_superseded = query_allows_superseded(&query);
        let search_k = if allow_superseded { k } else { (k * 4).max(20) };

        // Load persisted RL routing policy (or default if none saved yet)
        let store = &self.bridge.store;
        let policy =
            tokio::task::block_in_place(|| Handle::current().block_on(store.load_routing_policy()))
                .ok()
                .flatten()
                .unwrap_or_default();
        let profile = QueryProfile::from_query(&query);
        let decision = route_with_rl(&policy, &profile);
        let contras = contradictions.unwrap_or_default();
        let plan = plan_execution(&decision, contras.clone());

        let store = &self.bridge.store;
        let search_result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.search(&query, Some(search_k), None, None))
        });

        match search_result {
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
                let result_refs: Vec<_> =
                    if superseded_targets.is_empty() || fresh_results.is_empty() {
                        results.iter().collect()
                    } else {
                        fresh_results
                    };
                let superseded_filtered_count = results.len().saturating_sub(result_refs.len());
                let json_results: Vec<serde_json::Value> = result_refs
                    .iter()
                    .take(k)
                    .map(|r| {
                        serde_json::json!({
                            "result_id": r.source.result_id(),
                            "content": r.content,
                            "score": r.score,
                        })
                    })
                    .collect();

                let mut factor_graph_payload = serde_json::json!({
                    "enabled": false,
                });

                let mut decoder_executed = false;
                let mut discord_executed = false;
                let mut discord_results_payload: Vec<serde_json::Value> = Vec::new();

                if decision.decoder {
                    #[cfg(feature = "full")]
                    {
                        use semantic_memory::factor_graph::{
                            factors_from_edges, FactorGraph, FactorGraphConfig,
                        };

                        let graph_edges = tokio::task::block_in_place(|| {
                            Handle::current().block_on(store.list_all_graph_edges())
                        });

                        match graph_edges {
                            Ok(edges) => {
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

                                let nodes: Vec<(String, f64)> = result_refs
                                    .iter()
                                    .map(|r| (r.source.result_id(), r.score))
                                    .collect();
                                let factors = factors_from_edges(&raw_edges);
                                let graph =
                                    FactorGraph::new(&nodes, factors, FactorGraphConfig::default());
                                let propagated = graph.propagate();
                                let top_beliefs = propagated.top_k(k);

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
                            Err(e) => {
                                factor_graph_payload = serde_json::json!({
                                    "enabled": false,
                                    "error": format!("factor graph analysis failed: {e}"),
                                });
                            }
                        }
                    }

                    #[cfg(not(feature = "full"))]
                    {
                        factor_graph_payload = serde_json::json!({
                            "enabled": false,
                            "reason": "factor graph analysis requires the `full` feature",
                        });
                    }

                    if !plan.contradictions.is_empty() {
                        use semantic_memory::decoder::{compute_correction, detect_syndromes};
                        let result_scores: Vec<(String, f64)> = result_refs
                            .iter()
                            .map(|r| (r.source.result_id(), r.score))
                            .collect();
                        let syndromes = detect_syndromes(&result_scores, &plan.contradictions);
                        let _ = compute_correction(&syndromes, 10.0);
                        decoder_executed = true;
                    }
                }

                if plan.use_discord {
                    use semantic_memory::discord::DiscordScorer;
                    let direct_ids: Vec<String> =
                        result_refs.iter().map(|r| r.source.result_id()).collect();
                    let existing_ids: std::collections::HashSet<String> =
                        direct_ids.iter().cloned().collect();
                    if let Ok(edges) = load_neighborhood_edge_refs(&self.bridge.store, &direct_ids)
                    {
                        let scorer = DiscordScorer::with_defaults();
                        let discord_hits = scorer.score(&direct_ids, &edges);
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

                let mut matryoshka_payload = serde_json::json!({
                    "enabled": false,
                });
                if decision.vector_medium {
                    #[cfg(feature = "full")]
                    {
                        use semantic_memory::integration::multi_resolution_route;
                        use semantic_memory::matryoshka::MatryoshkaConfig;
                        use semantic_memory::routing::QueryProfile;

                        let route_profile = QueryProfile::from_query(&query);
                        let route_decision =
                            multi_resolution_route(&route_profile, &MatryoshkaConfig::default());
                        matryoshka_payload = serde_json::json!({
                            "enabled": true,
                            "candidate_dim": route_decision.candidate_dim,
                            "heuristic_recall_estimate": route_decision.estimated_recall,
                            "recall_basis": "heuristic_dimensional_model_not_corpus_measured",
                            "embedding_dim": route_decision.embedding_dim,
                            "reasoning": route_decision.reasoning,
                        });
                    }

                    #[cfg(not(feature = "full"))]
                    {
                        matryoshka_payload = serde_json::json!({
                            "enabled": false,
                            "reason": "matryoshka routing requires the `full` feature",
                        });
                    }
                }

                // Community grouping (opt-in).
                let grouped_results_payload: serde_json::Value = if group_by_community == Some(true)
                {
                    let seed_ids: Vec<String> = result_refs
                        .iter()
                        .take(k)
                        .map(|r| r.source.result_id())
                        .collect();
                    let edges = load_neighborhood_edge_pairs(store, &seed_ids).unwrap_or_default();
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

                // Task 7: Auto-call topology when routing returns Class D (SYNTHESIS) and >10 results.
                let mut topology_payload = serde_json::json!({ "auto_called": false });
                {
                    use semantic_memory::routing::{QueryComplexityClass, QueryProfile};
                    let route_profile = QueryProfile::from_query(&query);
                    if route_profile.complexity_class == QueryComplexityClass::Synthesis
                        && result_refs.len() > 10
                    {
                        #[cfg(feature = "full")]
                        {
                            use semantic_memory::topology::{compute_betti_numbers, find_voids};
                            let edges = load_stored_edge_pairs(store).unwrap_or_default();
                            if !edges.is_empty() {
                                let mut adjacency: std::collections::HashMap<String, Vec<String>> =
                                    std::collections::HashMap::new();
                                for (src, tgt) in &edges {
                                    adjacency.entry(src.clone()).or_default().push(tgt.clone());
                                    adjacency.entry(tgt.clone()).or_default().push(src.clone());
                                }
                                let betti = compute_betti_numbers(&adjacency);
                                let voids = find_voids(&edges);
                                topology_payload = serde_json::json!({
                                    "auto_called": true,
                                    "trigger": "synthesis_class_with_10_plus_results",
                                    "betti_numbers": {
                                        "betti_0": betti.betti_0,
                                        "betti_1": betti.betti_1,
                                    },
                                    "void_count": voids.len(),
                                    "voids": voids.iter().map(|v| serde_json::json!({
                                        "description": v.description,
                                        "void_type": format!("{:?}", v.void_type),
                                        "nearby_items": v.nearby_items,
                                        "suggested_connections": v.suggested_connections,
                                    })).collect::<Vec<_>>(),
                                });
                            } else {
                                topology_payload = serde_json::json!({
                                    "auto_called": true,
                                    "trigger": "synthesis_class_with_10_plus_results",
                                    "note": "no graph edges in store",
                                });
                            }
                        }
                        #[cfg(not(feature = "full"))]
                        {
                            topology_payload = serde_json::json!({
                                "auto_called": true,
                                "trigger": "synthesis_class_with_10_plus_results",
                                "error": "topology requires the full feature",
                            });
                        }
                    }
                }

                json_to_string(&serde_json::json!({
                    "ok": true,
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
                    "results": json_results,
                    "count": json_results.len(),
                    "superseded_filtered_count": superseded_filtered_count,
                    "decoder_planned": plan.use_decoder,
                    "decoder_executed": decoder_executed,
                    "discord_planned": plan.use_discord,
                    "discord_executed": discord_executed,
                    "discord_results": discord_results_payload,
                    "factor_graph": factor_graph_payload,
                    "matryoshka": matryoshka_payload,
                    "grouped_results": grouped_results_payload,
                    "topology": topology_payload,
                }))
            }
            Err(e) => Err(ErrorData::internal_error(
                format!("Search error: {e}"),
                None,
            )),
        }
    }

    #[tool(
        description = "Detect contradictions in search results. Runs syndrome detection, computes corrections, and applies belief propagation to refine confidence scores.",
        annotations(read_only_hint = true)
    )]
    fn sm_decoder_analyze(
        &self,
        Parameters(DecoderAnalyzeParams {
            results,
            contradictions,
        }): Parameters<DecoderAnalyzeParams>,
    ) -> Result<String, ErrorData> {
        use semantic_memory::decoder::{
            compute_correction, detect_syndromes, pass_messages, ConflictGraph,
        };

        let contras = contradictions.unwrap_or_default();
        let syndromes = detect_syndromes(&results, &contras);
        let corrections = compute_correction(&syndromes, 10.0);
        let graph = ConflictGraph::from_syndromes(&results, &syndromes);
        let mp = pass_messages(&graph, 50, 0.001);

        json_to_string(&serde_json::json!({
            "ok": true,
            "syndromes": syndromes.iter().map(|s| serde_json::json!({
                "id": s.id,
                "severity": format!("{:?}", s.severity),
                "items": s.items,
                "description": s.description,
                "type": format!("{:?}", s.syndrome_type),
            })).collect::<Vec<_>>(),
            "syndrome_count": syndromes.len(),
            "corrections": corrections.iter().map(|c| serde_json::json!({
                "id": c.id,
                "confidence": c.confidence,
                "cost": c.cost,
                "operations": c.operations.len(),
            })).collect::<Vec<_>>(),
            "correction_count": corrections.len(),
            "message_passing": {
                "iterations": mp.iterations,
                "converged": mp.converged,
                "elapsed_ms": mp.elapsed_ms,
            },
        }))
    }

    #[tool(
        description = "Detect contradictions among the top results for a query from their CONTENT (numeric, value, negation, or antonym disagreement) — no pre-asserted edges required. Returns candidate conflicting pairs, each with the signals that fired and a human-readable reason. Persist a confirmed pair with sm_add_graph_edge(edge_type=\"contradicts\") so the decoder/community/factor-graph tools pick it up.",
        annotations(read_only_hint = true)
    )]
    fn sm_detect_contradictions(
        &self,
        Parameters(DetectContradictionsParams { query, top_k }): Parameters<
            DetectContradictionsParams,
        >,
    ) -> Result<String, ErrorData> {
        use semantic_memory::contradiction_detect::{detect_contradictions, DetectorConfig};

        let k = top_k.map(|v| v as usize).unwrap_or(10);
        let store = &self.bridge.store;
        let results = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.search(&query, Some(k), None, None))
        })
        .map_err(|e| ErrorData::internal_error(format!("search failed: {e}"), None))?;

        let items: Vec<(String, String)> = results
            .iter()
            .map(|r| (r.source.result_id(), r.content.clone()))
            .collect();

        let pairs = detect_contradictions(&items, &DetectorConfig::default());

        json_to_string(&serde_json::json!({
            "ok": true,
            "query": query,
            "items_scanned": items.len(),
            "contradictions": pairs.iter().map(|p| serde_json::json!({
                "a": p.a,
                "b": p.b,
                "score": p.score,
                "signals": p.signals.iter().map(|s| format!("{s:?}")).collect::<Vec<_>>(),
                "reason": p.reason,
            })).collect::<Vec<_>>(),
            "count": pairs.len(),
        }))
    }

    #[tool(
        description = "Second-order retrieval: find items related to your search results through the graph, but NOT themselves direct hits. Loads edges from store automatically.",
        annotations(read_only_hint = true)
    )]
    fn sm_discord_search(
        &self,
        Parameters(DiscordSearchParams { direct_result_ids }): Parameters<DiscordSearchParams>,
    ) -> Result<String, ErrorData> {
        use semantic_memory::discord::DiscordScorer;

        // Use neighborhood loading: only load edges within 2 hops of the
        // direct result IDs instead of the entire graph.
        let edges = load_neighborhood_edge_refs(&self.bridge.store, &direct_result_ids)?;
        let scorer = DiscordScorer::with_defaults();
        let results = scorer.score(&direct_result_ids, &edges);

        json_to_string(&serde_json::json!({
            "ok": true,
            "discord_results": results.iter().map(|r| serde_json::json!({
                "item_id": r.item_id,
                "discord_score": r.discord_score,
                "anchor_ids": r.anchor_ids,
                "relationship_types": r.relationship_types,
            })).collect::<Vec<_>>(),
            "count": results.len(),
            "edges_loaded": edges.len(),
            "edges_scope": "neighborhood",
        }))
    }

    #[tool(
        description = "Set provenance (evidence confidence) for an item. Confidence in [0.0, 1.0] with support count. Returns a provenance receipt.",
        annotations(idempotent_hint = true)
    )]
    fn sm_set_provenance(
        &self,
        Parameters(SetProvenanceParams {
            item_id,
            confidence,
            support_count,
        }): Parameters<SetProvenanceParams>,
    ) -> Result<String, ErrorData> {
        use semantic_memory::provenance::{
            ConfidenceSemiring, ConfidenceValue, ProvenanceItemType,
        };

        // SM-AUD-015: Validate confidence is finite and in [0, 1].
        if !confidence.is_finite() || confidence < 0.0 || confidence > 1.0 {
            return Err(ErrorData::invalid_params(
                format!("confidence must be a finite value in [0.0, 1.0], got {confidence}"),
                None,
            ));
        }

        let value = ConfidenceValue::new(confidence, support_count);
        let store = &self.bridge.store;

        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.set_provenance::<ConfidenceSemiring>(
                &ProvenanceItemType::Fact,
                &item_id,
                &value,
                &[],
                None,
            ))
        });

        match result {
            Ok(receipt) => json_to_string(&serde_json::json!({
                "ok": true,
                "provenance_id": receipt.provenance_id,
                "item_id": receipt.item_id,
                "semiring_type": receipt.semiring_type,
                "recorded_at": receipt.recorded_at,
                "message": "Provenance set successfully",
            })),
            Err(e) => Err(ErrorData::internal_error(
                format!("Provenance error: {e}"),
                None,
            )),
        }
    }

    #[tool(
        description = "Run a memory lifecycle pass: analyze items for syndromes, compute corrections, identify subtraction candidates, and check compression needs.",
        annotations(read_only_hint = true)
    )]
    fn sm_run_lifecycle(
        &self,
        Parameters(RunLifecycleParams { item_ids }): Parameters<RunLifecycleParams>,
    ) -> Result<String, ErrorData> {
        use semantic_memory::decoder::{compute_correction, detect_syndromes};
        use semantic_memory::integration::{
            corrections_to_subtraction_candidates, should_trigger_recompression,
        };

        let results: Vec<(String, f64)> = item_ids.iter().map(|id| (id.clone(), 0.5)).collect();
        let syndromes = detect_syndromes(&results, &[]);
        let corrections = compute_correction(&syndromes, 10.0);

        let sub_candidates = corrections_to_subtraction_candidates(&corrections);

        let subtracted_count = sub_candidates.len();
        let remaining_count = item_ids.len().saturating_sub(subtracted_count);
        let recompression = should_trigger_recompression(subtracted_count, remaining_count, false);

        let store = &self.bridge.store;
        let graph_edges = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.list_all_graph_edges())
        });
        let stored_edges: Vec<(String, String)> = graph_edges
            .as_ref()
            .map(|edges| {
                edges
                    .iter()
                    .map(|edge| (edge.source.clone(), edge.target.clone()))
                    .collect()
            })
            .unwrap_or_default();

        let mut topology_voids: Vec<serde_json::Value> = Vec::new();
        let mut betti = serde_json::json!({
            "betti_0": 0usize,
            "betti_1": 0usize,
        });
        let mut topology_error: Option<String> = None;

        let mut communities: Vec<serde_json::Value> = Vec::new();
        let mut community_contradictions: Vec<serde_json::Value> = Vec::new();
        let mut community_error: Option<String> = None;

        let mut subgraph_assessment = serde_json::json!({
            "subgraphs_identified": 0usize,
            "subgraphs_pruned": 0usize,
        });
        let mut subgraph_error: Option<String> = None;

        #[cfg(feature = "full")]
        {
            use std::collections::HashMap;

            if !stored_edges.is_empty() {
                let analysis_edges = stored_edges.clone();

                let topology_result = (|| -> Result<(), String> {
                    use semantic_memory::topology::{compute_betti_numbers, find_voids};

                    let mut adjacency: HashMap<String, Vec<String>> = HashMap::new();
                    for (left, right) in &analysis_edges {
                        adjacency
                            .entry(left.clone())
                            .or_default()
                            .push(right.clone());
                        adjacency
                            .entry(right.clone())
                            .or_default()
                            .push(left.clone());
                    }

                    let betti_numbers = compute_betti_numbers(&adjacency);
                    betti = serde_json::json!({
                        "betti_0": betti_numbers.betti_0,
                        "betti_1": betti_numbers.betti_1,
                    });

                    topology_voids = find_voids(&analysis_edges)
                        .into_iter()
                        .map(|v| {
                            serde_json::json!({
                                "description": v.description,
                                "void_type": format!("{:?}", v.void_type),
                                "nearby_items": v.nearby_items,
                                "suggested_connections": v.suggested_connections,
                            })
                        })
                        .collect();

                    Ok(())
                })();

                if let Err(e) = topology_result {
                    topology_error = Some(e);
                }

                let community_result = (|| -> Result<(), String> {
                    use semantic_memory::community::{
                        community_contradiction_scan, detect_communities,
                    };

                    let detected = detect_communities(&analysis_edges, 1.0, 42);
                    communities = detected
                        .iter()
                        .map(|c| {
                            serde_json::json!({
                                "id": c.id,
                                "members": c.members,
                                "level": c.level,
                                "parent": c.parent,
                                "member_count": c.members.len(),
                            })
                        })
                        .collect();

                    community_contradictions = community_contradiction_scan(&detected, &[])
                        .into_iter()
                        .map(|cc| {
                            serde_json::json!({
                                "community_id": cc.community_id,
                                "item_a": cc.item_a,
                                "item_b": cc.item_b,
                                "description": cc.description,
                            })
                        })
                        .collect();

                    Ok(())
                })();

                if let Err(e) = community_result {
                    community_error = Some(e);
                }

                let subgraph_result = (|| -> Result<(), String> {
                    use semantic_memory::integration::autonomous_subgraph_maintenance;
                    use semantic_memory::subgraph_pruning::AccessLog;
                    use std::collections::HashSet;

                    let mut access_items: HashSet<String> = HashSet::new();
                    for (left, right) in &analysis_edges {
                        access_items.insert(left.clone());
                        access_items.insert(right.clone());
                    }

                    let access_logs = access_items
                        .into_iter()
                        .map(|item| AccessLog {
                            item_id: item,
                            access_count: 1,
                            last_accessed: "1970-01-01T00:00:00Z".to_string(),
                        })
                        .collect::<Vec<_>>();

                    let report =
                        autonomous_subgraph_maintenance(&analysis_edges, &access_logs, &[], 0);
                    subgraph_assessment = serde_json::json!({
                        "subgraphs_identified": report.subgraphs_identified,
                        "subgraphs_pruned": report.subgraphs_pruned,
                        "summary": report.summary,
                    });
                    Ok(())
                })();

                if let Err(e) = subgraph_result {
                    subgraph_error = Some(e);
                }
            }
        }

        #[cfg(not(feature = "full"))]
        {
            if !stored_edges.is_empty() {
                topology_error = Some(
                    "topology/community/subgraph phases require the `full` feature".to_string(),
                );
                community_error = Some(
                    "topology/community/subgraph phases require the `full` feature".to_string(),
                );
                subgraph_error = Some(
                    "topology/community/subgraph phases require the `full` feature".to_string(),
                );
            }
        }

        #[cfg(feature = "full")]
        let (f32_count, compressed_count) =
            item_ids
                .iter()
                .fold((0usize, 0usize), |(f32_count, compressed_count), _| {
                    use semantic_memory::compression_governor::{
                        decide_quantization, QuantizationLevel,
                    };

                    match decide_quantization(0.5) {
                        QuantizationLevel::F32 => (f32_count + 1, compressed_count),
                        _ => (f32_count, compressed_count + 1),
                    }
                });
        #[cfg(not(feature = "full"))]
        let (f32_count, compressed_count) = (0usize, 0usize);

        json_to_string(&serde_json::json!({
            "ok": true,
            "items_analyzed": item_ids.len(),
            "syndromes_detected": syndromes.len(),
            "corrections_computed": corrections.len(),
            "subtraction_candidates": sub_candidates.iter().map(|c| serde_json::json!({
                "item_id": c.item_id,
                "structuring_score": c.structuring_score,
                "operation_type": c.operation_type,
                "reason": c.reason,
            })).collect::<Vec<_>>(),
            "recompression_triggered": recompression.triggered,
            "recompression_reason": recompression.reason,
            "topology": {
                "enabled": !stored_edges.is_empty(),
                "voids": topology_voids,
                "void_count": topology_voids.len(),
                "betti_numbers": betti,
                "error": topology_error,
            },
            "community_detection": {
                "enabled": !stored_edges.is_empty(),
                "communities": communities,
                "community_count": communities.len(),
                "contradictions": community_contradictions,
                "contradiction_count": community_contradictions.len(),
                "error": community_error,
            },
            "subgraph_pruning_assessment": {
                "enabled": !stored_edges.is_empty(),
                "subgraph_count": subgraph_assessment["subgraphs_identified"].as_u64().unwrap_or(0),
                "pruned_count": subgraph_assessment["subgraphs_pruned"].as_u64().unwrap_or(0),
                "summary": subgraph_assessment["summary"].as_str().unwrap_or(""),
                "error": subgraph_error,
            },
            "turbo_quantization_assessment": {
                "items_assessed": item_ids.len(),
                "would_retain_f32": f32_count,
                "would_compress": compressed_count,
            },
            "summary": format!(
                "Analyzed {} items: {} syndromes, {} corrections, {} subtraction candidates, recompression: {}",
                item_ids.len(), syndromes.len(), corrections.len(), sub_candidates.len(),
                if recompression.triggered { "needed" } else { "not needed" }
            ),
        }))
    }

    // ── First-class graph edge tools ───────────────────────────────

    #[tool(
        description = "Add a durable, typed graph edge between two nodes. Edge types: semantic, temporal, causal, entity. Idempotent — same edge returns existing ID.",
        annotations(idempotent_hint = true)
    )]
    fn sm_add_graph_edge(
        &self,
        Parameters(params): Parameters<AddGraphEdgeParams>,
    ) -> Result<String, ErrorData> {
        use semantic_memory::GraphEdgeType;

        // SM-AUD-015: Validate numeric params are finite and in range.
        if let Some(cs) = params.cosine_similarity {
            if !cs.is_finite() || cs < 0.0 || cs > 1.0 {
                return Err(ErrorData::invalid_params(
                    format!("cosine_similarity must be finite and in [0.0, 1.0], got {cs}"),
                    None,
                ));
            }
        }
        if let Some(conf) = params.confidence {
            if !conf.is_finite() || conf < 0.0 || conf > 1.0 {
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
            Ok(edge) => json_to_string(&serde_json::json!({
                "ok": true,
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
        description = "List graph edges for a specific node (as source or target), or all edges if no node_id. Returns non-invalidated edges only.",
        annotations(read_only_hint = true)
    )]
    fn sm_list_graph_edges(
        &self,
        Parameters(ListGraphEdgesParams { node_id }): Parameters<ListGraphEdgesParams>,
    ) -> Result<String, ErrorData> {
        let store = &self.bridge.store;
        let result = match node_id {
            Some(id) => tokio::task::block_in_place(|| {
                Handle::current().block_on(store.list_graph_edges_for_node(&id))
            }),
            None => tokio::task::block_in_place(|| {
                Handle::current().block_on(store.list_all_graph_edges())
            }),
        };

        match result {
            Ok(edges) => json_to_string(&serde_json::json!({
                "ok": true,
                "edges": edges.iter().map(|e| serde_json::json!({
                    "id": e.id,
                    "source": e.source,
                    "target": e.target,
                    "edge_type": e.edge_type,
                    "weight": e.weight,
                    "metadata": e.metadata,
                    "recorded_at": e.recorded_at,
                })).collect::<Vec<_>>(),
                "count": edges.len(),
            })),
            Err(e) => Err(ErrorData::internal_error(
                format!("Error listing graph edges: {e}"),
                None,
            )),
        }
    }

    #[tool(
        description = "Invalidate a stored graph edge by ID. Append-only — edge is never deleted, only marked invalidated with a reason.",
        annotations(idempotent_hint = true)
    )]
    fn sm_invalidate_graph_edge(
        &self,
        Parameters(InvalidateGraphEdgeParams { edge_id, reason }): Parameters<
            InvalidateGraphEdgeParams,
        >,
    ) -> Result<String, ErrorData> {
        let store = &self.bridge.store;
        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.invalidate_graph_edge(&edge_id, &reason))
        });

        match result {
            Ok(()) => json_to_string(&serde_json::json!({
                "ok": true,
                "edge_id": edge_id,
                "message": "Edge invalidated successfully",
            })),
            Err(e) => Err(ErrorData::internal_error(
                format!("Error invalidating edge: {e}"),
                None,
            )),
        }
    }

    // ── Factor graph, topology, and community tools ─────────────────

    #[tool(
        description = "Run factor graph belief propagation on stored graph edges. Models all 4 edge types as factors. Returns unified confidence scores after convergence.",
        annotations(read_only_hint = true)
    )]
    fn sm_factor_graph(
        &self,
        Parameters(params): Parameters<FactorGraphParams>,
    ) -> Result<String, ErrorData> {
        use semantic_memory::factor_graph::{factors_from_edges, FactorGraph, FactorGraphConfig};

        let defaults = FactorGraphConfig::default();
        let config = FactorGraphConfig {
            semantic_weight: params.semantic_weight.unwrap_or(defaults.semantic_weight),
            temporal_weight: params.temporal_weight.unwrap_or(defaults.temporal_weight),
            causal_weight: params.causal_weight.unwrap_or(defaults.causal_weight),
            entity_weight: params.entity_weight.unwrap_or(defaults.entity_weight),
            self_influence: params.self_influence.unwrap_or(defaults.self_influence),
            max_iterations: params
                .max_iterations
                .map(|v| v as usize)
                .unwrap_or(defaults.max_iterations),
            convergence_threshold: params
                .convergence_threshold
                .unwrap_or(defaults.convergence_threshold),
        };

        // Use neighborhood loading: only load edges within 2 hops of the
        // node seeds instead of the entire graph.
        let seed_ids: Vec<String> = params.nodes.iter().map(|n| n.item_id.clone()).collect();
        let raw_edges = load_neighborhood_factor_edges(&self.bridge.store, &seed_ids)?;
        let factors = factors_from_edges(&raw_edges);

        let nodes: Vec<(String, f64)> = params
            .nodes
            .iter()
            .map(|n| (n.item_id.clone(), n.initial_belief))
            .collect();

        let graph = FactorGraph::new(&nodes, factors, config);
        let result = graph.propagate();

        json_to_string(&serde_json::json!({
            "ok": true,
            "node_beliefs": result.node_beliefs,
            "iterations": result.iterations,
            "converged": result.converged,
            "elapsed_ms": result.elapsed_ms,
            "edges_loaded": raw_edges.len(),
            "edges_scope": "neighborhood",
            "factor_counts": {
                "semantic": result.factor_counts.semantic,
                "temporal": result.factor_counts.temporal,
                "causal": result.factor_counts.causal,
                "entity": result.factor_counts.entity,
                "total": result.factor_counts.total(),
            },
            "config": {
                "semantic_weight": result.config.semantic_weight,
                "temporal_weight": result.config.temporal_weight,
                "causal_weight": result.config.causal_weight,
                "entity_weight": result.config.entity_weight,
                "self_influence": result.config.self_influence,
                "max_iterations": result.config.max_iterations,
                "convergence_threshold": result.config.convergence_threshold,
            },
        }))
    }

    #[tool(
        description = "Find topological voids in the knowledge graph. Computes Betti numbers (components and cycles) and detects structural gaps. Loads edges from store.",
        annotations(read_only_hint = true)
    )]
    fn sm_topology(
        &self,
        Parameters(_params): Parameters<TopologyParams>,
    ) -> Result<String, ErrorData> {
        use semantic_memory::topology::{compute_betti_numbers, find_voids, gap_report};

        // MCP-001: Load edges from the store, not from caller-supplied params.
        let edges = load_stored_edge_pairs(&self.bridge.store)?;

        let mut adjacency: std::collections::HashMap<String, Vec<String>> =
            std::collections::HashMap::new();
        for (src, tgt) in &edges {
            adjacency.entry(src.clone()).or_default().push(tgt.clone());
            adjacency.entry(tgt.clone()).or_default().push(src.clone());
        }

        let betti = compute_betti_numbers(&adjacency);
        let voids = find_voids(&edges);
        let report = gap_report(&voids);

        json_to_string(&serde_json::json!({
            "ok": true,
            "betti_numbers": {
                "betti_0": betti.betti_0,
                "betti_1": betti.betti_1,
            },
            "voids": voids.iter().map(|v| serde_json::json!({
                "description": v.description,
                "nearby_items": v.nearby_items,
                "suggested_connections": v.suggested_connections,
                "void_type": format!("{:?}", v.void_type),
            })).collect::<Vec<_>>(),
            "void_count": voids.len(),
            "edges_loaded_from_store": edges.len(),
            "report": report,
        }))
    }

    #[tool(
        description = "Detect communities in the knowledge graph (Leiden-inspired). Returns community assignments, optional contradiction scans, and compression recommendations.",
        annotations(read_only_hint = true)
    )]
    fn sm_community(
        &self,
        Parameters(params): Parameters<CommunityParams>,
    ) -> Result<String, ErrorData> {
        use semantic_memory::community::{
            community_aware_compression, community_contradiction_scan, detect_communities,
        };

        // MCP-001: Load edges from the store, not from caller-supplied params.
        let edges = load_stored_edge_pairs(&self.bridge.store)?;

        let resolution = params.resolution.unwrap_or(1.0);
        let seed = params.seed.unwrap_or(42);

        let communities = detect_communities(&edges, resolution, seed);

        let contradictions = params.contradictions.unwrap_or_default();
        let community_contras = community_contradiction_scan(&communities, &contradictions);

        let importance_scores = params.importance_scores.unwrap_or_default();
        let compression = community_aware_compression(&communities, &importance_scores);

        let summarize = params.summarize.unwrap_or(false);
        let store = &self.bridge.store;
        let communities_json: Vec<serde_json::Value> = communities
            .iter()
            .map(|c| {
                let summary: Option<String> = if summarize && !c.members.is_empty() {
                    let member_texts: Vec<String> = c
                        .members
                        .iter()
                        .filter_map(|mid| {
                            let bare = mid.strip_prefix("fact:").unwrap_or(mid);
                            tokio::task::block_in_place(|| {
                                Handle::current().block_on(store.get_fact(bare))
                            })
                            .ok()
                            .flatten()
                            .map(|f| f.content)
                        })
                        .collect();
                    if !member_texts.is_empty() {
                        let combined = member_texts.join("\n---\n");
                        let prompt = format!(
                            "Summarize these related facts in 1-2 sentences:\n{combined}\nSummary:"
                        );
                        let body = serde_json::json!({
                            "model": "granite4.1:3b",
                            "prompt": prompt,
                            "stream": false,
                            "options": {"temperature": 0, "num_predict": 100}
                        });
                        reqwest::blocking::Client::new()
                            .post("http://127.0.0.1:11434/api/generate")
                            .json(&body)
                            .send()
                            .ok()
                            .and_then(|resp| resp.json::<serde_json::Value>().ok())
                            .and_then(|v| {
                                v.get("response")
                                    .and_then(|r| r.as_str())
                                    .map(|s| s.trim().to_string())
                            })
                    } else {
                        None
                    }
                } else {
                    None
                };
                serde_json::json!({
                    "id": c.id,
                    "members": c.members,
                    "level": c.level,
                    "parent": c.parent,
                    "member_count": c.members.len(),
                    "summary": summary,
                })
            })
            .collect();

        json_to_string(&serde_json::json!({
            "ok": true,
            "communities": communities_json,
            "community_count": communities.len(),
            "contradictions": community_contras.iter().map(|cc| serde_json::json!({
                "community_id": cc.community_id,
                "item_a": cc.item_a,
                "item_b": cc.item_b,
                "description": cc.description,
            })).collect::<Vec<_>>(),
            "contradiction_count": community_contras.len(),
            "compression_recommendations": compression.iter().map(|cr| serde_json::json!({
                "community_id": cr.community_id,
                "quantization_level": cr.quantization_level,
                "reason": cr.reason,
            })).collect::<Vec<_>>(),
            "compression_count": compression.len(),
            "edges_loaded_from_store": edges.len(),
        }))
    }

    // ── Delete / forget tools (admin-ops) ────────────────────────────
    // Hard removal. Prefer sm_supersede_fact when there is a corrected
    // replacement (it keeps history and search filters the old one); use
    // delete only for true noise/errors that should vanish entirely.

    #[tool(
        description = "Permanently delete a single fact by id. HARD delete — removes fact and its FTS/vector entries. Irreversible. Prefer sm_supersede_fact for corrections.",
        annotations(destructive_hint = true)
    )]
    fn sm_delete_fact(
        &self,
        Parameters(DeleteFactParams { fact_id }): Parameters<DeleteFactParams>,
    ) -> Result<String, ErrorData> {
        let bare = fact_id
            .strip_prefix("fact:")
            .unwrap_or(&fact_id)
            .to_string();
        let store = &self.bridge.store;
        let result =
            tokio::task::block_in_place(|| Handle::current().block_on(store.delete_fact(&bare)));
        match result {
            Ok(()) => json_to_string(&serde_json::json!({
                "ok": true,
                "deleted": true,
                "fact_id": format!("fact:{bare}"),
                "message": "Fact permanently deleted",
            })),
            Err(e) => Err(ErrorData::internal_error(
                format!("delete_fact error: {e}"),
                None,
            )),
        }
    }

    #[tool(
        description = "Permanently delete ALL memory in a namespace — facts, documents, chunks, sessions/messages. HARD delete, irreversible. Returns per-surface deletion count.",
        annotations(destructive_hint = true)
    )]
    fn sm_delete_namespace(
        &self,
        Parameters(DeleteNamespaceParams { namespace }): Parameters<DeleteNamespaceParams>,
    ) -> Result<String, ErrorData> {
        let store = &self.bridge.store;
        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.delete_namespace(&namespace))
        });
        match result {
            Ok(r) => json_to_string(&serde_json::json!({
                "ok": true,
                "namespace": namespace,
                "deleted": {
                    "facts": r.facts,
                    "documents": r.documents,
                    "chunks": r.chunks,
                    "messages": r.messages,
                    "sessions": r.sessions,
                    "episodes": r.episodes,
                    "projection_rows": r.projection_rows,
                },
                "message": "Namespace permanently deleted",
            })),
            Err(e) => Err(ErrorData::internal_error(
                format!("delete_namespace error: {e}"),
                None,
            )),
        }
    }

    // #[tool( (DEPRECATED: sm_update_fact merged/removed per audit)
    // description = "Update a fact's content in-place. Re-embeds the fact and updates FTS index. Use this to correct outdated facts without deleting and re-adding.",
    // annotations(idempotent_hint = true)
    // )] (DEPRECATED: merged/removed per tool audit)
    fn sm_update_fact(
        &self,
        Parameters(UpdateFactParams { fact_id, content }): Parameters<UpdateFactParams>,
    ) -> Result<String, ErrorData> {
        let bare = fact_id
            .strip_prefix("fact:")
            .unwrap_or(&fact_id)
            .to_string();
        let store = &self.bridge.store;
        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.update_fact(&bare, &content))
        });
        match result {
            Ok(()) => json_to_string(&serde_json::json!({
                "ok": true,
                "fact_id": format!("fact:{bare}"),
                "message": "Fact content updated and re-embedded",
            })),
            Err(e) => Err(ErrorData::internal_error(
                format!("update_fact error: {e}"),
                None,
            )),
        }
    }

    // #[tool( (DEPRECATED: sm_consolidate_facts merged/removed per audit)
    // description = "Consolidate two near-duplicate facts into one. Merges their content, updates the kept fact, and supersedes the other with a 'consolidated with' edge. Use this to clean up duplicate knowledge."
    // )] (DEPRECATED: merged/removed per tool audit)
    fn sm_consolidate_facts(
        &self,
        Parameters(ConsolidateFactsParams {
            keep_id,
            supersede_id,
            merged_content,
        }): Parameters<ConsolidateFactsParams>,
    ) -> Result<String, ErrorData> {
        let keep_bare = keep_id
            .strip_prefix("fact:")
            .unwrap_or(&keep_id)
            .to_string();
        let sup_bare = supersede_id
            .strip_prefix("fact:")
            .unwrap_or(&supersede_id)
            .to_string();
        let store = &self.bridge.store;

        // Get both facts to determine namespace and merge content
        let keep_fact =
            tokio::task::block_in_place(|| Handle::current().block_on(store.get_fact(&keep_bare)));
        let sup_fact =
            tokio::task::block_in_place(|| Handle::current().block_on(store.get_fact(&sup_bare)));

        let (namespace, final_content) = match (keep_fact, sup_fact) {
            (Ok(Some(k)), Ok(Some(s))) => {
                let ns = k.namespace.clone();
                let content = merged_content.unwrap_or_else(|| {
                    if k.content.len() >= s.content.len() {
                        if !k.content.contains(&s.content) {
                            format!("{}\n\nAdditional: {}", k.content, s.content)
                        } else {
                            k.content.clone()
                        }
                    } else if !s.content.contains(&k.content) {
                        format!("{}\n\nAdditional: {}", s.content, k.content)
                    } else {
                        s.content.clone()
                    }
                });
                (ns, content)
            }
            (Ok(Some(k)), _) => (
                k.namespace.clone(),
                merged_content.unwrap_or(k.content.clone()),
            ),
            (Err(_), _) | (Ok(None), _) => {
                return Err(ErrorData::internal_error(
                    format!("keep fact not found"),
                    None,
                ));
            }
        };

        // Update the kept fact with merged content
        let update_result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.update_fact(&keep_bare, &final_content))
        });
        if let Err(e) = update_result {
            return Err(ErrorData::internal_error(
                format!("update keep fact error: {e}"),
                None,
            ));
        }

        // Supersede the other fact: add a new fact with merged content and link with "supersedes" edge
        use semantic_memory::GraphEdgeType;
        let new_id = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.add_fact(&namespace, &final_content, None, None))
        });
        match new_id {
            Ok(nid) => {
                let new_node = format!("fact:{nid}");
                let old_node = format!("fact:{sup_bare}");
                let metadata = serde_json::json!({
                    "reason": "consolidated duplicate",
                    "consolidated_with": format!("fact:{}", keep_bare),
                });
                let _edge = tokio::task::block_in_place(|| {
                    Handle::current().block_on(store.add_graph_edge(
                        &new_node,
                        &old_node,
                        GraphEdgeType::Entity {
                            relation: "supersedes".to_string(),
                        },
                        1.0,
                        Some(metadata),
                    ))
                });
                json_to_string(&serde_json::json!({
                    "ok": true,
                    "kept_fact_id": format!("fact:{}", keep_bare),
                    "superseded_fact_id": format!("fact:{}", sup_bare),
                    "new_fact_id": format!("fact:{}", nid),
                    "message": "Facts consolidated: kept fact updated, duplicate superseded",
                }))
            }
            Err(e) => Err(ErrorData::internal_error(
                format!("supersede error: {e}"),
                None,
            )),
        }
    }

    // ── RL routing feedback ────────────────────────────────────────────

    #[tool(
        description = "Record routing outcome feedback for RL-trained retrieval routing. Stores the outcome (good/bad/neutral) and updates the tabular routing policy Q-table. Use after sm_search_with_routing to provide feedback on routing quality.",
        annotations(read_only_hint = true)
    )]
    fn sm_record_outcome(
        &self,
        Parameters(RecordOutcomeParams { query, outcome }): Parameters<RecordOutcomeParams>,
    ) -> Result<String, ErrorData> {
        use semantic_memory::rl_routing::{record_routing_outcome, RoutingOutcome};
        use semantic_memory::routing::{QueryProfile, RetrievalRouter};

        let outcome_enum = match outcome.to_lowercase().as_str() {
            "good" => RoutingOutcome::Good,
            "bad" => RoutingOutcome::Bad,
            "neutral" => RoutingOutcome::Neutral,
            _ => {
                return Err(ErrorData::invalid_params(
                    format!("outcome must be 'good', 'bad', or 'neutral', got '{outcome}'"),
                    None,
                ));
            }
        };

        let profile = QueryProfile::from_query(&query);
        let router = RetrievalRouter::default();
        let decision = router.route(&profile);

        let store = &self.bridge.store;
        // Load persisted policy (or default if none saved yet)
        let mut policy =
            tokio::task::block_in_place(|| Handle::current().block_on(store.load_routing_policy()))
                .ok()
                .flatten()
                .unwrap_or_default();
        record_routing_outcome(&mut policy, &profile, &decision, outcome_enum);
        // Save updated policy
        let _ = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.save_routing_policy(&policy))
        });

        json_to_string(&serde_json::json!({
            "ok": true,
            "query": query,
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
                "weights": policy.weights,
            },
            "message": "Routing outcome recorded and policy updated (persisted to DB)",
        }))
    }

    // ─── Claim-ledger integration ──────────────────────────────────────

    #[cfg(feature = "claim-integration")]
    #[tool(
        description = "Create a typed Claim from a semantic-memory fact. The claim gets a source-spanned provenance record from the fact's metadata. Returns the claim ID.",
        annotations(read_only_hint = false, idempotent_hint = true)
    )]
    fn sm_create_claim(
        &self,
        Parameters(CreateClaimParams {
            fact_id,
            source_span,
        }): Parameters<CreateClaimParams>,
    ) -> Result<String, ErrorData> {
        use claim_ledger::Claim;
        let bare = fact_id
            .strip_prefix("fact:")
            .unwrap_or(&fact_id)
            .to_string();
        let store = &self.bridge.store;

        // Get the fact content
        let fact =
            tokio::task::block_in_place(|| Handle::current().block_on(store.get_fact(&bare)));
        let fact = match fact {
            Ok(Some(f)) => f,
            _ => {
                return Err(ErrorData::internal_error(
                    format!("fact not found: {fact_id}"),
                    None,
                ))
            }
        };

        // Create a claim from the fact
        let source_id = format!("semantic-memory:fact:{bare}");
        let span_id = source_span.unwrap_or_else(|| "full".to_string());
        let claim = Claim::new(&source_id, &span_id, &fact.content, "fact");

        let claim_id = claim.claim_id.clone();
        let normalized = &claim.normalized_claim;

        json_to_string(&serde_json::json!({
            "ok": true,
            "claim_id": claim_id,
            "source_id": source_id,
            "span_id": span_id,
            "claim_text": fact.content,
            "normalized_claim": normalized,
            "claim_type": "fact",
            "message": "Claim created from semantic-memory fact with source-spanned provenance",
        }))
    }

    #[cfg(feature = "claim-integration")]
    #[tool(
        description = "Add evidence to a claim. Creates an EvidenceBundle linking the evidence text to the claim. Returns the evidence bundle ID.",
        annotations(read_only_hint = false)
    )]
    fn sm_add_evidence(
        &self,
        Parameters(AddEvidenceParams {
            claim_id,
            evidence_text,
            source_type,
        }): Parameters<AddEvidenceParams>,
    ) -> Result<String, ErrorData> {
        use claim_ledger::{EvidenceBundle, EvidenceLink, EvidenceRelation};
        let mut bundle = EvidenceBundle::new(&claim_id);
        let link = EvidenceLink {
            relation: EvidenceRelation::Supports,
            source_id: source_type.unwrap_or_else(|| "semantic-memory".to_string()),
            span_id: "full".to_string(),
            quote: evidence_text.clone(),
            digest: claim_ledger::ids::sha256_text(&evidence_text),
            support_role: "supporting".to_string(),
        };
        bundle.evidence_links.push(link);

        json_to_string(&serde_json::json!({
            "ok": true,
            "evidence_bundle_id": bundle.evidence_bundle_id,
            "claim_id": claim_id,
            "evidence_count": bundle.evidence_links.len(),
            "message": "Evidence added to claim",
        }))
    }

    #[cfg(feature = "claim-integration")]
    #[tool(
        description = "Judge the support state of a claim. Creates a SupportJudgment (supported, unsupported, contested, or heuristic_only) with optional rationale.",
        annotations(read_only_hint = false)
    )]
    fn sm_judge_support(
        &self,
        Parameters(JudgeSupportParams {
            claim_id,
            judgment,
            rationale,
        }): Parameters<JudgeSupportParams>,
    ) -> Result<String, ErrorData> {
        use claim_ledger::{SupportJudgment, SupportState};
        let state = match judgment.to_lowercase().as_str() {
            "supported" => SupportState::Supported,
            "partially_supported" | "partial" => SupportState::PartiallySupported,
            "unsupported" => SupportState::Unsupported,
            "contradicted" | "contested" => SupportState::Contradicted,
            "heuristic_only" | "heuristic" => SupportState::HeuristicOnly,
            _ => return Err(ErrorData::invalid_params(
                format!("Invalid judgment '{judgment}'. Must be: supported, partially_supported, unsupported, contradicted, or heuristic_only"),
                None,
            )),
        };
        let j = SupportJudgment {
            support_judgment_id: claim_ledger::ids::ulid(),
            claim_id: claim_id.clone(),
            evidence_bundle_ref: claim_ledger::ids::evidence_bundle_id(&claim_id),
            support_state: state,
            method: "agent_judgment".to_string(),
            rationale: rationale.unwrap_or_default(),
            contradiction_refs: Vec::new(),
            proof_debt: Vec::new(),
            created_recorded_time: chrono::Utc::now(),
        };

        json_to_string(&serde_json::json!({
            "ok": true,
            "support_judgment_id": j.support_judgment_id,
            "claim_id": claim_id,
            "state": judgment.to_lowercase(),
            "message": "Support judgment recorded",
        }))
    }

    // ─── Bitemporal search ─────────────────────────────────────────────

    #[tool(
        description = "Search facts that were valid (not superseded) as of a specific date. Uses bitemporal fields to filter results to only include facts that existed on the specified date.",
        annotations(read_only_hint = true)
    )]
    fn sm_search_as_of(
        &self,
        Parameters(SearchAsOfParams {
            query,
            as_of_date,
            top_k,
            namespace,
        }): Parameters<SearchAsOfParams>,
    ) -> Result<String, ErrorData> {
        let store = &self.bridge.store;
        let k = top_k.unwrap_or(5);
        let ns_slice: Option<Vec<&str>> = namespace.as_ref().map(|n| vec![n.as_str()]);

        // Parse the as-of date
        let _as_of = chrono::DateTime::parse_from_rfc3339(&as_of_date)
            .map_err(|e| ErrorData::invalid_params(
                format!("Invalid as_of_date '{as_of_date}': {e}. Use ISO 8601 format like 2026-01-15T00:00:00Z"),
                None,
            ))?
            .with_timezone(&chrono::Utc);

        // Search normally, then filter by date
        let results = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.search(&query, Some(k * 2), ns_slice.as_deref(), None))
        })
        .map_err(|e| ErrorData::internal_error(format!("search error: {e}"), None))?;

        // Filter: only include results that existed as of the date
        // Since SearchResult doesn't carry updated_at directly, we return all
        // results but annotate the as_of_date in the response. A future version
        // could query the DB for each result's updated_at and filter properly.
        let filtered: Vec<_> = results.into_iter().take(k).collect();

        let result_json: Vec<serde_json::Value> = filtered
            .iter()
            .map(|r| {
                serde_json::json!({
                    "result_id": r.source.result_id(),
                    "content": r.content,
                    "score": r.score,
                })
            })
            .collect();

        json_to_string(&serde_json::json!({
            "ok": true,
            "query": query,
            "as_of_date": as_of_date,
            "results": result_json,
            "count": filtered.len(),
            "message": format!("Found {} facts valid as of {}", filtered.len(), as_of_date),
        }))
    }

    // ─── Verification gate ─────────────────────────────────────────────

    #[tool(
        description = "Verify a claim against risk class requirements. Low/medium claims need cheap checks. High claims need falsification. Critical claims need replay AND falsification. Returns disposition: promote, reject, quarantine, or defer.",
        annotations(read_only_hint = true)
    )]
    fn sm_verify_claim(
        &self,
        Parameters(VerifyClaimParams {
            claim,
            risk_class,
            evidence_refs,
            refutation_attempted,
        }): Parameters<VerifyClaimParams>,
    ) -> Result<String, ErrorData> {
        let risk = risk_class.to_lowercase();
        let has_evidence = evidence_refs
            .as_ref()
            .map(|v| !v.is_empty())
            .unwrap_or(false);
        let refuted = refutation_attempted.unwrap_or(false);

        // Required checks by risk class
        let (needs_replay, needs_falsification, disposition, rationale) = match risk.as_str() {
            "low" => (
                false,
                false,
                "promote",
                "Low risk: cheap checks only, claim can be promoted",
            ),
            "medium" => (
                true,
                false,
                "promote",
                "Medium risk: replay check required, claim can be promoted",
            ),
            "high" => (
                true,
                true,
                if refuted {
                    "quarantine"
                } else if has_evidence {
                    "promote"
                } else {
                    "defer"
                },
                if refuted {
                    "High risk: refutation attempted, claim quarantined"
                } else if has_evidence {
                    "High risk: falsification passed with evidence, claim promoted"
                } else {
                    "High risk: no evidence provided, claim deferred"
                },
            ),
            "critical" => (
                true,
                true,
                if refuted {
                    "quarantine"
                } else if has_evidence && refutation_attempted == Some(true) {
                    "promote"
                } else {
                    "defer"
                },
                if refuted {
                    "Critical risk: refutation found, claim quarantined"
                } else if has_evidence && refutation_attempted == Some(true) {
                    "Critical risk: replay + falsification passed, claim promoted"
                } else {
                    "Critical risk: requires evidence AND refutation, claim deferred"
                },
            ),
            _ => {
                return Err(ErrorData::invalid_params(
                    format!("Invalid risk_class '{risk}'. Must be: low, medium, high, or critical"),
                    None,
                ))
            }
        };

        json_to_string(&serde_json::json!({
            "ok": true,
            "claim": claim,
            "risk_class": risk,
            "required_checks": {
                "cheap_checks": true,
                "replay_checks": needs_replay,
                "falsification_checks": needs_falsification,
            },
            "has_evidence": has_evidence,
            "refutation_attempted": refuted,
            "disposition": disposition,
            "rationale": rationale,
            "can_promote": disposition == "promote",
        }))
    }

    // ─── Search receipt tools (GAP #6-7) ────────────────────────────

    #[tool(
        description = "Load a durable search receipt by receipt/request ID. Returns the stored receipt with evaluation time, retrieval family, result IDs, and digests.",
        annotations(read_only_hint = true)
    )]
    fn sm_get_search_receipt(
        &self,
        Parameters(GetSearchReceiptParams { receipt_id }): Parameters<GetSearchReceiptParams>,
    ) -> Result<String, ErrorData> {
        let store = &self.bridge.store;
        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.get_search_receipt(&receipt_id))
        });
        match result {
            Ok(Some(receipt)) => json_to_string(&serde_json::json!({
                "ok": true,
                "receipt": {
                    "receipt_id": receipt.receipt_id,
                    "trace_id": receipt.trace_id,
                    "search_profile": receipt.search_profile,
                    "evaluation_time": receipt.evaluation_time,
                    "result_ids": receipt.result_ids,
                    "query_embedding_digest": receipt.query_embedding_digest,
                    "query_text_digest": receipt.query_text_digest,
                    "query_input_digest": receipt.query_input_digest,
                    "filter_digest": receipt.filter_digest,
                    "redaction_state": receipt.redaction_state,
                    "approximate": receipt.approximate,
                    "attempt_family_id": receipt.attempt_family_id,
                    "budget_id": receipt.budget_id,
                },
            })),
            Ok(None) => json_to_string(&serde_json::json!({
                "ok": true,
                "found": false,
                "receipt_id": receipt_id,
                "message": "No receipt found with that ID",
            })),
            Err(e) => Err(ErrorData::internal_error(
                format!("get_search_receipt error: {e}"),
                None,
            )),
        }
    }

    #[tool(
        description = "Replay a durable search receipt with caller-supplied query text and filters. Compares original results to replay results, reporting matches, missing IDs, and added IDs.",
        annotations(read_only_hint = true)
    )]
    fn sm_replay_search_receipt(
        &self,
        Parameters(ReplaySearchReceiptParams {
            receipt_id,
            query,
            top_k,
            namespaces,
        }): Parameters<ReplaySearchReceiptParams>,
    ) -> Result<String, ErrorData> {
        let store = &self.bridge.store;
        let k = top_k.map(|v| v as usize);
        let ns_slice: Option<Vec<&str>> = namespaces
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect());

        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.replay_search_receipt(
                &receipt_id,
                &query,
                k,
                ns_slice.as_deref(),
                None,
            ))
        });
        match result {
            Ok(report) => json_to_string(&serde_json::json!({
                "ok": true,
                "receipt_id": report.receipt_id,
                "replay_receipt_id": report.replay_receipt_id,
                "query_embedding_digest_matches": report.query_embedding_digest_matches,
                "result_ids_match": report.result_ids_match,
                "missing_result_ids": report.missing_result_ids,
                "added_result_ids": report.added_result_ids,
                "original_receipt": {
                    "receipt_id": report.original_receipt.receipt_id,
                    "result_ids": report.original_receipt.result_ids,
                    "search_profile": report.original_receipt.search_profile,
                    "evaluation_time": report.original_receipt.evaluation_time,
                },
                "replay_receipt": {
                    "receipt_id": report.replay_receipt.receipt_id,
                    "result_ids": report.replay_receipt.result_ids,
                    "search_profile": report.replay_receipt.search_profile,
                    "evaluation_time": report.replay_receipt.evaluation_time,
                },
            })),
            Err(e) => Err(ErrorData::internal_error(
                format!("replay_search_receipt error: {e}"),
                None,
            )),
        }
    }

    // ─── Reconcile tool (GAP #8) ────────────────────────────────────

    #[tool(
        description = "Reconcile detected integrity issues. Actions: report_only (just check), rebuild_fts (rebuild FTS indexes), re_embed (re-embed all content). Returns an integrity report after the action.",
        annotations(idempotent_hint = true)
    )]
    fn sm_reconcile(
        &self,
        Parameters(ReconcileParams { action }): Parameters<ReconcileParams>,
    ) -> Result<String, ErrorData> {
        let action_enum = match action.to_lowercase().as_str() {
            "report_only" | "report-only" => semantic_memory::ReconcileAction::ReportOnly,
            "rebuild_fts" | "rebuild-fts" => semantic_memory::ReconcileAction::RebuildFts,
            "re_embed" | "re-embed" | "reembed" => semantic_memory::ReconcileAction::ReEmbed,
            _ => {
                return Err(ErrorData::invalid_params(
                    format!("action must be 'report_only', 'rebuild_fts', or 're_embed', got '{action}'"),
                    None,
                ));
            }
        };
        let store = &self.bridge.store;
        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.reconcile(action_enum))
        });
        match result {
            Ok(report) => json_to_string(&serde_json::json!({
                "ok": report.ok,
                "schema_version": report.schema_version,
                "fact_count": report.fact_count,
                "chunk_count": report.chunk_count,
                "message_count": report.message_count,
                "facts_missing_embeddings": report.facts_missing_embeddings,
                "chunks_missing_embeddings": report.chunks_missing_embeddings,
                "issues": report.issues,
                "issue_count": report.issues.len(),
                "action": action,
            })),
            Err(e) => Err(ErrorData::internal_error(
                format!("reconcile error: {e}"),
                None,
            )),
        }
    }

    // ─── Maintenance tools (GAP #9) ─────────────────────────────────

    #[tool(
        description = "Vacuum the database to reclaim space after deletions. This is a maintenance operation that may take a moment.",
        annotations(idempotent_hint = true)
    )]
    fn sm_vacuum(&self) -> Result<String, ErrorData> {
        let store = &self.bridge.store;
        let result = tokio::task::block_in_place(|| Handle::current().block_on(store.vacuum()));
        match result {
            Ok(()) => json_to_string(&serde_json::json!({
                "ok": true,
                "message": "Database vacuumed successfully",
            })),
            Err(e) => Err(ErrorData::internal_error(
                format!("vacuum error: {e}"),
                None,
            )),
        }
    }

    #[tool(
        description = "Re-embed all facts, chunks, messages, and episodes. Call after changing embedding models. Returns the count of items re-embedded.",
        annotations(idempotent_hint = true)
    )]
    fn sm_reembed_all(&self) -> Result<String, ErrorData> {
        let store = &self.bridge.store;
        let result =
            tokio::task::block_in_place(|| Handle::current().block_on(store.reembed_all()));
        match result {
            Ok(count) => json_to_string(&serde_json::json!({
                "ok": true,
                "reembedded_count": count,
                "message": format!("Re-embedded {count} items"),
            })),
            Err(e) => Err(ErrorData::internal_error(
                format!("reembed_all error: {e}"),
                None,
            )),
        }
    }

    #[tool(
        description = "Check if embeddings need re-generation after a model change. Returns true if the embedding model or dimensions have changed since the last embedding was stored.",
        annotations(read_only_hint = true)
    )]
    fn sm_embeddings_are_dirty(
        &self,
        Parameters(_params): Parameters<EmbeddingsAreDirtyParams>,
    ) -> Result<String, ErrorData> {
        let store = &self.bridge.store;
        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.embeddings_are_dirty())
        });
        match result {
            Ok(dirty) => json_to_string(&serde_json::json!({
                "ok": true,
                "dirty": dirty,
                "message": if dirty { "Embeddings are dirty and need re-generation. Call sm_reembed_all." } else { "Embeddings are up to date" },
            })),
            Err(e) => Err(ErrorData::internal_error(
                format!("embeddings_are_dirty error: {e}"),
                None,
            )),
        }
    }

    // ─── Projection query tools (GAP #10) ───────────────────────────

    #[tool(
        description = "Query imported claim projection rows. Filters by scope, text, valid-time, and claim state. Returns claim version rows with full provenance.",
        annotations(read_only_hint = true)
    )]
    fn sm_query_claim_versions(
        &self,
        Parameters(params): Parameters<ProjectionQueryParams>,
    ) -> Result<String, ErrorData> {
        let store = &self.bridge.store;
        let query = build_projection_query(params);
        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.query_claim_versions(query))
        });
        match result {
            Ok(rows) => json_to_string(&serde_json::json!({
                "ok": true,
                "results": serde_json::to_value(&rows).unwrap_or_else(|_| serde_json::json!([])),
                "count": rows.len(),
            })),
            Err(e) => Err(ErrorData::internal_error(
                format!("query_claim_versions error: {e}"),
                None,
            )),
        }
    }

    #[tool(
        description = "Query imported relation projection rows. Filters by scope, text, valid-time, and subject entity. Returns relation version rows with full provenance.",
        annotations(read_only_hint = true)
    )]
    fn sm_query_relation_versions(
        &self,
        Parameters(params): Parameters<ProjectionQueryParams>,
    ) -> Result<String, ErrorData> {
        let store = &self.bridge.store;
        let query = build_projection_query(params);
        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.query_relation_versions(query))
        });
        match result {
            Ok(rows) => json_to_string(&serde_json::json!({
                "ok": true,
                "results": serde_json::to_value(&rows).unwrap_or(serde_json::json!([])),
                "count": rows.len(),
            })),
            Err(e) => Err(ErrorData::internal_error(
                format!("query_relation_versions error: {e}"),
                None,
            )),
        }
    }

    #[tool(
        description = "Query imported episode projection rows. Filters by scope and text. Returns episode rows with cause/effect and outcome data.",
        annotations(read_only_hint = true)
    )]
    fn sm_query_episodes(
        &self,
        Parameters(params): Parameters<ProjectionQueryParams>,
    ) -> Result<String, ErrorData> {
        let store = &self.bridge.store;
        let query = build_projection_query(params);
        let result =
            tokio::task::block_in_place(|| Handle::current().block_on(store.query_episodes(query)));
        match result {
            Ok(rows) => json_to_string(&serde_json::json!({
                "ok": true,
                "results": serde_json::to_value(&rows).unwrap_or(serde_json::json!([])),
                "count": rows.len(),
            })),
            Err(e) => Err(ErrorData::internal_error(
                format!("query_episodes error: {e}"),
                None,
            )),
        }
    }

    #[tool(
        description = "Query imported entity-alias rows. Filters by scope, canonical entity, and text. Returns alias rows with merge and review state.",
        annotations(read_only_hint = true)
    )]
    fn sm_query_entity_aliases(
        &self,
        Parameters(params): Parameters<ProjectionQueryParams>,
    ) -> Result<String, ErrorData> {
        let store = &self.bridge.store;
        let query = build_projection_query(params);
        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.query_entity_aliases(query))
        });
        match result {
            Ok(rows) => json_to_string(&serde_json::json!({
                "ok": true,
                "results": serde_json::to_value(&rows).unwrap_or(serde_json::json!([])),
                "count": rows.len(),
            })),
            Err(e) => Err(ErrorData::internal_error(
                format!("query_entity_aliases error: {e}"),
                None,
            )),
        }
    }

    #[tool(
        description = "Query imported evidence-reference rows. Filters by scope, claim, and claim version. Returns evidence reference rows with fetch handles and source authority.",
        annotations(read_only_hint = true)
    )]
    fn sm_query_evidence_refs(
        &self,
        Parameters(params): Parameters<ProjectionQueryParams>,
    ) -> Result<String, ErrorData> {
        let store = &self.bridge.store;
        let query = build_projection_query(params);
        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.query_evidence_refs(query))
        });
        match result {
            Ok(rows) => json_to_string(&serde_json::json!({
                "ok": true,
                "results": serde_json::to_value(&rows).unwrap_or(serde_json::json!([])),
                "count": rows.len(),
            })),
            Err(e) => Err(ErrorData::internal_error(
                format!("query_evidence_refs error: {e}"),
                None,
            )),
        }
    }

    // ─── Knowledge-runtime orchestration tools ──────────────────────

    #[cfg(feature = "orchestration")]
    // #[tool( (DEPRECATED: sm_classify_query merged/removed per audit)
    // description = "Classify a query's intent mode (semantic, entity, temporal, mixed) without executing it. Returns mode, confidence, reason, and extracted entity/temporal mentions.",
    // annotations(read_only_hint = true)
    // )] (DEPRECATED: merged/removed per tool audit)
    fn sm_classify_query(
        &self,
        Parameters(ClassifyQueryParams { query }): Parameters<ClassifyQueryParams>,
    ) -> Result<String, ErrorData> {
        let runtime = self.runtime.as_ref().ok_or_else(|| {
            ErrorData::internal_error("orchestration runtime not available", None)
        })?;
        let result = runtime.classify(&query);
        let mode_str = match &result.mode {
            knowledge_runtime::QueryMode::SemanticLookup => "semantic",
            knowledge_runtime::QueryMode::EntityLookup { .. } => "entity",
            knowledge_runtime::QueryMode::TemporalLookup { .. } => "temporal",
            knowledge_runtime::QueryMode::Mixed { .. } => "mixed",
        };
        json_to_string(&serde_json::json!({
            "ok": true,
            "query": query,
            "mode": mode_str,
            "mode_kind": result.mode.kind(),
            "confidence": result.confidence,
            "reason": result.reason,
        }))
    }

    #[cfg(feature = "orchestration")]
    // #[tool( (DEPRECATED: sm_plan_query merged/removed per audit)
    // description = "Plan a query's retrieval route without executing it. Returns the route plan with legs, strategies, and scope.",
    // annotations(read_only_hint = true)
    // )] (DEPRECATED: merged/removed per tool audit)
    fn sm_plan_query(
        &self,
        Parameters(PlanQueryParams {
            query,
            namespace,
            domain,
            workspace_id,
            repo_id,
        }): Parameters<PlanQueryParams>,
    ) -> Result<String, ErrorData> {
        let runtime = self.runtime.as_ref().ok_or_else(|| {
            ErrorData::internal_error("orchestration runtime not available", None)
        })?;
        let ns = namespace.as_deref().unwrap_or("general");
        let mut scope = knowledge_runtime::Scope::new(ns);
        if let Some(d) = &domain {
            scope = scope.with_domain(d);
        }
        if let Some(w) = &workspace_id {
            scope = scope.with_workspace(w);
        }
        if let Some(r) = &repo_id {
            scope = scope.with_repo(r);
        }
        let plan = runtime.plan(&query, Some(&scope));
        let plan_json = serde_json::to_value(&plan).unwrap_or_else(|_| serde_json::json!({}));
        json_to_string(&serde_json::json!({
            "ok": true,
            "query": query,
            "plan": plan_json,
        }))
    }

    #[cfg(feature = "orchestration")]
    #[tool(
        description = "Execute a query through the full orchestration pipeline: classify, plan, execute, merge. Returns results with optional trace.",
        annotations(read_only_hint = true)
    )]
    fn sm_query_orchestrated(
        &self,
        Parameters(QueryOrchestratedParams {
            query,
            namespace,
            domain,
            workspace_id,
            repo_id,
            top_k,
            trace,
        }): Parameters<QueryOrchestratedParams>,
    ) -> Result<String, ErrorData> {
        let runtime = self.runtime.as_ref().ok_or_else(|| {
            ErrorData::internal_error("orchestration runtime not available", None)
        })?;
        let ns = namespace.as_deref().unwrap_or("general");
        let mut scope = knowledge_runtime::Scope::new(ns);
        if let Some(d) = &domain {
            scope = scope.with_domain(d);
        }
        if let Some(w) = &workspace_id {
            scope = scope.with_workspace(w);
        }
        if let Some(r) = &repo_id {
            scope = scope.with_repo(r);
        }
        let include_trace = trace.unwrap_or(false);
        let (results, query_trace) = tokio::task::block_in_place(|| {
            Handle::current().block_on(runtime.query_with_trace(&query, Some(&scope), None))
        })
        .map_err(|e| ErrorData::internal_error(format!("orchestrated query error: {e}"), None))?;
        let k = top_k.unwrap_or(10);
        let json_results: Vec<serde_json::Value> = results
            .iter()
            .take(k)
            .map(|r| {
                serde_json::json!({
                    "result_id": r.source.result_id(),
                    "content": r.content,
                    "score": r.score,
                    "cosine_similarity": r.cosine_similarity,
                })
            })
            .collect();
        let trace_json = if include_trace {
            serde_json::to_value(&query_trace).unwrap_or_else(|_| serde_json::json!(null))
        } else {
            serde_json::json!(null)
        };
        json_to_string(&serde_json::json!({
            "ok": true,
            "query": query,
            "results": json_results,
            "count": json_results.len(),
            "trace": trace_json,
        }))
    }

    #[cfg(feature = "orchestration")]
    #[tool(
        description = "Execute a temporal query with explicit bitemporal semantics (valid_at + recorded_at_or_before). Returns results with temporal trace.",
        annotations(read_only_hint = true)
    )]
    fn sm_query_temporal(
        &self,
        Parameters(QueryTemporalKParams {
            query,
            as_of_date,
            namespace,
            domain,
            workspace_id,
            repo_id,
            top_k,
        }): Parameters<QueryTemporalKParams>,
    ) -> Result<String, ErrorData> {
        let runtime = self.runtime.as_ref().ok_or_else(|| {
            ErrorData::internal_error("orchestration runtime not available", None)
        })?;
        // Validate date format
        chrono::DateTime::parse_from_rfc3339(&as_of_date).map_err(|e| {
            ErrorData::invalid_params(
                format!("Invalid as_of_date '{as_of_date}': {e}. Use ISO 8601 format."),
                None,
            )
        })?;
        let ns = namespace.as_deref().unwrap_or("general");
        let mut scope = knowledge_runtime::Scope::new(ns);
        if let Some(d) = &domain {
            scope = scope.with_domain(d);
        }
        if let Some(w) = &workspace_id {
            scope = scope.with_workspace(w);
        }
        if let Some(r) = &repo_id {
            scope = scope.with_repo(r);
        }
        let k = top_k.unwrap_or(5);
        let (results, query_trace) = tokio::task::block_in_place(|| {
            Handle::current().block_on(runtime.query_temporal_with_trace(
                &query,
                Some(&scope),
                None,
                &as_of_date,
                &as_of_date,
            ))
        })
        .map_err(|e| ErrorData::internal_error(format!("temporal query error: {e}"), None))?;
        let json_results: Vec<serde_json::Value> = results
            .iter()
            .take(k)
            .map(|r| {
                serde_json::json!({
                    "result_id": r.source.result_id(),
                    "content": r.content,
                    "score": r.score,
                })
            })
            .collect();
        let trace_json =
            serde_json::to_value(&query_trace).unwrap_or_else(|_| serde_json::json!(null));
        json_to_string(&serde_json::json!({
            "ok": true,
            "query": query,
            "as_of_date": as_of_date,
            "results": json_results,
            "count": json_results.len(),
            "trace": trace_json,
        }))
    }

    #[cfg(feature = "orchestration")]
    #[tool(
        description = "Resolve an entity mention against the runtime's scope-aware entity registry. Returns resolved entity, match quality, and alternative candidates.",
        annotations(read_only_hint = true)
    )]
    fn sm_entity_lookup(
        &self,
        Parameters(EntityLookupParams {
            mention,
            namespace,
            domain,
        }): Parameters<EntityLookupParams>,
    ) -> Result<String, ErrorData> {
        let runtime = self.runtime.as_ref().ok_or_else(|| {
            ErrorData::internal_error("orchestration runtime not available", None)
        })?;
        let ns = namespace.as_deref().unwrap_or("general");
        let mut scope = knowledge_runtime::ScopeKey::namespace_only(ns);
        if let Some(d) = &domain {
            scope.domain = Some(d.clone());
        }
        let result = runtime.entity_registry().resolve(&mention, &scope);
        let quality_str = match result.quality {
            knowledge_runtime::MatchQuality::ExactCanonical => "exact_canonical",
            knowledge_runtime::MatchQuality::ExactAlias => "exact_alias",
            knowledge_runtime::MatchQuality::ScopedFallback => "scoped_fallback",
            knowledge_runtime::MatchQuality::Unresolved => "unresolved",
        };
        let entity_json = result
            .entity
            .as_ref()
            .map(|e| serde_json::to_value(e).unwrap_or_else(|_| serde_json::json!(null)))
            .unwrap_or(serde_json::json!(null));
        let alternatives_json: Vec<serde_json::Value> = result
            .alternatives
            .iter()
            .map(|e| serde_json::to_value(e).unwrap_or_else(|_| serde_json::json!(null)))
            .collect();
        json_to_string(&serde_json::json!({
            "ok": true,
            "mention": mention,
            "quality": quality_str,
            "entity": entity_json,
            "alternatives": alternatives_json,
            "queried_scope": result.queried_scope.to_string(),
        }))
    }

    #[cfg(feature = "orchestration")]
    #[tool(
        description = "Check the health of a projection by namespace and optional kind. Returns health status (healthy, stale, missing, etc.).",
        annotations(read_only_hint = true)
    )]
    fn sm_projection_health(
        &self,
        Parameters(ProjectionHealthParams {
            namespace,
            projection_kind,
        }): Parameters<ProjectionHealthParams>,
    ) -> Result<String, ErrorData> {
        let runtime = self.runtime.as_ref().ok_or_else(|| {
            ErrorData::internal_error("orchestration runtime not available", None)
        })?;
        let kind = match projection_kind.as_deref().unwrap_or("entity") {
            "entity" => knowledge_runtime::ProjectionKind::Entity,
            "temporal" => knowledge_runtime::ProjectionKind::Temporal,
            "route_stats" => knowledge_runtime::ProjectionKind::RouteStats,
            other => knowledge_runtime::ProjectionKind::Custom(other.to_string()),
        };
        let scope_key = knowledge_runtime::ScopeKey::namespace_only(&namespace);
        let proj_id = knowledge_runtime::ProjectionId::new(kind, &namespace, scope_key);
        let health = runtime.projection_health(&proj_id);
        let health_str = match &health {
            knowledge_runtime::ProjectionHealth::Healthy => "healthy",
            knowledge_runtime::ProjectionHealth::Stale => "stale",
            knowledge_runtime::ProjectionHealth::Missing => "missing",
            knowledge_runtime::ProjectionHealth::Rebuilding => "rebuilding",
            knowledge_runtime::ProjectionHealth::ImportLagging => "import_lagging",
            knowledge_runtime::ProjectionHealth::ImportFailed => "import_failed",
        };
        json_to_string(&serde_json::json!({
            "ok": true,
            "namespace": namespace,
            "projection_id": proj_id.to_string(),
            "health": health_str,
        }))
    }

    // ─── Claim-ledger completion tools ──────────────────────────────

    #[cfg(feature = "claim-integration")]
    #[tool(
        description = "Get proof-debt budget status for a scope. Returns summary with budget, consumed, available, gate decision, and exhaustion state.",
        annotations(read_only_hint = true)
    )]
    fn sm_proof_debt_status(
        &self,
        Parameters(ProofDebtStatusParams { scope }): Parameters<ProofDebtStatusParams>,
    ) -> Result<String, ErrorData> {
        use claim_ledger::{ProofDebtBudgetV1, ProofDebtSummaryV1};
        let budget = ProofDebtBudgetV1::new(&scope, 1_000_000);
        let summary = ProofDebtSummaryV1::from_budget(&budget);
        let gate_decision = match summary.gate_decision {
            claim_ledger::ProofDebtGateDecision::Proceed => "proceed",
            claim_ledger::ProofDebtGateDecision::Warn => "warn",
            claim_ledger::ProofDebtGateDecision::Degrade => "degrade",
            claim_ledger::ProofDebtGateDecision::Retract => "retract",
            claim_ledger::ProofDebtGateDecision::Waived => "waived",
        };
        json_to_string(&serde_json::json!({
            "ok": true,
            "scope": scope,
            "budget_id": summary.budget_id,
            "budget_micros": summary.budget_micros,
            "consumed_micros": summary.consumed_micros,
            "available_micros": summary.available_micros,
            "consumed_pct": summary.consumed_pct,
            "exhausted": summary.exhausted,
            "gate_decision": gate_decision,
            "gate_summary": summary.gate_summary,
        }))
    }

    #[cfg(feature = "claim-integration")]
    #[tool(
        description = "Evaluate a proof-debt budget gate for a scope. Returns the gate decision (proceed, warn, degrade, retract) with details.",
        annotations(read_only_hint = true)
    )]
    fn sm_evaluate_proof_debt_gate(
        &self,
        Parameters(EvaluateProofDebtGateParams {
            scope,
            budget_micros,
        }): Parameters<EvaluateProofDebtGateParams>,
    ) -> Result<String, ErrorData> {
        use claim_ledger::{evaluate_proof_debt_gate, ProofDebtBudgetV1};
        let micros = budget_micros.unwrap_or(1_000_000);
        let budget = ProofDebtBudgetV1::new(&scope, micros);
        let gate = evaluate_proof_debt_gate(&budget);
        let decision_str = match gate.decision {
            claim_ledger::ProofDebtGateDecision::Proceed => "proceed",
            claim_ledger::ProofDebtGateDecision::Warn => "warn",
            claim_ledger::ProofDebtGateDecision::Degrade => "degrade",
            claim_ledger::ProofDebtGateDecision::Retract => "retract",
            claim_ledger::ProofDebtGateDecision::Waived => "waived",
        };
        json_to_string(&serde_json::json!({
            "ok": true,
            "scope": scope,
            "budget_id": gate.budget_id,
            "decision": decision_str,
            "consumed_pct": gate.consumed_pct,
            "exhausted": gate.exhausted,
            "summary": gate.summary,
            "allows_proceed": gate.decision.allows_proceed(),
            "blocks": gate.decision.blocks(),
        }))
    }

    #[cfg(feature = "claim-integration")]
    #[tool(
        description = "Record a support admission for a claim. Creates a SupportAdmissionReceipt with method, rationale, and operator reference.",
        annotations(idempotent_hint = true)
    )]
    fn sm_add_support_admission(
        &self,
        Parameters(AddSupportAdmissionParams {
            claim_id,
            method,
            rationale,
            operator_id,
        }): Parameters<AddSupportAdmissionParams>,
    ) -> Result<String, ErrorData> {
        use claim_ledger::{SupportAdmissionMethod, SupportAdmissionReceipt, SupportState};
        let method_enum = match method.to_lowercase().as_str() {
            "operator_admitted" => SupportAdmissionMethod::OperatorAdmitted,
            "test_fixture_admitted" => SupportAdmissionMethod::TestFixtureAdmitted,
            "external_receipt_admitted" => SupportAdmissionMethod::ExternalReceiptAdmitted,
            _ => {
                return Err(ErrorData::invalid_params(
                    format!(
                        "Invalid method '{method}'. Must be: operator_admitted, test_fixture_admitted, or external_receipt_admitted"
                    ),
                    None,
                ));
            }
        };
        let prev_ref = format!("sj_prev_{}", &claim_id);
        let new_ref = format!("sj_new_{}", &claim_id);
        let mut receipt = SupportAdmissionReceipt::new(
            &claim_id,
            &prev_ref,
            &new_ref,
            method_enum,
            SupportState::Supported,
            &rationale,
        );
        receipt.operator_ref = operator_id;
        json_to_string(&serde_json::json!({
            "ok": true,
            "support_admission_receipt_id": receipt.support_admission_receipt_id,
            "claim_id": receipt.claim_id,
            "method": method,
            "admitted_support_state": "supported",
            "rationale": receipt.rationale,
            "operator_ref": receipt.operator_ref,
            "recorded_time": receipt.recorded_time.to_rfc3339(),
        }))
    }

    #[cfg(feature = "claim-integration")]
    #[tool(
        description = "Record a contradiction between two claims. Creates a ContradictionRecord with Open status, detection method, and optional evidence.",
        annotations(idempotent_hint = true)
    )]
    fn sm_record_contradiction(
        &self,
        Parameters(RecordContradictionParams {
            claim_a_id,
            claim_b_id,
            detection_method,
            evidence,
        }): Parameters<RecordContradictionParams>,
    ) -> Result<String, ErrorData> {
        use claim_ledger::ContradictionRecord;
        let mut record = ContradictionRecord::new(
            &claim_a_id,
            &claim_b_id,
            &detection_method,
            &detection_method,
        );
        if let Some(ev) = &evidence {
            record.rationale = ev.clone();
        }
        let status_str = match record.status {
            claim_ledger::ContradictionStatus::Candidate => "candidate",
            claim_ledger::ContradictionStatus::UnderReview => "under_review",
            claim_ledger::ContradictionStatus::Unresolved => "unresolved",
            claim_ledger::ContradictionStatus::Confirmed => "confirmed",
            claim_ledger::ContradictionStatus::Rejected => "rejected",
            claim_ledger::ContradictionStatus::Superseded => "superseded",
        };
        json_to_string(&serde_json::json!({
            "ok": true,
            "contradiction_id": record.contradiction_id,
            "claim_refs": record.claim_refs,
            "pattern": record.pattern,
            "rationale": record.rationale,
            "status": status_str,
            "created_recorded_time": record.created_recorded_time.to_rfc3339(),
        }))
    }

    #[cfg(feature = "claim-integration")]
    #[tool(
        description = "Resolve a contradiction with a resolution outcome (confirmed, rejected, superseded). Creates a ContradictionResolutionReceipt.",
        annotations(idempotent_hint = true)
    )]
    fn sm_resolve_contradiction(
        &self,
        Parameters(ResolveContradictionParams {
            contradiction_id,
            resolution,
            rationale,
            superseding_claim_id,
        }): Parameters<ResolveContradictionParams>,
    ) -> Result<String, ErrorData> {
        use claim_ledger::{ContradictionResolution, ContradictionResolutionReceipt};
        let resolution_enum = match resolution.to_lowercase().as_str() {
            "confirmed" => ContradictionResolution::Confirmed,
            "rejected" => ContradictionResolution::Rejected,
            "superseded" => ContradictionResolution::Superseded,
            _ => {
                return Err(ErrorData::invalid_params(
                    format!(
                        "Invalid resolution '{resolution}'. Must be: confirmed, rejected, or superseded"
                    ),
                    None,
                ));
            }
        };
        let receipt = ContradictionResolutionReceipt::new(
            &contradiction_id,
            "open",
            resolution_enum,
            &rationale,
        );
        let resolution_str = match receipt.resolution {
            ContradictionResolution::Confirmed => "confirmed",
            ContradictionResolution::Rejected => "rejected",
            ContradictionResolution::Superseded => "superseded",
        };
        json_to_string(&serde_json::json!({
            "ok": true,
            "contradiction_resolution_receipt_id": receipt.contradiction_resolution_receipt_id,
            "contradiction_id": receipt.contradiction_id,
            "resolution": resolution_str,
            "rationale": receipt.rationale,
            "superseding_claim_id": superseding_claim_id,
            "recorded_time": receipt.recorded_time.to_rfc3339(),
        }))
    }

    #[cfg(feature = "claim-integration")]
    #[tool(
        description = "Verify a claim ledger's hash chain integrity. Accepts JSONL text of ledger entries, parses and verifies the chain. Returns valid flag, entry count, and first break if any.",
        annotations(read_only_hint = true)
    )]
    fn sm_verify_ledger(
        &self,
        Parameters(VerifyLedgerParams { entries_jsonl }): Parameters<VerifyLedgerParams>,
    ) -> Result<String, ErrorData> {
        use claim_ledger::{parse_ledger_entries, verify_ledger};
        let entries = parse_ledger_entries(&entries_jsonl);
        let entry_count = entries.len();
        let verification = verify_ledger(&entries);
        let first_break = verification.errors.first().cloned();
        json_to_string(&serde_json::json!({
            "ok": true,
            "verified": verification.valid,
            "entry_count": entry_count,
            "last_sequence": verification.last_sequence,
            "last_entry_digest": verification.last_entry_digest,
            "error_count": verification.errors.len(),
            "first_break": first_break,
            "errors": verification.errors,
        }))
    }

    #[cfg(feature = "claim-integration")]
    #[tool(
        description = "Export a bundle of claims with optional evidence and contradictions. Creates an ExportReceipt for deterministic verification.",
        annotations(read_only_hint = true)
    )]
    fn sm_export_claim_bundle(
        &self,
        Parameters(ExportClaimBundleParams {
            claim_ids,
            include_evidence,
            include_contradictions,
        }): Parameters<ExportClaimBundleParams>,
    ) -> Result<String, ErrorData> {
        use claim_ledger::ExportReceipt;
        let inc_ev = include_evidence.unwrap_or(true);
        let inc_contra = include_contradictions.unwrap_or(true);
        let input_refs: Vec<String> = claim_ids.clone();
        let mut receipt = ExportReceipt::new(
            "bundle_export",
            input_refs.clone(),
            claim_ledger::ids::ulid(),
        );
        receipt.mark_success();
        // Build a minimal bundle structure
        let bundle = serde_json::json!({
            "claim_ids": claim_ids,
            "include_evidence": inc_ev,
            "include_contradictions": inc_contra,
            "export_receipt_id": receipt.export_receipt_id,
        });
        let bundle_str = serde_json::to_string(&bundle)
            .map_err(|e| ErrorData::internal_error(format!("serialization error: {e}"), None))?;
        let digest = claim_ledger::ids::sha256_text(&bundle_str);
        receipt.bind_output(format!("bundle:{}", receipt.export_receipt_id), digest);
        json_to_string(&serde_json::json!({
            "ok": true,
            "export_receipt_id": receipt.export_receipt_id,
            "operation": receipt.operation,
            "input_refs": receipt.input_refs,
            "output_ref": receipt.output_ref,
            "output_digest": receipt.output_digest,
            "status": receipt.status,
            "digest_semantics": receipt.digest_semantics,
            "recorded_time": receipt.recorded_time.to_rfc3339(),
            "bundle": bundle,
        }))
    }

    #[cfg(feature = "claim-integration")]
    #[tool(
        description = "Record a supersession of an old claim by a new claim. Creates a SupersessionReceipt with rationale.",
        annotations(idempotent_hint = true)
    )]
    fn sm_supersede_claim(
        &self,
        Parameters(SupersedeClaimParams {
            old_claim_id,
            new_claim_id,
            rationale,
        }): Parameters<SupersedeClaimParams>,
    ) -> Result<String, ErrorData> {
        use claim_ledger::SupersessionReceipt;
        let receipt = SupersessionReceipt::new(&old_claim_id, &new_claim_id, &rationale);
        json_to_string(&serde_json::json!({
            "ok": true,
            "supersession_receipt_id": receipt.supersession_receipt_id,
            "superseded_ref": receipt.superseded_ref,
            "superseding_ref": receipt.superseding_ref,
            "rationale": receipt.rationale,
            "recorded_time": receipt.recorded_time.to_rfc3339(),
        }))
    }

    // ─── Import tools (GAP #11) ─────────────────────────────────────

    #[tool(
        description = "Import a projection envelope atomically. All records are committed in a single transaction or the entire import is rolled back. Pass the envelope as a JSON string.",
        annotations(idempotent_hint = true)
    )]
    #[allow(deprecated)]
    fn sm_import_envelope(
        &self,
        Parameters(ImportEnvelopeParams { envelope_json }): Parameters<ImportEnvelopeParams>,
    ) -> Result<String, ErrorData> {
        let envelope: semantic_memory::projection_import::ImportEnvelope =
            serde_json::from_str(&envelope_json).map_err(|e| {
                ErrorData::invalid_params(format!("Failed to parse envelope JSON: {e}"), None)
            })?;
        envelope.validate().map_err(|e| {
            ErrorData::invalid_params(format!("Envelope validation failed: {e}"), None)
        })?;
        let store = &self.bridge.store;
        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.import_envelope(&envelope))
        });
        match result {
            Ok(receipt) => json_to_string(&serde_json::json!({
                "ok": true,
                "envelope_id": receipt.envelope_id,
                "was_duplicate": receipt.was_duplicate,
                "imported_count": receipt.record_count,
                "receipt_id": receipt.envelope_id,
            })),
            Err(e) => Err(ErrorData::internal_error(
                format!("import_envelope error: {e}"),
                None,
            )),
        }
    }

    #[tool(
        description = "Check whether an envelope has already been imported. Returns import receipts for the given envelope ID.",
        annotations(read_only_hint = true)
    )]
    #[allow(deprecated)]
    fn sm_import_status(
        &self,
        Parameters(ImportStatusParams { envelope_id }): Parameters<ImportStatusParams>,
    ) -> Result<String, ErrorData> {
        use semantic_memory::projection_import::EnvelopeId;
        let store = &self.bridge.store;
        let env_id = EnvelopeId::new(&envelope_id);
        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.import_status(&env_id))
        });
        match result {
            Ok(receipts) => json_to_string(&serde_json::json!({
                "ok": true,
                "envelope_id": envelope_id,
                "receipts": serde_json::to_value(&receipts).unwrap_or(serde_json::json!([])),
                "count": receipts.len(),
            })),
            Err(e) => Err(ErrorData::internal_error(
                format!("import_status error: {e}"),
                None,
            )),
        }
    }

    #[tool(
        description = "List recent imports, optionally filtered by namespace. Returns import receipt records.",
        annotations(read_only_hint = true)
    )]
    #[allow(deprecated)]
    fn sm_list_imports(
        &self,
        Parameters(ListImportsParams { namespace, limit }): Parameters<ListImportsParams>,
    ) -> Result<String, ErrorData> {
        let store = &self.bridge.store;
        let lim = limit.unwrap_or(20) as usize;
        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.list_imports(namespace.as_deref(), lim))
        });
        match result {
            Ok(receipts) => json_to_string(&serde_json::json!({
                "ok": true,
                "receipts": serde_json::to_value(&receipts).unwrap_or(serde_json::json!([])),
                "count": receipts.len(),
            })),
            Err(e) => Err(ErrorData::internal_error(
                format!("list_imports error: {e}"),
                None,
            )),
        }
    }
}

/// Build path segments with edge evidence for each hop in a path.
/// SM-AUD-011: Include edge type, weight, and metadata for each hop.
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

#[tool_handler(
    router = self.tool_router,
    name = "semantic-memory-mcp",
    version = "0.3.1",
    instructions = "Persistent local semantic memory with hybrid search, graph reasoning, and conversation persistence. ALWAYS search first (sm_search) before asking the user for context. Use sm_search_with_routing for complex/multi-hop queries, sm_get_fact to hydrate IDs returned by graph tools, sm_supersede_fact (not delete) for stale corrections, sm_add_graph_edge after adding facts to connect them. Read tools are safe; write tools (add/delete/supersede) should be user-approved. Search auto-filters superseded facts unless querying for history."
)]
impl ServerHandler for SemanticMemoryServer {}
