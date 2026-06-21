//! MCP server handler using rmcp's #[tool_router] macro.
//!
//! Each #[tool] method becomes an MCP tool that Hermes/Claude Desktop
//! can discover and call. The rmcp macro auto-generates JSON Schema
//! from the parameter structs in tools.rs.

use crate::bridge::MemoryBridge;
use crate::tools::*;
use rmcp::{handler::server::wrapper::Parameters, tool, tool_router, ErrorData};
use std::sync::Arc;
use tokio::runtime::Handle;

// Re-export the specific parameter types we use in tool signatures.
use crate::tools::{
    AddGraphEdgeParams, CommunityParams, FactorGraphParams, InvalidateGraphEdgeParams,
    ListGraphEdgesParams, TopologyParams,
};

pub struct SemanticMemoryServer {
    bridge: Arc<MemoryBridge>,
}

impl SemanticMemoryServer {
    pub fn new(bridge: MemoryBridge) -> Self {
        Self {
            bridge: Arc::new(bridge),
        }
    }
}

/// Helper: load all stored graph edges from the store as GraphEdgeRef tuples
/// for discord scoring.
fn load_stored_edge_refs(
    store: &semantic_memory::MemoryStore,
) -> Result<Vec<semantic_memory::discord::GraphEdgeRef>, ErrorData> {
    let edges = tokio::task::block_in_place(|| Handle::current().block_on(store.list_all_graph_edges()))
        .map_err(|e| ErrorData::internal_error(format!("Failed to load graph edges: {e}"), None))?;
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
    let edges = tokio::task::block_in_place(|| Handle::current().block_on(store.list_all_graph_edges()))
        .map_err(|e| ErrorData::internal_error(format!("Failed to load graph edges: {e}"), None))?;
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
    let edges = tokio::task::block_in_place(|| Handle::current().block_on(store.list_all_graph_edges()))
        .map_err(|e| ErrorData::internal_error(format!("Failed to load graph edges: {e}"), None))?;
    let pairs = edges
        .iter()
        .map(|edge| (edge.source.clone(), edge.target.clone()))
        .collect();
    Ok(pairs)
}

/// Serialize a JSON value to a pretty string, mapping serialization errors
/// to protocol-level errors instead of success strings.
fn json_to_string(value: &serde_json::Value) -> Result<String, ErrorData> {
    serde_json::to_string_pretty(value)
        .map_err(|e| ErrorData::internal_error(format!("Serialization error: {e}"), None))
}

#[tool_router(server_handler)]
impl SemanticMemoryServer {
    // ── Core search tools ────────────────────────────────────────────

    #[tool(description = "Semantic hybrid search over the knowledge base. Combines BM25 keyword matching with vector similarity and Reciprocal Rank Fusion. Returns ranked results with content, scores, and stable result IDs for downstream tool chaining.")]
    fn sm_search(
        &self,
        Parameters(SearchParams { query, top_k, namespaces }): Parameters<SearchParams>,
    ) -> Result<String, ErrorData> {
        let k = top_k.map(|v| v as usize);
        let ns: Option<Vec<&str>> = namespaces
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect());

        let store = &self.bridge.store;
        let result = tokio::task::block_in_place(|| Handle::current().block_on(store.search(&query, k, ns.as_deref(), None)));

        match result {
            Ok(results) => {
                let json_results: Vec<serde_json::Value> = results
                    .iter()
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
                }))
            }
            Err(e) => Err(ErrorData::internal_error(format!("Search error: {e}"), None)),
        }
    }

    #[tool(description = "Search with full score breakdown showing how BM25 and vector scores combine. Includes RRF contributions, rerank status, and configured weights. Useful for debugging retrieval quality.")]
    fn sm_search_explained(
        &self,
        Parameters(SearchExplainedParams { query, top_k }): Parameters<SearchExplainedParams>,
    ) -> Result<String, ErrorData> {
        let k = top_k.map(|v| v as usize);
        let store = &self.bridge.store;
        let result = tokio::task::block_in_place(|| Handle::current().block_on(store.search_explained(&query, k, None, None)));

        match result {
            Ok(results) => {
                let json_results: Vec<serde_json::Value> = results
                    .iter()
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
                    "count": results.len(),
                }))
            }
            Err(e) => Err(ErrorData::internal_error(format!("Search error: {e}"), None)),
        }
    }

    #[tool(description = "Add a fact to the knowledge base. The fact will be embedded and indexed for semantic search. Returns the fact ID and content digest.")]
    fn sm_add_fact(
        &self,
        Parameters(AddFactParams { content, namespace, source }): Parameters<AddFactParams>,
    ) -> Result<String, ErrorData> {
        let store = &self.bridge.store;
        let src = source.as_deref();
        let result = tokio::task::block_in_place(|| Handle::current().block_on(store.add_fact(&namespace, &content, src, None)));

        match result {
            Ok(id) => json_to_string(&serde_json::json!({
                "ok": true,
                "fact_id": id,
                "namespace": namespace,
                "message": "Fact added successfully",
            })),
            Err(e) => Err(ErrorData::internal_error(format!("Error adding fact: {e}"), None)),
        }
    }

    #[tool(description = "Ingest a document with automatic chunking. The document is split into chunks, each embedded and indexed. Returns document ID and chunk count.")]
    fn sm_ingest_document(
        &self,
        Parameters(IngestDocumentParams { content, title, namespace }): Parameters<IngestDocumentParams>,
    ) -> Result<String, ErrorData> {
        let store = &self.bridge.store;
        let result = tokio::task::block_in_place(|| Handle::current().block_on(store.ingest_document(&title, &content, &namespace, None, None)));

        match result {
            Ok(doc_id) => {
                let chunk_count = tokio::task::block_in_place(|| Handle::current().block_on(store.count_chunks_for_document(&doc_id)))
                    .unwrap_or(0);
                json_to_string(&serde_json::json!({
                    "ok": true,
                    "document_id": doc_id,
                    "title": title,
                    "chunk_count": chunk_count,
                    "message": "Document ingested successfully",
                }))
            }
            Err(e) => Err(ErrorData::internal_error(format!("Error ingesting document: {e}"), None)),
        }
    }

    #[tool(description = "Get knowledge base statistics: fact count, chunk count, document count, database size, embedding model and dimensions, and stored graph edge count.")]
    fn sm_stats(&self) -> Result<String, ErrorData> {
        let store = &self.bridge.store;
        let result = tokio::task::block_in_place(|| Handle::current().block_on(store.stats()));

        match result {
            Ok(stats) => {
                // Load graph edge count separately — propagates errors
                // instead of hiding them (SM-AUD-016).
                let graph_edge_count = tokio::task::block_in_place(|| Handle::current().block_on(store.list_all_graph_edges()))
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

    #[tool(description = "Find the shortest path between two items in the knowledge graph. Traverses semantic, temporal, causal, entity, and stored graph edges. Returns the path as a list of node IDs with edge evidence for each hop.")]
    fn sm_graph_path(
        &self,
        Parameters(GraphPathParams { from_id, to_id, max_depth }): Parameters<GraphPathParams>,
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
            Err(e) => Err(ErrorData::internal_error(format!("Graph view error: {e}"), None)),
        }
    }

    // ── Feature-gated tools ──────────────────────────────────────────
    // Note: cfg gates are removed from individual tool methods because
    // rmcp's #[tool_router] macro needs all tools visible at expansion
    // time. The `full` feature in Cargo.toml already enables the
    // semantic-memory sub-features these tools depend on.

    #[tool(description = "Profile a query and get an adaptive routing decision. Determines which retrieval stages (BM25, vector, rerank, graph, decoder, discord) should be activated for this query.")]
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

    #[tool(description = "Adaptive search: profiles the query, routes to appropriate stages, and applies factor graph belief propagation if the decoder stage is activated. Returns results with stable IDs for downstream tool chaining, routing decision, decoder status, and factor graph analysis.")]
    fn sm_search_with_routing(
        &self,
        Parameters(SearchWithRoutingParams { query, top_k, contradictions }): Parameters<SearchWithRoutingParams>,
    ) -> Result<String, ErrorData> {
        use semantic_memory::integration::plan_execution;
        use semantic_memory::routing::RetrievalRouter;

        let k = top_k.map(|v| v as usize).unwrap_or(5);
        let router = RetrievalRouter {
            decoder_enabled: true,
            discord_enabled: true,
            corpus_density: 0.5,
            ..Default::default()
        };

        let decision = router.route_query(&query);
        let contras = contradictions.unwrap_or_default();
        let plan = plan_execution(&decision, contras.clone());

        // Execute search — both branches currently call plain search.
        // SM-AUD-007: report decoder_executed=false when decoder is planned
        // but not actually applied to the result ranking.
        let store = &self.bridge.store;
        let search_result = tokio::task::block_in_place(|| Handle::current().block_on(store.search(&query, Some(k), None, None)));

        match search_result {
            Ok(results) => {
                let json_results: Vec<serde_json::Value> = results
                    .iter()
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

                // Track whether decoder actually affected ranking.
                // Currently both branches call plain search, so decoder
                // never affects ranking (SM-AUD-007).
                let decoder_executed = false;

                if decision.decoder {
                    #[cfg(feature = "full")]
                    {
                        use semantic_memory::factor_graph::{
                            factors_from_edges, FactorGraph, FactorGraphConfig,
                        };

                        let graph_edges = tokio::task::block_in_place(|| Handle::current().block_on(
                            store.list_all_graph_edges()
                        ));

                        match graph_edges {
                            Ok(edges) => {
                                let raw_edges: Vec<(String, String, semantic_memory::GraphEdgeType, f64, Option<String>)> =
                                    edges
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

                                let nodes: Vec<(String, f64)> =
                                    results.iter().map(|r| (r.source.result_id(), r.score)).collect();
                                let factors = factors_from_edges(&raw_edges);
                                let graph = FactorGraph::new(&nodes, factors, FactorGraphConfig::default());
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
                        // SM-AUD-007 / MCP-006: renamed from estimated_recall to
                        // heuristic_recall_estimate with recall_basis field.
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
                    "decoder_planned": plan.use_decoder,
                    "decoder_executed": decoder_executed,
                    "factor_graph": factor_graph_payload,
                    "matryoshka": matryoshka_payload,
                }))
            }
            Err(e) => Err(ErrorData::internal_error(format!("Search error: {e}"), None)),
        }
    }

    #[tool(description = "Detect contradictions and inconsistencies in search results. Runs syndrome detection, computes corrections, and applies belief propagation to refine confidence scores. This tool operates on caller-supplied results and does not require graph edges from the store.")]
    fn sm_decoder_analyze(
        &self,
        Parameters(DecoderAnalyzeParams { results, contradictions }): Parameters<DecoderAnalyzeParams>,
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

    #[tool(description = "Second-order retrieval: find items related to your search results through the knowledge graph, but NOT themselves direct hits. Loads graph edges from the store automatically — caller supplies only the direct result IDs.")]
    fn sm_discord_search(
        &self,
        Parameters(DiscordSearchParams { direct_result_ids }): Parameters<DiscordSearchParams>,
    ) -> Result<String, ErrorData> {
        use semantic_memory::discord::DiscordScorer;

        // MCP-001: Load edges from the store, not from caller-supplied params.
        let edges = load_stored_edge_refs(&self.bridge.store)?;
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
            "edges_loaded_from_store": edges.len(),
        }))
    }

    #[tool(description = "Set provenance (evidence confidence) for an item. Uses the ConfidenceSemiring: confidence in [0.0, 1.0] with a support count of independent observations. Returns a provenance receipt.")]
    fn sm_set_provenance(
        &self,
        Parameters(SetProvenanceParams { item_id, confidence, support_count }): Parameters<SetProvenanceParams>,
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

        let result = tokio::task::block_in_place(|| Handle::current().block_on(
            store.set_provenance::<ConfidenceSemiring>(
                &ProvenanceItemType::Fact,
                &item_id,
                &value,
                &[],
                None,
            ),
        ));

        match result {
            Ok(receipt) => json_to_string(&serde_json::json!({
                "ok": true,
                "provenance_id": receipt.provenance_id,
                "item_id": receipt.item_id,
                "semiring_type": receipt.semiring_type,
                "recorded_at": receipt.recorded_at,
                "message": "Provenance set successfully",
            })),
            Err(e) => Err(ErrorData::internal_error(format!("Provenance error: {e}"), None)),
        }
    }

    #[tool(description = "Run a memory lifecycle pass: analyze items for syndromes, compute corrections, identify subtraction candidates, and check if compression recompression is needed. This is the autonomous memory health check.")]
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
        let recompression = should_trigger_recompression(
            subtracted_count,
            remaining_count,
            false,
        );

        let store = &self.bridge.store;
        let graph_edges = tokio::task::block_in_place(|| Handle::current().block_on(
            store.list_all_graph_edges()
        ));
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
                        .map(|v| serde_json::json!({
                            "description": v.description,
                            "void_type": format!("{:?}", v.void_type),
                            "nearby_items": v.nearby_items,
                            "suggested_connections": v.suggested_connections,
                        }))
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
                        .map(|c| serde_json::json!({
                            "id": c.id,
                            "members": c.members,
                            "level": c.level,
                            "parent": c.parent,
                            "member_count": c.members.len(),
                        }))
                        .collect();

                    community_contradictions = community_contradiction_scan(&detected, &[])
                        .into_iter()
                        .map(|cc| serde_json::json!({
                            "community_id": cc.community_id,
                            "item_a": cc.item_a,
                            "item_b": cc.item_b,
                            "description": cc.description,
                        }))
                        .collect();

                    Ok(())
                })();

                if let Err(e) = community_result {
                    community_error = Some(e);
                }

                let subgraph_result = (|| -> Result<(), String> {
                    use std::collections::HashSet;
                    use semantic_memory::integration::autonomous_subgraph_maintenance;
                    use semantic_memory::subgraph_pruning::AccessLog;

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

                    let report = autonomous_subgraph_maintenance(
                        &analysis_edges,
                        &access_logs,
                        &[],
                        0,
                    );
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
        let (f32_count, compressed_count) = item_ids.iter().fold(
            (0usize, 0usize),
            |(f32_count, compressed_count), _| {
                use semantic_memory::compression_governor::{
                    decide_quantization, QuantizationLevel,
                };

                match decide_quantization(0.5) {
                    QuantizationLevel::F32 => (f32_count + 1, compressed_count),
                    _ => (f32_count, compressed_count + 1),
                }
            },
        );
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

    #[tool(description = "Add a durable, typed graph edge between two nodes in the knowledge graph. Nodes use prefixed IDs (e.g. fact:<uuid>, namespace:<name>, document:<id>). Edge types: semantic, temporal, causal, entity. Insertion is idempotent — same edge returns existing ID. Returns the edge ID and metadata.")]
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

        let edge_type = match params.edge_type.as_str() {
            "semantic" => GraphEdgeType::Semantic {
                cosine_similarity: params.cosine_similarity.unwrap_or(0.5),
            },
            "temporal" => GraphEdgeType::Temporal {
                delta_secs: params.delta_secs.unwrap_or(0),
            },
            "causal" => GraphEdgeType::Causal {
                confidence: params.confidence.unwrap_or(0.5),
                evidence_ids: params.evidence_ids.unwrap_or_default(),
            },
            "entity" => GraphEdgeType::Entity {
                relation: params.relation.unwrap_or_else(|| "related".to_string()),
            },
            other => return Err(ErrorData::invalid_params(
                format!("Invalid edge_type '{other}'. Must be one of: semantic, temporal, causal, entity"),
                None,
            )),
        };

        // MCP-004: Reject malformed metadata JSON instead of silently dropping it.
        let metadata = match params.metadata.as_deref() {
            None => None,
            Some(s) => match serde_json::from_str::<serde_json::Value>(s) {
                Ok(v) => Some(v),
                Err(e) => return Err(ErrorData::invalid_params(
                    format!("metadata is not valid JSON: {e}"),
                    None,
                )),
            },
        };

        let store = &self.bridge.store;
        let result = tokio::task::block_in_place(|| Handle::current().block_on(
            store.add_graph_edge(&params.source, &params.target, edge_type, params.weight, metadata)
        ));

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
            Err(e) => Err(ErrorData::internal_error(format!("Error adding graph edge: {e}"), None)),
        }
    }

    #[tool(description = "List graph edges for a specific node (as source or target), or all stored graph edges if no node_id is provided. Returns non-invalidated edges only.")]
    fn sm_list_graph_edges(
        &self,
        Parameters(ListGraphEdgesParams { node_id }): Parameters<ListGraphEdgesParams>,
    ) -> Result<String, ErrorData> {
        let store = &self.bridge.store;
        let result = match node_id {
            Some(id) => tokio::task::block_in_place(|| Handle::current().block_on(
                store.list_graph_edges_for_node(&id)
            )),
            None => tokio::task::block_in_place(|| Handle::current().block_on(
                store.list_all_graph_edges()
            )),
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
            Err(e) => Err(ErrorData::internal_error(format!("Error listing graph edges: {e}"), None)),
        }
    }

    #[tool(description = "Invalidate a stored graph edge by ID. Append-only — the edge row is never deleted, only marked invalidated with a reason.")]
    fn sm_invalidate_graph_edge(
        &self,
        Parameters(InvalidateGraphEdgeParams { edge_id, reason }): Parameters<InvalidateGraphEdgeParams>,
    ) -> Result<String, ErrorData> {
        let store = &self.bridge.store;
        let result = tokio::task::block_in_place(|| Handle::current().block_on(
            store.invalidate_graph_edge(&edge_id, &reason)
        ));

        match result {
            Ok(()) => json_to_string(&serde_json::json!({
                "ok": true,
                "edge_id": edge_id,
                "message": "Edge invalidated successfully",
            })),
            Err(e) => Err(ErrorData::internal_error(format!("Error invalidating edge: {e}"), None)),
        }
    }

    // ── Factor graph, topology, and community tools ─────────────────

    #[tool(description = "Run factor graph belief propagation on heterogeneous graph edges stored in the knowledge base. Models all 4 edge types (semantic, temporal, causal, entity) as factors in a single probabilistic reasoning framework. Loads edges from the store automatically — caller supplies only node initial beliefs and optional config overrides. Returns unified confidence scores after message propagation converges.")]
    fn sm_factor_graph(
        &self,
        Parameters(params): Parameters<FactorGraphParams>,
    ) -> Result<String, ErrorData> {
        use semantic_memory::factor_graph::{
            factors_from_edges, FactorGraph, FactorGraphConfig,
        };

        let defaults = FactorGraphConfig::default();
        let config = FactorGraphConfig {
            semantic_weight: params.semantic_weight.unwrap_or(defaults.semantic_weight),
            temporal_weight: params.temporal_weight.unwrap_or(defaults.temporal_weight),
            causal_weight: params.causal_weight.unwrap_or(defaults.causal_weight),
            entity_weight: params.entity_weight.unwrap_or(defaults.entity_weight),
            self_influence: params.self_influence.unwrap_or(defaults.self_influence),
            max_iterations: params.max_iterations.map(|v| v as usize).unwrap_or(defaults.max_iterations),
            convergence_threshold: params.convergence_threshold.unwrap_or(defaults.convergence_threshold),
        };

        // MCP-001: Load edges from the store, not from caller-supplied params.
        // MCP-002: No hardcoded GraphEdgeType literals — use actual stored values.
        let raw_edges = load_stored_factor_edges(&self.bridge.store)?;
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
            "edges_loaded_from_store": raw_edges.len(),
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

    #[tool(description = "Find topological voids in the knowledge graph. Computes Betti numbers (connected components and independent cycles) and detects structural gaps. Loads edges from the store automatically — caller does not supply edges.")]
    fn sm_topology(&self, Parameters(_params): Parameters<TopologyParams>) -> Result<String, ErrorData> {
        use semantic_memory::topology::{compute_betti_numbers, find_voids, gap_report};

        // MCP-001: Load edges from the store, not from caller-supplied params.
        let edges = load_stored_edge_pairs(&self.bridge.store)?;

        let mut adjacency: std::collections::HashMap<String, Vec<String>> =
            std::collections::HashMap::new();
        for (src, tgt) in &edges {
            adjacency
                .entry(src.clone())
                .or_default()
                .push(tgt.clone());
            adjacency
                .entry(tgt.clone())
                .or_default()
                .push(src.clone());
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

    #[tool(description = "Detect communities in the knowledge graph using a Leiden-inspired algorithm. Loads edges from the store automatically. Returns community assignments with member lists, optional within-community contradiction scans, and optional community-aware compression recommendations.")]
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

        json_to_string(&serde_json::json!({
            "ok": true,
            "communities": communities.iter().map(|c| serde_json::json!({
                "id": c.id,
                "members": c.members,
                "level": c.level,
                "parent": c.parent,
                "member_count": c.members.len(),
            })).collect::<Vec<_>>(),
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
                    (e.source == *from && e.target == *to)
                        || (e.source == *to && e.target == *from)
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
                        semantic_memory::GraphEdgeType::Causal { confidence, evidence_ids } => {
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