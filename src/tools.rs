//! Tool parameter structs for MCP tools.
//! Each struct derives schemars::JsonSchema so rmcp can auto-generate
//! the JSON Schema for the tool's inputSchema.

use schemars::JsonSchema;
use serde::Deserialize;

/// Edge type for graph edges. JSON Schema enum helps LLMs pick the
/// right value without guessing.
#[derive(Debug, Deserialize, JsonSchema)]
pub enum EdgeType {
    /// Semantic similarity edge (requires cosine_similarity)
    Semantic,
    /// Temporal ordering edge (requires delta_secs)
    Temporal,
    /// Causal relationship edge (requires confidence)
    Causal,
    /// Named relationship edge (requires relation)
    Entity,
}

// ─── Core search tools ──────────────────────────────────────────────────

/// Parameters for sm_search
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchParams {
    /// The search query string
    pub query: String,
    /// Maximum number of results to return (default 5)
    #[serde(default)]
    pub top_k: Option<u32>,
    /// Optional namespace filter (restrict search to these namespaces)
    #[serde(default)]
    pub namespaces: Option<Vec<String>>,
}

/// Parameters for sm_search_explained
#[allow(dead_code)]
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchExplainedParams {
    /// The query string
    pub query: String,
    /// Maximum number of results to return (default 5)
    #[serde(default)]
    pub top_k: Option<u32>,
}

/// Parameters for sm_add_fact
#[derive(Debug, Deserialize, JsonSchema)]
pub struct AddFactParams {
    /// The fact content text
    pub content: String,
    /// Namespace to store the fact in (e.g. "general", "research", "coding")
    pub namespace: String,
    /// Optional source attribution
    #[serde(default)]
    pub source: Option<String>,
    /// When true, extract named entities via Ollama and link them as graph edges (opt-in)
    #[serde(default)]
    pub extract_entities: Option<bool>,
    /// Memory kind classification: durable_fact, preference, project_state, instruction_policy,
    /// correction, observation, episode_summary, skill_procedure, ephemeral_inference.
    /// Default: durable_fact. Ephemeral inferences require evidence_refs to promote.
    #[serde(default)]
    pub memory_kind: Option<String>,
    /// Sensitivity class: public, internal, confidential, restricted.
    /// Default: internal. Confidential/restricted facts are blocked from autocapture.
    #[serde(default)]
    pub sensitivity: Option<String>,
    /// Evidence references supporting this fact (URLs, fact IDs, source paths).
    #[serde(default)]
    pub evidence_refs: Option<Vec<String>>,
}

/// Parameters for sm_ingest_document
#[derive(Debug, Deserialize, JsonSchema)]
pub struct IngestDocumentParams {
    /// The document content text
    pub content: String,
    /// Document title
    pub title: String,
    /// Namespace to store the document in
    pub namespace: String,
}

/// Parameters for sm_graph_path
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GraphPathParams {
    /// Starting item ID
    pub from_id: String,
    /// Target item ID
    pub to_id: String,
    /// Maximum BFS depth (default 5)
    #[serde(default)]
    pub max_depth: Option<u32>,
}

// ─── Feature-gated tools (full feature only) ────────────────────────────
// Note: cfg gates removed so rmcp's #[tool_router] macro can see all
// tool parameter types at expansion time. The `full` feature in
// Cargo.toml enables the corresponding semantic-memory sub-features.

/// Parameters for sm_route_query
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RouteQueryParams {
    /// The query string to profile and route
    pub query: String,
}

/// Parameters for sm_detect_contradictions
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DetectContradictionsParams {
    /// The query whose top results are scanned for content contradictions
    pub query: String,
    /// How many top results to scan (default 10)
    #[serde(default)]
    pub top_k: Option<u32>,
}

/// Parameters for sm_search_with_routing
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchWithRoutingParams {
    /// The search query string
    pub query: String,
    /// Maximum number of results (default 5)
    #[serde(default)]
    pub top_k: Option<u32>,
    /// Known contradiction pairs (item_a, item_b)
    #[serde(default)]
    pub contradictions: Option<Vec<(String, String)>>,
    /// When true, group search results by knowledge graph community membership
    #[serde(default)]
    pub group_by_community: Option<bool>,
}

/// Parameters for sm_set_provenance
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SetProvenanceParams {
    /// The item ID to set provenance for
    pub item_id: String,
    /// Confidence score 0.0-1.0
    pub confidence: f64,
    /// Number of supporting observations
    pub support_count: u32,
}

/// Parameters for sm_decoder_analyze
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DecoderAnalyzeParams {
    /// Search results as (item_id, score) pairs
    pub results: Vec<(String, f64)>,
    /// Known contradiction pairs
    #[serde(default)]
    pub contradictions: Option<Vec<(String, String)>>,
}

/// Parameters for sm_discord_search
///
/// MCP-001: graph_edges field removed — edges are now loaded from the store
/// automatically. Caller supplies only the direct result IDs.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DiscordSearchParams {
    /// Direct result item IDs (the top-K from search)
    pub direct_result_ids: Vec<String>,
}

/// Parameters for sm_run_lifecycle
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RunLifecycleParams {
    /// Item IDs to process in the lifecycle pass
    pub item_ids: Vec<String>,
}

// ─── Graph edge tools ──────────────────────────────────────────────────

/// Parameters for sm_add_graph_edge
#[derive(Debug, Deserialize, JsonSchema)]
pub struct AddGraphEdgeParams {
    /// Source node ID (prefixed, e.g. "fact:<uuid>", "namespace:<name>")
    pub source: String,
    /// Target node ID (prefixed)
    pub target: String,
    /// Edge type: semantic, temporal, causal, or entity
    pub edge_type: EdgeType,
    /// Edge weight (default 1.0)
    #[serde(default = "default_weight")]
    pub weight: f64,
    /// For semantic edges: cosine similarity (0.0-1.0). Ignored for other types.
    #[serde(default)]
    pub cosine_similarity: Option<f32>,
    /// For temporal edges: time delta in seconds. Ignored for other types.
    #[serde(default)]
    pub delta_secs: Option<u64>,
    /// For causal edges: confidence (0.0-1.0). Ignored for other types.
    #[serde(default)]
    pub confidence: Option<f32>,
    /// For causal edges: evidence IDs. Ignored for other types.
    #[serde(default)]
    pub evidence_ids: Option<Vec<String>>,
    /// For entity edges: relationship name (e.g. "mentions", "modifies"). Ignored for other types.
    #[serde(default)]
    pub relation: Option<String>,
    /// Optional metadata as a JSON object string
    #[serde(default)]
    pub metadata: Option<String>,
}

fn default_weight() -> f64 {
    1.0
}

/// Parameters for sm_list_graph_edges
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ListGraphEdgesParams {
    /// Node ID to list edges for. If omitted, lists all edges.
    #[serde(default)]
    pub node_id: Option<String>,
}

/// Parameters for sm_invalidate_graph_edge
#[derive(Debug, Deserialize, JsonSchema)]
pub struct InvalidateGraphEdgeParams {
    /// Edge ID to invalidate
    pub edge_id: String,
    /// Reason for invalidation
    pub reason: String,
}

// ─── Factor graph tools ─────────────────────────────────────────────────

/// A node input for the factor graph.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct FactorGraphNodeInput {
    /// Node/item ID (e.g. "fact:<uuid>")
    pub item_id: String,
    /// Initial belief score (0.0-1.0, from provenance or search score)
    pub initial_belief: f64,
}

/// Parameters for sm_factor_graph
///
/// MCP-001: edges field removed — edges are now loaded from the store
/// automatically. Caller supplies only node initial beliefs and optional
/// config overrides.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct FactorGraphParams {
    /// Nodes with their initial belief scores
    pub nodes: Vec<FactorGraphNodeInput>,
    /// Weight for semantic factors (default 0.35)
    #[serde(default)]
    pub semantic_weight: Option<f64>,
    /// Weight for temporal factors (default 0.20)
    #[serde(default)]
    pub temporal_weight: Option<f64>,
    /// Weight for causal factors (default 0.30)
    #[serde(default)]
    pub causal_weight: Option<f64>,
    /// Weight for entity factors (default 0.15)
    #[serde(default)]
    pub entity_weight: Option<f64>,
    /// How much the node's own initial belief matters vs neighbor influence (default 0.6)
    #[serde(default)]
    pub self_influence: Option<f64>,
    /// Max iterations for message passing (default 50)
    #[serde(default)]
    pub max_iterations: Option<u32>,
    /// Convergence threshold (default 0.001)
    #[serde(default)]
    pub convergence_threshold: Option<f64>,
}

// ─── Topology tools ─────────────────────────────────────────────────────

/// Parameters for sm_topology
///
/// MCP-001: edges field removed — edges are now loaded from the store
/// automatically. The params struct is kept for schema compatibility but
/// the edges field is no longer used.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct TopologyParams {
    // No caller-supplied edges — loaded from store internally.
}

// ─── Community tools ────────────────────────────────────────────────────

/// Parameters for sm_community
///
/// MCP-001: edges field removed — edges are now loaded from the store
/// automatically. Configuration params (resolution, seed, contradictions,
/// importance_scores) remain caller-supplied.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CommunityParams {
    /// Resolution parameter for community detection (default 1.0). Higher values favor smaller communities.
    #[serde(default)]
    pub resolution: Option<f64>,
    /// Random seed for deterministic results (default 42)
    #[serde(default)]
    pub seed: Option<u64>,
    /// Optional contradiction pairs to scan within communities
    #[serde(default)]
    pub contradictions: Option<Vec<(String, String)>>,
    /// Optional importance scores per item for community-aware compression recommendations
    #[serde(default)]
    pub importance_scores: Option<Vec<(String, f64)>>,
    /// When true, generate an LLM summary for each community via Ollama (opt-in)
    #[serde(default)]
    pub summarize: Option<bool>,
}

// ─── Direct read and supersession tools (v0.3.1) ─────────────────────────

/// Parameters for sm_get_fact
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetFactParams {
    /// The fact id. Accepts a bare UUID or a prefixed id like "fact:<uuid>".
    pub fact_id: String,
}

/// Parameters for sm_list_facts
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ListFactsParams {
    /// Namespace to enumerate facts from.
    pub namespace: String,
    /// Maximum facts to return (default 50).
    #[serde(default)]
    pub limit: Option<u32>,
    /// Offset for pagination (default 0).
    #[serde(default)]
    pub offset: Option<u32>,
}

/// Parameters for sm_get_fact_neighbors
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetFactNeighborsParams {
    /// The node id whose neighbors to fetch (bare UUID or prefixed "fact:<uuid>").
    pub item_id: String,
}

/// Parameters for sm_supersede_fact
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SupersedeFactParams {
    /// Existing stale fact id. Accepts a bare UUID or prefixed "fact:<uuid>".
    pub old_fact_id: String,
    /// Replacement fact content.
    pub content: String,
    /// Optional namespace for the replacement fact. Defaults to the old fact's namespace.
    #[serde(default)]
    pub namespace: Option<String>,
    /// Optional source attribution for the replacement fact.
    #[serde(default)]
    pub source: Option<String>,
    /// Optional reason stored on the supersedes graph edge.
    #[serde(default)]
    pub reason: Option<String>,
}

// ─── Conversation / session tools (v0.3.0) ──────────────────────────────

/// Parameters for sm_create_session
#[allow(dead_code)]
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CreateSessionParams {
    /// Channel/label for the session (e.g. "claude-code", "project-x").
    pub channel: String,
    /// Optional metadata as a JSON object string.
    #[serde(default)]
    pub metadata: Option<String>,
}

/// Parameters for sm_add_message
#[allow(dead_code)]
#[derive(Debug, Deserialize, JsonSchema)]
pub struct AddMessageParams {
    /// Session id to append to.
    pub session_id: String,
    /// Role: "user", "assistant", "system", or "tool".
    pub role: String,
    /// Message content (embedded + FTS-indexed for conversation search).
    pub content: String,
}

/// Parameters for sm_list_sessions
#[allow(dead_code)]
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ListSessionsParams {
    /// Maximum sessions to return (default 20).
    #[serde(default)]
    pub limit: Option<u32>,
    /// Offset for pagination (default 0).
    #[serde(default)]
    pub offset: Option<u32>,
}

/// Parameters for sm_get_messages
#[allow(dead_code)]
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetMessagesParams {
    /// Session id to read messages from.
    pub session_id: String,
    /// Return the most recent messages fitting within this token budget (default 4000).
    #[serde(default)]
    pub max_tokens: Option<u32>,
}

/// Parameters for sm_search_conversations
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchConversationsParams {
    /// The search query string.
    pub query: String,
    /// Maximum number of results (default 5).
    #[serde(default)]
    pub top_k: Option<u32>,
}

// ─── Delete / forget tools (admin-ops) ──────────────────────────────────

/// Parameters for sm_delete_fact
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DeleteFactParams {
    /// The fact id to permanently delete (bare UUID or prefixed "fact:<uuid>").
    pub fact_id: String,
}

/// Parameters for sm_delete_namespace
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DeleteNamespaceParams {
    /// The namespace to permanently delete (all its facts, documents, chunks, and namespaced sessions).
    pub namespace: String,
}

/// Parameters for sm_update_fact
#[derive(Debug, Deserialize, JsonSchema)]
pub struct UpdateFactParams {
    /// The fact ID to update (with or without "fact:" prefix).
    pub fact_id: String,
    /// The new content for the fact.
    pub content: String,
}

/// Parameters for sm_consolidate_facts
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ConsolidateFactsParams {
    /// First fact ID (will be kept and updated with merged content).
    pub keep_id: String,
    /// Second fact ID (will be superseded by the kept fact).
    pub supersede_id: String,
    /// Optional merged content. If not provided, content from both facts will be concatenated.
    pub merged_content: Option<String>,
}

// ─── RL routing feedback tool ──────────────────────────────────────────

/// Parameters for sm_record_outcome
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RecordOutcomeParams {
    /// The query string that was routed.
    pub query: String,
    /// The outcome of the routing decision: "good", "bad", or "neutral".
    pub outcome: String,
}

// ─── Claim-ledger integration ──────────────────────────────────────────

/// Parameters for sm_create_claim
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CreateClaimParams {
    /// The fact ID to create a claim from (with or without "fact:" prefix).
    pub fact_id: String,
    /// Optional source span description (e.g., "line 42-58").
    pub source_span: Option<String>,
}

/// Parameters for sm_add_evidence
#[derive(Debug, Deserialize, JsonSchema)]
pub struct AddEvidenceParams {
    /// The claim ID to add evidence to.
    pub claim_id: String,
    /// The evidence text supporting the claim.
    pub evidence_text: String,
    /// Optional source type (e.g., "document", "web", "experiment").
    pub source_type: Option<String>,
}

/// Parameters for sm_judge_support
#[derive(Debug, Deserialize, JsonSchema)]
pub struct JudgeSupportParams {
    /// The claim ID to judge.
    pub claim_id: String,
    /// The support judgment: "supported", "unsupported", "contested", or "heuristic_only".
    pub judgment: String,
    /// Optional rationale for the judgment.
    pub rationale: Option<String>,
}

// ─── Bitemporal search ─────────────────────────────────────────────────

/// Parameters for sm_search_as_of
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchAsOfParams {
    /// The search query.
    pub query: String,
    /// The date to search as of (ISO 8601, e.g., "2026-01-15T00:00:00Z").
    pub as_of_date: String,
    /// Maximum number of results (default: 5).
    #[serde(default)]
    pub top_k: Option<usize>,
    /// Optional namespace filter.
    pub namespace: Option<String>,
}

// ─── Verification gate ─────────────────────────────────────────────────

/// Parameters for sm_verify_claim
#[derive(Debug, Deserialize, JsonSchema)]
pub struct VerifyClaimParams {
    /// The claim text to verify.
    pub claim: String,
    /// Risk class: low, medium, high, critical.
    pub risk_class: String,
    /// Optional evidence references supporting the claim.
    #[serde(default)]
    pub evidence_refs: Option<Vec<String>>,
    /// Whether refutation was attempted (if false, high/critical claims cannot be promoted).
    #[serde(default)]
    pub refutation_attempted: Option<bool>,
}

// ─── Search receipt tools ──────────────────────────────────────────────

/// Parameters for sm_get_search_receipt
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetSearchReceiptParams {
    /// The receipt/request ID to look up.
    pub receipt_id: String,
}

/// Parameters for sm_replay_search_receipt
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ReplaySearchReceiptParams {
    /// The receipt/request ID to replay.
    pub receipt_id: String,
    /// The query text to use for replay (receipts don't store query text).
    pub query: String,
    /// Optional top_k for replay (defaults to original receipt's result count).
    #[serde(default)]
    pub top_k: Option<u32>,
    /// Optional namespace filter for replay.
    #[serde(default)]
    pub namespaces: Option<Vec<String>>,
}

// ─── Reconcile tool ───────────────────────────────────────────────────

/// Parameters for sm_reconcile
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ReconcileParams {
    /// Action to take: "report_only", "rebuild_fts", or "re_embed".
    pub action: String,
}

// ─── Maintenance tools ────────────────────────────────────────────────

/// Parameters for sm_embeddings_are_dirty (no params needed, but struct for consistency)
#[derive(Debug, Deserialize, JsonSchema)]
pub struct EmbeddingsAreDirtyParams {}

/// Parameters for sm_list_imports
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ListImportsParams {
    /// Optional namespace filter.
    #[serde(default)]
    pub namespace: Option<String>,
    /// Maximum number of import records to return (default 20).
    #[serde(default)]
    pub limit: Option<u32>,
}

/// Parameters for sm_import_status
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ImportStatusParams {
    /// The envelope ID to check import status for.
    pub envelope_id: String,
}

/// Parameters for sm_import_envelope
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ImportEnvelopeParams {
    /// The import envelope as a JSON string.
    pub envelope_json: String,
}

// ─── Projection query tools ───────────────────────────────────────────

// ─── Knowledge-runtime orchestration tools ─────────────────────────────

/// Parameters for sm_classify_query
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct ClassifyQueryParams {
    /// The query string to classify.
    pub query: String,
}

/// Parameters for sm_plan_query
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct PlanQueryParams {
    /// The query string to plan.
    pub query: String,
    /// Optional namespace for the scope.
    #[serde(default)]
    pub namespace: Option<String>,
    /// Optional domain for the scope.
    #[serde(default)]
    pub domain: Option<String>,
    /// Optional workspace ID for the scope.
    #[serde(default)]
    pub workspace_id: Option<String>,
    /// Optional repo ID for the scope.
    #[serde(default)]
    pub repo_id: Option<String>,
}

/// Parameters for sm_query_orchestrated
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct QueryOrchestratedParams {
    /// The query string.
    pub query: String,
    /// Optional namespace for the scope.
    #[serde(default)]
    pub namespace: Option<String>,
    /// Optional domain for the scope.
    #[serde(default)]
    pub domain: Option<String>,
    /// Optional workspace ID for the scope.
    #[serde(default)]
    pub workspace_id: Option<String>,
    /// Optional repo ID for the scope.
    #[serde(default)]
    pub repo_id: Option<String>,
    /// Maximum results (default 10).
    #[serde(default)]
    pub top_k: Option<usize>,
    /// When true, include the query trace in the response.
    #[serde(default)]
    pub trace: Option<bool>,
}

/// Parameters for sm_query_temporal
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct QueryTemporalKParams {
    /// The query string.
    pub query: String,
    /// Valid-time as-of date (ISO 8601, e.g. "2026-01-15T00:00:00Z").
    pub as_of_date: String,
    /// Optional namespace for the scope.
    #[serde(default)]
    pub namespace: Option<String>,
    /// Optional domain for the scope.
    #[serde(default)]
    pub domain: Option<String>,
    /// Optional workspace ID for the scope.
    #[serde(default)]
    pub workspace_id: Option<String>,
    /// Optional repo ID for the scope.
    #[serde(default)]
    pub repo_id: Option<String>,
    /// Maximum results (default 5).
    #[serde(default)]
    pub top_k: Option<usize>,
}

/// Parameters for sm_entity_lookup
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct EntityLookupParams {
    /// The entity mention to resolve.
    pub mention: String,
    /// Optional namespace for the scope.
    #[serde(default)]
    pub namespace: Option<String>,
    /// Optional domain for the scope.
    #[serde(default)]
    pub domain: Option<String>,
}

/// Parameters for sm_projection_health
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct ProjectionHealthParams {
    /// Namespace for the projection scope.
    pub namespace: String,
    /// Optional projection kind (entity, temporal, route_stats, or custom string).
    #[serde(default)]
    pub projection_kind: Option<String>,
}

// ─── Claim-ledger completion tools ──────────────────────────────────────

/// Parameters for sm_proof_debt_status
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct ProofDebtStatusParams {
    /// Scope this budget applies to (e.g., claim_id, case_id, "session").
    pub scope: String,
}

/// Parameters for sm_evaluate_proof_debt_gate
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct EvaluateProofDebtGateParams {
    /// Scope this budget applies to.
    pub scope: String,
    /// Budget in micros (1_000_000 = one full proof unit). Default 1_000_000.
    #[serde(default)]
    pub budget_micros: Option<u64>,
}

/// Parameters for sm_add_support_admission
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct AddSupportAdmissionParams {
    /// The claim being admitted.
    pub claim_id: String,
    /// The admission method: "operator_admitted", "test_fixture_admitted", or "external_receipt_admitted".
    pub method: String,
    /// Rationale for the admission.
    pub rationale: String,
    /// Optional operator reference (if operator-admitted).
    #[serde(default)]
    pub operator_id: Option<String>,
}

/// Parameters for sm_record_contradiction
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct RecordContradictionParams {
    /// First claim ID involved in the contradiction.
    pub claim_a_id: String,
    /// Second claim ID involved in the contradiction.
    pub claim_b_id: String,
    /// Detection method/pattern (e.g., "numeric_disagreement", "negation_conflict").
    pub detection_method: String,
    /// Optional evidence supporting the contradiction finding.
    #[serde(default)]
    pub evidence: Option<String>,
}

/// Parameters for sm_resolve_contradiction
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct ResolveContradictionParams {
    /// The contradiction ID to resolve.
    pub contradiction_id: String,
    /// Resolution outcome: "confirmed", "rejected", or "superseded".
    pub resolution: String,
    /// Rationale for the resolution.
    pub rationale: String,
    /// Optional superseding claim ID if resolution is "superseded".
    #[serde(default)]
    pub superseding_claim_id: Option<String>,
}

/// Parameters for sm_verify_ledger
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct VerifyLedgerParams {
    /// Ledger entries as JSONL (newline-delimited JSON).
    pub entries_jsonl: String,
}

/// Parameters for sm_export_claim_bundle
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct ExportClaimBundleParams {
    /// Claim IDs to include in the export.
    pub claim_ids: Vec<String>,
    /// When true, include evidence bundles in the export (default true).
    #[serde(default)]
    pub include_evidence: Option<bool>,
    /// When true, include contradiction records in the export (default true).
    #[serde(default)]
    pub include_contradictions: Option<bool>,
}

/// Parameters for sm_supersede_claim
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct SupersedeClaimParams {
    /// The old claim ID being superseded.
    pub old_claim_id: String,
    /// The new claim ID that supersedes it.
    pub new_claim_id: String,
    /// Rationale for the supersession.
    pub rationale: String,
}

// ─── Projection query tools ───────────────────────────────────────────

/// Parameters for projection query tools (sm_query_claim_versions, etc.)
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ProjectionQueryParams {
    /// Namespace to query within.
    pub namespace: String,
    /// Optional domain scope filter.
    #[serde(default)]
    pub domain: Option<String>,
    /// Optional workspace ID scope filter.
    #[serde(default)]
    pub workspace_id: Option<String>,
    /// Optional repo ID scope filter.
    #[serde(default)]
    pub repo_id: Option<String>,
    /// Optional free-text query.
    #[serde(default)]
    pub text_query: Option<String>,
    /// Optional valid-time as-of filter (ISO 8601).
    #[serde(default)]
    pub valid_at: Option<String>,
    /// Optional transaction-time cutoff (ISO 8601).
    #[serde(default)]
    pub recorded_at_or_before: Option<String>,
    /// Optional subject entity ID filter.
    #[serde(default)]
    pub subject_entity_id: Option<String>,
    /// Optional canonical entity ID filter.
    #[serde(default)]
    pub canonical_entity_id: Option<String>,
    /// Optional claim state filter.
    #[serde(default)]
    pub claim_state: Option<String>,
    /// Optional claim ID filter.
    #[serde(default)]
    pub claim_id: Option<String>,
    /// Optional claim version ID filter.
    #[serde(default)]
    pub claim_version_id: Option<String>,
    /// Maximum results to return (default 10).
    #[serde(default)]
    pub limit: Option<u32>,
}
