//! Tool parameter structs for MCP tools.
//! Each struct derives schemars::JsonSchema so rmcp can auto-generate
//! the JSON Schema for the tool's inputSchema.

use schemars::JsonSchema;
use serde::Deserialize;

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
    /// Edge type: "semantic", "temporal", "causal", or "entity"
    pub edge_type: String,
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
}