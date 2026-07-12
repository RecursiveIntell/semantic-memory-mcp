//! Input contracts compiled into the stable-only MCP artifact.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

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

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchWitnessedParams {
    pub query: String,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub namespaces: Option<Vec<String>>,
    /// Optional caller correlation ID; generated when omitted.
    #[serde(default)]
    pub request_id: Option<String>,
    /// Retrieval stage selection. Defaults to the current hybrid behavior.
    #[serde(default)]
    pub retrieval_mode: Option<RetrievalModeParam>,
    /// Replay input retention. Defaults to no_replay for privacy.
    #[serde(default)]
    pub replay_mode: Option<ReplayModeParam>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RetrievalModeParam {
    Hybrid,
    FtsOnly,
    VectorOnly,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ReplayModeParam {
    NoReplay,
    StoreInputs,
}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct GovernedNamespaceScopeParams {
    pub namespace: String,
    #[serde(default)]
    pub domain: Option<String>,
    #[serde(default)]
    pub workspace_id: Option<String>,
    #[serde(default)]
    pub repo_id: Option<String>,
}

#[derive(Debug, Clone, Copy, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum GovernedAccessPurposeParam {
    Recall,
    Assertion,
    Action,
    Export,
    Replay,
    Admin,
}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct GovernedLeaseParams {
    pub lease_id: String,
    pub delegator: String,
    pub delegatee: String,
    pub purposes: Vec<GovernedAccessPurposeParam>,
    pub scope: GovernedNamespaceScopeParams,
    #[serde(default)]
    pub audiences: Vec<String>,
    pub expires_at: String,
    #[serde(default)]
    pub revoked: bool,
    #[serde(default)]
    pub elevation: bool,
}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct GovernedDecisionParams {
    pub fact_id: String,
    pub caller: String,
    pub subject: String,
    pub audiences: Vec<String>,
    pub scope: GovernedNamespaceScopeParams,
    #[serde(default)]
    pub delegation_or_elevation: Option<GovernedLeaseParams>,
}

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
    /// Optional caller-provided idempotency key. Retries with the same key and
    /// payload return the original fact; omit it for a distinct append.
    #[serde(default)]
    pub idempotency_key: Option<String>,
}

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

#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetFactParams {
    /// The fact id. Accepts a bare UUID or a prefixed id like "fact:<uuid>".
    pub fact_id: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetFactNeighborsParams {
    /// The node id whose neighbors to fetch (bare UUID or prefixed "fact:<uuid>").
    pub item_id: String,
}

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

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchConversationsParams {
    /// The search query string.
    pub query: String,
    /// Maximum number of results (default 5).
    #[serde(default)]
    pub top_k: Option<u32>,
}
