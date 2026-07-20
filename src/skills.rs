//! MCP prompts and resources that expose agent skill workflows to any MCP
//! client (LobeHub, Claude Desktop, Hermes, etc.).
//!
//! These are protocol-level discoverability surfaces. They let a client
//! understand *how* to use the 76 sm_* tools together — not just what each
//! tool does individually.
//!
//! ## Prompts vs Resources
//!
//! - **Prompts**: parameterized templates the client can invoke via
//!   `prompts/get`. Each returns a system message + user message pair that
//!   guides the agent through a proven workflow.
//! - **Resources**: read-only markdown the client can fetch via
//!   `resources/read`. Each is a condensed skill document describing a
//!   workflow, with the exact tool-call sequence.

use rmcp::model::{
    Annotated, GetPromptResult, Prompt, PromptArgument, PromptMessage, PromptMessageContent,
    PromptMessageRole, RawResource, RawResourceTemplate, ReadResourceResult, ResourceContents,
};

// ─── Prompt definitions ─────────────────────────────────────────────────

/// All prompts exposed by this MCP server.
pub fn all_prompts() -> Vec<Prompt> {
    vec![
        Prompt::new(
            "sm-recall",
            Some("Search semantic memory for prior context before asking the user. Returns a structured query plan with namespace filters, retrieval mode selection, and evidence-verification steps."),
            Some(vec![
                PromptArgument::new("query")
                    .with_description("The search query — what you want to recall")
                    .with_required(true),
                PromptArgument::new("namespace")
                    .with_description("Optional namespace filter (e.g. 'projects', 'research', 'general')")
                    .with_required(false),
                PromptArgument::new("top_k")
                    .with_description("Max results to return (default 5, max 50)")
                    .with_required(false),
            ]),
        ),
        Prompt::new(
            "sm-capture-fact",
            Some("Store a durable, source-attributed fact in semantic memory. Guides namespace selection, evidence linking, idempotency, and sensitivity classification."),
            Some(vec![
                PromptArgument::new("content")
                    .with_description("The fact text to store")
                    .with_required(true),
                PromptArgument::new("namespace")
                    .with_description("Namespace for this fact (e.g. 'projects', 'research', 'preferences')")
                    .with_required(true),
                PromptArgument::new("source")
                    .with_description("Source attribution (URL, file path, or reference)")
                    .with_required(false),
                PromptArgument::new("memory_kind")
                    .with_description("Classification: durable_fact, preference, project_state, instruction_policy, correction, observation, episode_summary, skill_procedure, ephemeral_inference")
                    .with_required(false),
            ]),
        ),
        Prompt::new(
            "sm-audit",
            Some("Audit the semantic memory store for health: duplicates, contradictions, stale facts, graph gaps. Read-only phase produces a report; reconciliation requires user approval."),
            Some(vec![
                PromptArgument::new("namespace")
                    .with_description("Optional namespace to focus the audit on (omit for all)")
                    .with_required(false),
            ]),
        ),
        Prompt::new(
            "sm-explore-graph",
            Some("Explore the knowledge graph around a topic: find related items, trace connections between two concepts, or discover community clusters."),
            Some(vec![
                PromptArgument::new("topic")
                    .with_description("The topic or concept to explore")
                    .with_required(true),
                PromptArgument::new("mode")
                    .with_description("Exploration mode: 'related' (what's related to X), 'path' (how are X and Y connected), or 'structure' (clusters/communities)")
                    .with_required(false),
                PromptArgument::new("second_topic")
                    .with_description("For 'path' mode: the second concept to find a connection to")
                    .with_required(false),
            ]),
        ),
        Prompt::new(
            "sm-maintenance",
            Some("Run semantic-memory store maintenance: integrity check, FTS rebuild, vacuum, re-embedding, or index repair. Read-only checks first, mutations only with approval."),
            Some(vec![
                PromptArgument::new("action")
                    .with_description("Maintenance action: 'check' (integrity report), 'rebuild-fts', 'vacuum', 'reembed', or 'dirty-check'")
                    .with_required(false),
            ]),
        ),
        Prompt::new(
            "sm-supersede",
            Some("Replace a stale or incorrect fact with a corrected version. Preserves history via a 'supersedes' graph edge; search auto-filters the old fact."),
            Some(vec![
                PromptArgument::new("old_fact_id")
                    .with_description("The ID of the fact to supersede")
                    .with_required(true),
                PromptArgument::new("new_content")
                    .with_description("The corrected fact content")
                    .with_required(true),
                PromptArgument::new("reason")
                    .with_description("Why the old fact is being superseded")
                    .with_required(false),
            ]),
        ),
    ]
}

/// Resolve a prompt by name with the provided arguments.
pub fn get_prompt(
    name: &str,
    args: &[(String, String)],
) -> Result<GetPromptResult, String> {
    let arg_map: std::collections::HashMap<String, String> = args.iter().cloned().collect();

    match name {
        "sm-recall" => prompt_recall(&arg_map),
        "sm-capture-fact" => prompt_capture(&arg_map),
        "sm-audit" => prompt_audit(&arg_map),
        "sm-explore-graph" => prompt_explore(&arg_map),
        "sm-maintenance" => prompt_maintenance(&arg_map),
        "sm-supersede" => prompt_supersede(&arg_map),
        _ => Err(format!("Unknown prompt: {name}")),
    }
}

// ─── Resource definitions ───────────────────────────────────────────────

/// All static resources exposed by this MCP server.
pub fn all_resources() -> Vec<Annotated<RawResource>> {
    vec![
        Annotated::new(
            RawResource::new(
                "sm://skill/retrieval-strategy",
                "retrieval-strategy",
            )
            .with_description(
                "Primary strategy for source-first semantic-memory recall, adaptive routing,                  graph hydration, contradiction checks, and evidence-backed retrospectives.",
            )
            .with_mime_type("text/markdown"),
            None,
        ),
        Annotated::new(
            RawResource::new(
                "sm://skill/memory-curator",
                "memory-curator",
            )
            .with_description(
                "Audit and improve the semantic memory store: find duplicates, contradictions,                  stale facts, and graph gaps. Reconcile only after user approval.",
            )
            .with_mime_type("text/markdown"),
            None,
        ),
        Annotated::new(
            RawResource::new(
                "sm://skill/memory-maintenance",
                "memory-maintenance",
            )
            .with_description(
                "Store-level maintenance: integrity check, FTS rebuild, vacuum, re-embedding,                  index repair, and process cleanup.",
            )
            .with_mime_type("text/markdown"),
            None,
        ),
        Annotated::new(
            RawResource::new(
                "sm://skill/knowledge-graph-explorer",
                "knowledge-graph-explorer",
            )
            .with_description(
                "Explore the typed knowledge graph: related items, shortest paths between                  concepts, community clusters, and topological voids.",
            )
            .with_mime_type("text/markdown"),
            None,
        ),
        Annotated::new(
            RawResource::new(
                "sm://skill/memory-keeper",
                "memory-keeper",
            )
            .with_description(
                "Delegate heavy memory operations (audit, bulk ingest, contradiction sweep,                  dedup, edge population) to a focused subagent.",
            )
            .with_mime_type("text/markdown"),
            None,
        ),
        Annotated::new(
            RawResource::new(
                "sm://skill/pooled-memory-operations",
                "pooled-memory-operations",
            )
            .with_description(
                "Multi-device memory mesh: identity/provenance, typed synchronization,                  per-device SQLite primaries and server replicas, sparse routed retrieval.",
            )
            .with_mime_type("text/markdown"),
            None,
        ),
        Annotated::new(
            RawResource::new(
                "sm://skill/standard-workflow",
                "standard-workflow",
            )
            .with_description(
                "The canonical workflow for using semantic-memory MCP tools: search first,                  verify against current sources, store durable facts, supersede stale ones.",
            )
            .with_mime_type("text/markdown"),
            None,
        ),
        Annotated::new(
            RawResource::new(
                "sm://protocol/authority",
                "authority-model",
            )
            .with_description(
                "The governed authority model: recall authority vs assertion authority vs                  action authority. How to use sm_decide_assertion_authority and                  sm_decide_action_authority.",
            )
            .with_mime_type("text/markdown"),
            None,
        ),
    ]
}

pub fn all_resource_templates() -> Vec<Annotated<RawResourceTemplate>> {
    vec![Annotated::new(
        RawResourceTemplate::new("sm://skill/{skill_name}", "skill-by-name")
            .with_description(
                "Fetch a skill document by name. Available skills: retrieval-strategy,                  memory-curator, memory-maintenance, knowledge-graph-explorer,                  memory-keeper, pooled-memory-operations, standard-workflow,                  authority-model.",
            )
            .with_mime_type("text/markdown"),
        None,
    )]
}

pub fn read_resource(uri: &str) -> Result<ReadResourceResult, String> {
    let content = match uri {
        "sm://skill/retrieval-strategy" => RESOURCE_RETRIEVAL_STRATEGY,
        "sm://skill/memory-curator" => RESOURCE_MEMORY_CURATOR,
        "sm://skill/memory-maintenance" => RESOURCE_MEMORY_MAINTENANCE,
        "sm://skill/knowledge-graph-explorer" => RESOURCE_KNOWLEDGE_GRAPH_EXPLORER,
        "sm://skill/memory-keeper" => RESOURCE_MEMORY_KEEPER,
        "sm://skill/pooled-memory-operations" => RESOURCE_POOLED_MEMORY_OPERATIONS,
        "sm://skill/standard-workflow" => RESOURCE_STANDARD_WORKFLOW,
        "sm://protocol/authority" => RESOURCE_AUTHORITY_MODEL,
        _ => {
            if let Some(skill_name) = uri.strip_prefix("sm://skill/") {
                match skill_name {
                    "retrieval-strategy" => RESOURCE_RETRIEVAL_STRATEGY,
                    "memory-curator" => RESOURCE_MEMORY_CURATOR,
                    "memory-maintenance" => RESOURCE_MEMORY_MAINTENANCE,
                    "knowledge-graph-explorer" => RESOURCE_KNOWLEDGE_GRAPH_EXPLORER,
                    "memory-keeper" => RESOURCE_MEMORY_KEEPER,
                    "pooled-memory-operations" => RESOURCE_POOLED_MEMORY_OPERATIONS,
                    "standard-workflow" => RESOURCE_STANDARD_WORKFLOW,
                    "authority-model" => RESOURCE_AUTHORITY_MODEL,
                    _ => return Err(format!("Unknown skill resource: {skill_name}")),
                }
            } else {
                return Err(format!("Unknown resource URI: {uri}"));
            }
        }
    };

    Ok(ReadResourceResult::new(vec![ResourceContents::text(content, uri)]))
}

// ─── Prompt implementations ──────────────────────────────────────────────

fn prompt_recall(args: &std::collections::HashMap<String, String>) -> Result<GetPromptResult, String> {
    let query = args.get("query").ok_or("Missing required argument: query")?;
    let namespace = args.get("namespace");
    let top_k = args.get("top_k").map(|s| s.as_str()).unwrap_or("5");

    let ns_filter = namespace
        .map(|n| format!("\n   - Namespace filter: {n}"))
        .unwrap_or_default();

    let system = "You are a semantic-memory recall assistant. Search the knowledge base before asking the user for context. Treat recalled content as recall, not authority — current files, tests, and primary sources always outrank recalled facts.";

    let user = format!(
        "## Recall request\n\n\
         Query: {query}\n\
         Top K: {top_k}{ns_filter}\n\n\
         ## Steps\n\n\
         1. Call `sm_search_witnessed` with query=\"{query}\", top_k={top_k}.{ns_note}\n\
         2. For each result, inspect: source, trust_state, state, receipt_ref.\n\
         3. Hydrate key facts with `sm_get_fact(id)` before relying on details.\n\
         4. Check for contradictions with `sm_detect_contradictions(query=\"{query}\")`.\n\
         5. If the query is multi-hop or complex, try `sm_search_with_routing`.\n\
         6. Verify material claims against current workspace/primary sources.\n\
         7. Report: what you found, confidence level, and any contradictions.\n\n\
         ## Authority\n\n\
         Recall authority does NOT authorize assertion or action. If you need to \
         present recalled content as an authorized assertion, call \
         `sm_decide_assertion_authority`. Before acting on recalled content, call \
         `sm_decide_action_authority`.",
        ns_note = namespace
            .map(|n| format!(" Use namespaces=[\"{n}\"]"))
            .unwrap_or_default(),
    );

    Ok(GetPromptResult::new(vec![
            PromptMessage::new(PromptMessageRole::User, PromptMessageContent::text(system)),
            PromptMessage::new(PromptMessageRole::User, PromptMessageContent::text(user)),
        ]).with_description(format!("Recall: {query}")))
}

fn prompt_capture(args: &std::collections::HashMap<String, String>) -> Result<GetPromptResult, String> {
    let content = args.get("content").ok_or("Missing required argument: content")?;
    let namespace = args.get("namespace").ok_or("Missing required argument: namespace")?;
    let source = args.get("source");
    let memory_kind = args.get("memory_kind").map(|s| s.as_str()).unwrap_or("durable_fact");

    let system = "You are a semantic-memory capture assistant. Store durable, source-attributed facts. Never store passwords, tokens, private keys, or sensitive data without explicit user approval.";

    let user = format!(
        "## Capture request\n\n\
         Content: {content}\n\
         Namespace: {namespace}\n\
         Memory kind: {memory_kind}\n\
         Source: {source_str}\n\n\
         ## Steps\n\n\
         1. Call `sm_add_fact` with content=\"{content}\", namespace=\"{namespace}\", \
         memory_kind=\"{memory_kind}\"{source_arg}.\n\
         2. If the fact relates to existing facts, call `sm_add_graph_edge` to link them.\n\
         3. If evidence confidence matters, call `sm_set_provenance` with confidence and \
         support_count.\n\
         4. Use `sm_search_witnessed` first to check if a similar fact already exists — \
         if so, use `sm_supersede_fact` instead of creating a duplicate.\n\n\
         ## Idempotency\n\n\
         If this is a retry, pass the same `idempotency_key` to avoid duplicates.",
        source_str = source.map(|s| s.as_str()).unwrap_or("(none)"),
        source_arg = source
            .map(|s| format!(", source=\"{s}\""))
            .unwrap_or_default(),
    );

    Ok(GetPromptResult::new(vec![
            PromptMessage::new(PromptMessageRole::User, PromptMessageContent::text(system)),
            PromptMessage::new(PromptMessageRole::User, PromptMessageContent::text(user)),
        ]).with_description(format!("Capture fact in {namespace}")))
}

fn prompt_audit(args: &std::collections::HashMap<String, String>) -> Result<GetPromptResult, String> {
    let namespace = args.get("namespace");

    let system = "You are a semantic-memory audit assistant. Audit first (read-only), present a health report, then reconcile ONLY after the user approves. Never destructively delete without explicit sign-off.";

    let user = format!(
        "## Audit request\n\n\
         Namespace: {ns}\n\n\
         ## Phase 1 — Audit (read-only)\n\n\
         1. `sm_stats` — size, fact/chunk/document/edge counts, embedding model.\n\
         2. `sm_list_namespaces` then `sm_list_facts(namespace, ...)` — exhaustively \
         enumerate each namespace{ns_focus}. Scan for duplicates, contradictions, stale entries.\n\
         3. `sm_community` (resolution 1.0) — community structure + within-community \
         contradiction scan.\n\
         4. `sm_run_lifecycle` on representative item_ids — syndromes, subtraction \
         candidates, recompression need.\n\
         5. `sm_topology` — Betti numbers, structural voids, weakly-connected facts.\n\
         6. `sm_detect_contradictions` — content-based contradiction detection.\n\n\
         Present a concise **health report**: store size, suspected duplicates/contradictions, \
         stale/forgettable items, graph gaps. **Stop and ask for approval before changing \
         anything.**\n\n\
         ## Phase 2 — Reconcile (only after approval)\n\n\
         - Stale fact with replacement → `sm_supersede_fact` (keeps history, auto-filters old).\n\
         - Near-duplicate facts → `sm_supersede_fact` with merged content.\n\
         - Pure noise/error → `sm_delete_fact` (HARD, irreversible — use sparingly).\n\
         - Bad ingest → `sm_delete_namespace` (confirm contents first).\n\
         - Contradictions → supersede the losing side + `sm_invalidate_graph_edge`.\n\
         - Graph gaps → `sm_add_graph_edge` to connect related-but-unlinked facts.",
        ns = namespace.map(|s| s.as_str()).unwrap_or("(all)"),
        ns_focus = namespace
            .map(|n| format!(", focusing on '{n}'"))
            .unwrap_or_default(),
    );

    Ok(GetPromptResult::new(vec![
            PromptMessage::new(PromptMessageRole::User, PromptMessageContent::text(system)),
            PromptMessage::new(PromptMessageRole::User, PromptMessageContent::text(user)),
        ]).with_description(format!("Audit semantic memory{}", namespace.map(|n| format!(": {n}")).unwrap_or_default())))
}

fn prompt_explore(args: &std::collections::HashMap<String, String>) -> Result<GetPromptResult, String> {
    let topic = args.get("topic").ok_or("Missing required argument: topic")?;
    let mode = args.get("mode").map(|s| s.as_str()).unwrap_or("related");
    let second_topic = args.get("second_topic");

    let system = "You are a knowledge-graph exploration assistant. Go beyond flat search: traverse the typed graph to surface related, adjacent, and connecting knowledge.";

    let (mode_desc, steps) = match mode {
        "path" => {
            let second = second_topic.ok_or("Missing required argument: second_topic for path mode")?;
            ("Path: how are two concepts connected", format!(
                "1. `sm_search(\"{topic}\")` → resolve to a result_id.\n\
                 2. `sm_search(\"{second}\")` → resolve to a result_id.\n\
                 3. `sm_graph_path(from_id, to_id)` → shortest path with per-hop edge evidence.\n\
                 4. Read facts along the path with `sm_get_fact(id)`.\n\
                 5. Explain the chain in plain language — edge types, weights, and what they mean."
            ))
        }
        "structure" => ("Structure: clusters and communities", format!(
            "1. `sm_search(\"{topic}\")` → get result_ids to seed exploration.\n\
             2. `sm_community(resolution=1.0)` → community detection across the graph.\n\
             3. `sm_topology` → components, cycles, and structural voids.\n\
             4. Optionally `sm_factor_graph` with initial beliefs → propagated confidence.\n\
             5. Report: which community {topic} sits in, its members, and the overall graph structure."
        )),
        _ => ("Related: what's related to this topic", format!(
            "1. `sm_search(\"{topic}\")` → take the top result_ids (direct hits).\n\
             2. `sm_get_fact_neighbors(result_id)` → for each anchor, get the fact plus its \
             graph neighbors WITH their content in one call.\n\
             3. `sm_discord_search(direct_result_ids)` → second-order neighbors (related but \
             not direct hits). Hydrate any you want to discuss with `sm_get_fact`.\n\
             4. Optionally `sm_community` → which community {topic} sits in, and its members.\n\
             5. Synthesize: what's central, what's adjacent, and the relationships between them.\n\n\
             Report edge types/weights — they carry meaning (e.g. 'depends_on', weight 3.0 for hubs).\n\
             If the graph is sparse around the topic, say so and suggest running the audit prompt."
        )),
    };

    let user = format!(
        "## Graph exploration\n\n\
         Topic: {topic}\n\
         Mode: {mode_desc}\n\n\
         ## Steps\n\n{steps}\n\n\
         ## Tips\n\n\
         - Discord search is what makes this worth more than `sm_search` — always run it \
         for 'related to' questions.\n\
         - Report edge types and weights — they carry meaning.\n\
         - If the graph is sparse, suggest running the `sm-audit` prompt to fill gaps.",
    );

    Ok(GetPromptResult::new(vec![
            PromptMessage::new(PromptMessageRole::User, PromptMessageContent::text(system)),
            PromptMessage::new(PromptMessageRole::User, PromptMessageContent::text(user)),
        ]).with_description(format!("Explore graph: {topic}")))
}

fn prompt_maintenance(args: &std::collections::HashMap<String, String>) -> Result<GetPromptResult, String> {
    let action = args.get("action").map(|s| s.as_str()).unwrap_or("check");

    let (desc, steps) = match action {
        "rebuild-fts" => (
            "Rebuild FTS indexes",
            "1. `sm_reconcile(action=\"ReportOnly\")` — get a baseline integrity report.\n\
             2. `sm_reconcile(action=\"RebuildFts\")` — rebuild FTS indexes from source data.\n\
             3. `sm_reconcile(action=\"ReportOnly\")` — verify the fix.\n\
             4. Report: what was corrupted, what was rebuilt, current integrity status.",
        ),
        "vacuum" => (
            "Vacuum the SQLite database",
            "1. `sm_stats` — get current DB size.\n\
             2. `sm_vacuum` — compact and defragment the SQLite DB.\n\
             3. `sm_stats` — compare size before/after.\n\
             4. Report: space reclaimed, any issues.",
        ),
        "reembed" => (
            "Re-embed all facts",
            "1. `sm_embeddings_are_dirty` — check if re-embedding is actually needed.\n\
             2. If dirty: `sm_reembed_all` — re-embed every fact with the current model.\n\
             3. `sm_embeddings_are_dirty` — verify all embeddings are clean.\n\
             4. Report: facts re-embedded, time taken, any failures.\n\n\
             WARNING: This is expensive (~138ms/fact on CPU). Warn the user before running.",
        ),
        "dirty-check" => (
            "Check if embeddings are stale",
            "1. `sm_embeddings_are_dirty` — check if any facts lack embeddings or have stale vectors.\n\
             2. Report: dirty count, total count, recommendation.",
        ),
        _ => (
            "Integrity check (read-only)",
            "1. `sm_stats` — size, counts, embedding model, DB health.\n\
             2. `sm_reconcile(action=\"ReportOnly\")` — full integrity report.\n\
             3. `sm_embeddings_are_dirty` — check for stale embeddings.\n\
             4. `sm_list_namespaces` — namespace overview.\n\
             5. Report: overall health, any issues found, recommended actions.",
        ),
    };

    let system = "You are a semantic-memory maintenance assistant. Run read-only checks first, then mutations only with explicit user approval. Warn the user before expensive operations.";

    let user = format!(
        "## Maintenance request\n\n\
         Action: {desc}\n\n\
         ## Steps\n\n{steps}\n\n\
         ## Discipline\n\n\
         - Read-only checks are always safe.\n\
         - Mutations (rebuild, vacuum, reembed) require user approval.\n\
         - Warn before expensive operations (re-embedding can take minutes).\n\
         - Report exactly what changed and the before/after state.",
    );

    Ok(GetPromptResult::new(vec![
            PromptMessage::new(PromptMessageRole::User, PromptMessageContent::text(system)),
            PromptMessage::new(PromptMessageRole::User, PromptMessageContent::text(user)),
        ]).with_description(format!("Maintenance: {desc}")))
}

fn prompt_supersede(args: &std::collections::HashMap<String, String>) -> Result<GetPromptResult, String> {
    let old_fact_id = args.get("old_fact_id").ok_or("Missing required argument: old_fact_id")?;
    let new_content = args.get("new_content").ok_or("Missing required argument: new_content")?;
    let reason = args.get("reason").map(|s| s.as_str()).unwrap_or("corrected content");

    let system = "You are a semantic-memory supersession assistant. Replace stale facts with corrected versions while preserving history. The old fact is linked via a 'supersedes' edge and auto-filtered from search results.";

    let user = format!(
        "## Supersession request\n\n\
         Old fact ID: {old_fact_id}\n\
         New content: {new_content}\n\
         Reason: {reason}\n\n\
         ## Steps\n\n\
         1. `sm_get_fact(\"{old_fact_id}\")` — verify the old fact exists and read its current content.\n\
         2. `sm_supersede_fact(old_fact_id=\"{old_fact_id}\", content=\"{new_content}\", reason=\"{reason}\")`.\n\
         3. Verify: `sm_search` for the topic — the old fact should be filtered, the new one visible.\n\
         4. If the old fact had graph edges, check whether they should be re-created for the new fact.\n\
         5. Report: what was superseded, the new fact ID, and confirmation that search filters the old one.",
    );

    Ok(GetPromptResult::new(vec![
            PromptMessage::new(PromptMessageRole::User, PromptMessageContent::text(system)),
            PromptMessage::new(PromptMessageRole::User, PromptMessageContent::text(user)),
        ]).with_description(format!("Supersede fact {old_fact_id}")))
}

// ─── Embedded resource content ──────────────────────────────────────────

const RESOURCE_STANDARD_WORKFLOW: &str = r#"# Standard Semantic-Memory Workflow

Treat stored memory as recall, not authority. Current files, tests, connected
systems, and primary sources outrank recalled content.

## Standard workflow

1. For relevant prior context, call `sm_search_witnessed`. Prefer a focused
   query and namespace filters. Use raw search only when the active operator
   profile deliberately exposes it.
2. Inspect the result's source, trust state, state, and receipt reference.
   Hydrate a fact with `sm_get_fact` before relying on details when available.
3. Verify material or current claims against the workspace or primary source.
4. Store only durable, source-attributed facts. Keep one fact per call and use a
   useful project/domain namespace.
5. Correct stale facts with `sm_supersede_fact`; do not delete history merely
   because it became old.
6. Use `sm_set_provenance` when evidence confidence and support count matter.
7. Add graph edges only for relationships you can explain and source.

## Replay and authority

- Witnessed search defaults to `replay_mode: no_replay`. Use `store_inputs`
  only when retaining the full query and filters is acceptable, then replay via
  `sm_replay_search` and its receipt ID.
- Recall authority does not authorize an assertion or action. Before presenting
  recalled content as an authorized assertion, use
  `sm_decide_assertion_authority`. Before acting on recalled content, use
  `sm_decide_action_authority`.
- Authority decision tools return content-free receipts. A positive decision is
  permission for that purpose, not proof that the fact is true.

## Trust states

Interpret `supported`, `partially_supported`, `unsupported`, `contradicted`,
`heuristic_only`, and `persisted_unjudged` as claim-ledger enrichment. If trust
enrichment is disabled after ledger verification failure, semantic recall can
still work, but do not infer claim support.

## Safety

- Never store passwords, tokens, private keys, credential-bearing logs, or
  sensitive personal data without an explicit need and user approval.
- Do not opt into replay-input retention silently.
- Ask before using destructive/admin tools. The `full` profile can permanently
  delete, import, rebuild, compact, vacuum, or otherwise mutate operator state.
- If an expected tool is absent, trust MCP `tools/list`: tool visibility depends
  on the binary's build and runtime profile.
"#;

const RESOURCE_AUTHORITY_MODEL: &str = r#"# Governed Authority Model

The semantic-memory MCP server implements a three-purpose authority model.
Each purpose is independently governed — authority for one never implies
authority for another.

## Three purposes

1. **Recall** — searching and reading stored memory. `sm_search_witnessed`
   returns content with receipt provenance. This is the default; all read
   operations are recall.

2. **Assertion** — presenting recalled content as an established fact. Before
   doing this, call `sm_decide_assertion_authority` with the fact_id, caller,
   subject, audiences, and scope. The tool returns a content-free receipt
   that either permits or denies the assertion for that purpose.

3. **Action** — acting on recalled content (e.g., executing a stored procedure,
   applying a configuration, making a change). Before acting, call
   `sm_decide_action_authority` with the same parameter shape. The tool returns
   a content-free receipt that either permits or denies the action.

## Key rules

- A positive recall does NOT authorize assertion or action.
- Authority decision tools return **content-free** receipts. They never return
  the fact's text. A positive decision is permission for that purpose, not proof
  that the fact is true.
- Delegation and elevation leases can grant cross-principal authority for
  specific purposes, scopes, and time windows.
- The scope is namespace-exact: `namespace`, plus optional `domain`,
  `workspace_id`, and `repo_id`.

## Parameter shape

All authority decision tools take:
- `fact_id`: the fact being evaluated
- `caller`: the principal requesting the decision
- `subject`: the subject the decision is about
- `audiences`: who the result is intended for
- `scope`: namespace/resource scope (NamespaceScopeV1)
- `delegation_or_elevation`: optional existing lease contract

## When to use

- Anytime you want to cite a recalled fact as established truth → assertion authority.
- Anytime you want to take an action based on recalled content → action authority.
- For autonomous/agent loops that need to decide whether stored memory
  authorizes a step → both assertion and action authority.
"#;

const RESOURCE_RETRIEVAL_STRATEGY: &str = r#"# Semantic-Memory Retrieval Strategy

Primary strategy for source-first recall, adaptive routing, graph hydration,
contradiction checks, and evidence-backed retrospectives.

## Source-first principle

Always search memory BEFORE asking the user for context. But always verify
recalled content against current sources — memory is recall, not authority.

## Query workflow

1. **Start with `sm_search_witnessed`** — the mandatory witnessed retrieval.
   It bypasses cache, verifies durable receipt persistence, and defaults to
   Current state. Use focused queries and namespace filters.

2. **Inspect results** — check source, trust_state, state, and receipt_ref
   for each result. Hydrate key facts with `sm_get_fact(id)`.

3. **Check for contradictions** — call `sm_detect_contradictions` with the
   same query. If contradictions are found, use `sm_search_with_routing` which
   profiles the query and routes to appropriate retrieval stages.

4. **Hydrate the graph** — for multi-hop or relationship questions, use:
   - `sm_get_fact_neighbors(id)` — fact + graph neighbors in one call
   - `sm_discord_search(direct_result_ids)` — second-order related items
   - `sm_graph_path(from_id, to_id)` — shortest path between two concepts

5. **Verify** — for material claims, check against current workspace files,
   tests, connected systems, or primary sources. If memory conflicts with
   current evidence, current evidence wins.

## Adaptive routing

For complex queries, use `sm_search_with_routing` instead of `sm_search`:
- Profiles the query type (factual, relational, temporal, etc.)
- Routes to appropriate retrieval stages
- Applies factor-graph belief propagation
- Groups results by knowledge graph community

## Provenance and trust

- `sm_set_provenance` — set confidence score (0.0-1.0) and support count
- Trust states: supported, partially_supported, unsupported, contradicted,
  heuristic_only, persisted_unjudged
- If trust enrichment is disabled (ledger failure), semantic recall still
  works but don't infer claim support

## Capturing facts

- One fact per `sm_add_fact` call
- Use meaningful namespaces (projects, research, preferences)
- Include source attribution
- Set `memory_kind` appropriately (durable_fact, preference, correction, etc.)
- Use `idempotency_key` for retries to avoid duplicates
- Set sensitivity (public, internal, confidential, restricted)

## Supersession

- Use `sm_supersede_fact` to correct stale facts (NOT delete)
- Preserves history via 'supersedes' graph edge
- Search auto-filters superseded facts unless querying for history
"#;

const RESOURCE_MEMORY_CURATOR: &str = r#"# Memory Curator

Audit and improve the semantic memory store. Audit first, present a report,
then reconcile **only after the user approves** — always by append/supersede,
never hard deletion without sign-off.

## Phase 1 — Audit (read-only)

1. `sm_stats` — size, fact/chunk/document/edge counts, embedding model.
2. `sm_list_namespaces` then `sm_list_facts(namespace, ...)` — exhaustively
   enumerate each namespace. Scan for duplicates, contradictions, stale entries.
   Use `sm_get_fact(id)` to inspect any candidate in full.
3. `sm_community` (resolution 1.0) — community structure + within-community
   contradiction scan.
4. `sm_run_lifecycle` on representative item_ids — syndromes, subtraction
   candidates, recompression need, quantization assessment.
5. `sm_topology` — Betti numbers, structural voids, weakly-connected facts.
6. `sm_detect_contradictions` — content-based contradiction detection.

Present a concise **health report**: store size, suspected duplicates/
contradictions, stale/forgettable items, graph gaps. **Stop and ask for
approval before changing anything.**

## Phase 2 — Reconcile (only after approval)

- **Stale fact with replacement** → `sm_supersede_fact`: writes corrected fact,
  links it, marks old as superseded. DEFAULT for "outdated, here's current truth."
- **Near-duplicate facts** → `sm_supersede_fact` with merged content.
- **Pure noise/error** → `sm_delete_fact` (HARD, irreversible — use sparingly).
- **Bad ingest** → `sm_delete_namespace` (confirm contents first).
- **Contradictions** → supersede the losing side + `sm_invalidate_graph_edge`.
- **Graph gaps** → `sm_add_graph_edge` to connect related-but-unlinked facts.

## Guardrails

- Never delete; the store evolves by append, supersession, and edge invalidation.
- Never let memory outrank current artifacts.
- Batch related changes and keep reasons in the receipts.
- Prefer `sm_supersede_fact` over hard delete.
"#;

const RESOURCE_MEMORY_MAINTENANCE: &str = r#"# Memory Maintenance

Store-level maintenance for semantic-memory: integrity check, FTS rebuild,
vacuum, re-embedding, and index repair.

## When to use

- "check memory health" / "is the DB ok"
- "compact" / "vacuum" / "rebuild indexes"
- "re-embed" / "embeddings are dirty"
- After changing the embedding model
- After large bulk ingestions

## Tools

- **sm_reconcile** — reconcile integrity. Actions: ReportOnly (check),
  RebuildFts (rebuild FTS indexes), ReEmbed (after model change).
- **sm_vacuum** — compact the SQLite database. Safe to run anytime.
- **sm_reembed_all** — re-embed every fact. Expensive (~138ms/fact on CPU).
  Warn the user before running.
- **sm_embeddings_are_dirty** — check if embeddings are stale (read-only).
  Run BEFORE re-embedding to avoid unnecessary work.
- **sm_get_search_receipt** / **sm_replay_search_receipt** — audit search
  receipts for reproducibility.
- **sm_query_claim_versions** / **sm_query_relation_versions** /
  **sm_query_episodes** / **sm_query_entity_aliases** /
  **sm_query_evidence_refs** — bitemporal projection queries.
- **sm_import_envelope** / **sm_import_status** / **sm_list_imports** —
  bulk import system for typed data with provenance.

## Workflow

1. **Check**: `sm_stats` + `sm_reconcile(action="ReportOnly")` +
   `sm_embeddings_are_dirty`
2. **Diagnose**: identify FTS corruption, stale embeddings, DB bloat, or
   orphaned processes
3. **Fix** (with approval): `sm_reconcile(action="RebuildFts")` for FTS,
   `sm_vacuum` for bloat, `sm_reembed_all` for stale embeddings
4. **Verify**: re-run the check to confirm the fix

## Pitfalls

- Maintenance ops ARE now exposed via MCP (sm_reconcile, sm_vacuum,
  sm_reembed_all, sm_embeddings_are_dirty).
- `sm_run_lifecycle` flags subtraction candidates; supersede and consolidate
  work for most cases. For DB compaction after large deletions, use sm_vacuum.
- For re-embedding after model change, use sm_reembed_all (check
  sm_embeddings_are_dirty first).
"#;

const RESOURCE_KNOWLEDGE_GRAPH_EXPLORER: &str = r#"# Knowledge Graph Explorer

Go beyond a flat search: traverse the typed graph to surface related, adjacent,
and connecting knowledge that a single query misses.

## Choose the question type

**"What's related to X / what do I know about X?"**
1. `sm_search(X)` -> take the top `result_id`s (the direct hits).
2. `sm_get_fact_neighbors(result_id)` -> for each anchor, get the fact **plus its
   graph neighbors WITH their content** in one call.
3. `sm_discord_search(direct_result_ids)` -> second-order neighbors (related
   but not direct hits). Hydrate any you want to discuss with `sm_get_fact`.
4. Optionally `sm_community` -> which community X sits in, and its members.
5. Synthesize: what's central, what's adjacent, and the relationships between them.

**"How are X and Y connected?"**
1. `sm_search(X)` and `sm_search(Y)` -> resolve each to a `result_id`.
2. `sm_graph_path(from_id, to_id)` -> shortest path with per-hop edge evidence
   (relation, weight). Read any along the way with `sm_get_fact`.
3. Explain the chain in plain language.

**"Show me the structure / clusters."**
1. `sm_community(resolution)` -> communities and members.
2. `sm_topology` -> components, cycles, and gaps.
3. Optionally `sm_factor_graph` with initial beliefs -> propagated confidence.

## Tips

- Discord search is what makes this worth more than `sm_search` — always run it
  for "related to" questions.
- Report edge **types/weights** (e.g. `depends_on`, weight 3.0 for hubs) — they
  carry meaning.
- If the graph is sparse around the topic, say so and suggest running the
  memory-curator to fill gaps.
"#;

const RESOURCE_MEMORY_KEEPER: &str = r#"# Memory-Keeper: Delegated Memory Operations

Delegate heavy memory operations to a focused subagent that returns a structured
summary without flooding the main agent's context.

## Available operations

### Namespace audit
Delegate to audit a namespace: run sm_stats, list facts (up to 200), cluster by
topic, identify stale/duplicate/orphaned candidates. Returns JSON with
total_facts, topic_breakdown, stale_candidates, duplicate_groups.

### Contradiction sweep
Delegate to detect contradictions: run sm_detect_contradictions with 5+ distinct
domain-spanning queries. Returns JSON with queries_run, contradictions_found
(id_a, id_b, reason).

### Bulk ingest with dedup
Delegate to ingest a codebase: check if already in KB, extract metadata, add
missing facts only (with source attribution), link via entity edges, set
provenance. Returns JSON with facts_existing, facts_added, edges_created.

### Edge population check
Delegate to check graph connectivity: list facts, check edges for each, add
missing edges for orphaned facts. Returns JSON with total_facts,
facts_with_edges, edges_added.

## Constraints

- The subagent gets a FRESH terminal session
- It cannot call delegate_task (leaf agent)
- All results are self-reported — verify critical numbers on return
- Timeout: 120 seconds for most tasks
- Always include 'Return JSON:' in the context for parseable structured data
"#;

const RESOURCE_POOLED_MEMORY_OPERATIONS: &str = r#"# Pooled Memory Operations

Multi-device memory mesh: identity/provenance, typed synchronization, per-device
SQLite primaries and server replicas, sparse routed retrieval.

## Architecture

`pooled-memory` is a standalone crate that adds multi-device identity, actor
provenance, idempotent operation envelopes, replica synchronization, and sparse
cross-device retrieval on top of `semantic-memory`.

**Key design principle**: `pooled-memory` is a SEPARATE repo from
`semantic-memory`. Devices keep an independently usable local `semantic-memory`
primary; the pooled layer is opt-in.

## MCP tools

- **sm_list_devices** — list registered devices in the mesh
- **sm_register_device** — register a new device (label, platform, hostname)
- **sm_register_actor** — register an actor on a device
- **sm_submit_operation** — submit an idempotent operation envelope
- **sm_get_operation** — retrieve a persisted operation by ID
- **sm_heartbeat** — device heartbeat for liveness tracking
- **sm_search_witnessed** (pooled) — sparse cross-device retrieval
- **sm_health** — service health check
- **sm_stats** — registry and semantic memory statistics
- **sm_verify_integrity** — run integrity checks across the mesh

## Design principles

- Device-owned semantic primaries with separate server replicas
- Authoritative replay, apply ledger, stream head, and durable ACK belong in the
  replica semantic database transaction
- pooled.db holds control-plane observations only
- Never flattened shadow truth — devices own their primaries
"#;