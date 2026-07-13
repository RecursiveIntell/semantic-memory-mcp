---
name: semantic-memory
description: Use semantic-memory-mcp for persistent local recall, source-aware fact capture, supersession, provenance, witnessed replay, and governed assertion/action decisions during Codex work. Treat recalled memory as context, not authority.
---

# Semantic memory workflow

Use semantic memory when prior project decisions, durable facts, user
preferences, or cross-session context may help.

1. Search with `sm_search_witnessed` using a focused query and namespace when
   known. The active MCP `tools/list` is authoritative; do not assume operator
   tools are present.
2. Read source, trust, state, and receipt fields. Hydrate fact IDs when the
   active profile exposes `sm_get_fact`.
3. Verify material claims against current repository files, tests, connected
   systems, or primary sources. If memory conflicts with current evidence,
   prefer current evidence.
4. Persist one durable, source-attributed fact per `sm_add_fact` call. Do not
   store transient logs, generated build output, guesses, secrets, tokens, or
   private keys.
5. Replace verified stale facts with `sm_supersede_fact` rather than deletion or
   a competing unmarked fact.
6. Set provenance and add graph edges only when the evidence and relationship
   are understood.

Witnessed search defaults to privacy-preserving `no_replay`. Select
`store_inputs` only with a reason to retain query/filter inputs; use the receipt
ID with `sm_replay_search`.

Recall permission is not assertion or action permission. Use
`sm_decide_assertion_authority` before an authorized assertion and
`sm_decide_action_authority` before acting on recalled content. These decisions
are content-free permission receipts, not truth proofs.

Trust enrichment has six quality states: `supported`, `partially_supported`,
`unsupported`, `contradicted`, `heuristic_only`, and `persisted_unjudged`.
Ledger corruption disables trust enrichment while ordinary semantic recall may
continue. Never promote disabled or missing enrichment into evidence.

Ask before destructive or broad operator actions. The `full` profile can expose
deletion, import, compaction, reconciliation, re-embedding, vacuum, and other
maintenance tools. Prefer `agent` for trusted coding work and `lean` for
autonomous read-only recall.
