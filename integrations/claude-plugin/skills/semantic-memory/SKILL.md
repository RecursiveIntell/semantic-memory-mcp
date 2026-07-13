---
name: semantic-memory
description: Use the local semantic-memory MCP server for cross-session recall, durable verified facts, corrections, provenance, witnessed search, replay, and governed assertion/action decisions. Do not use it as authority for current workspace or external facts.
---

# Semantic memory

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
