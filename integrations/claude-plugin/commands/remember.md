---
description: Save one durable, source-attributed fact to semantic memory
argument-hint: <fact and source>
disable-model-invocation: true
---

Save the durable fact in "$ARGUMENTS" using `sm_add_fact`. Before writing:

1. Refuse secrets or credential-bearing content.
2. Ask for a namespace or source only when they cannot be inferred safely from
   the current project and the request.
3. Search for a likely current duplicate.
4. If this corrects a stale fact, use `sm_supersede_fact` instead of adding a
   competing current fact.
5. Report the returned fact and receipt identifiers.
