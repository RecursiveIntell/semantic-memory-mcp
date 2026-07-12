---
description: Check the semantic-memory MCP connection, active surface, and store statistics
disable-model-invocation: true
---

Inspect `/mcp` and the currently available `sm_*` tools. If `sm_stats` is
available, call it and summarize store counts, model/dimensions, and any health
warning. If the active profile does not expose `sm_stats`, explain that
`tools/list` is authoritative and run a narrowly scoped witnessed-search smoke
test only if it can be done without exposing sensitive content. Do not mutate
memory.
