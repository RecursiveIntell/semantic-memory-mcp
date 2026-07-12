# Agent integration packages

These packages connect the same stdio MCP server to three agent ecosystems.
They do not fork or wrap the memory API: the running server's MCP `tools/list`
remains authoritative for the selected build and runtime profile.

| Client | Integration asset | Install | Connection test | Skill/command test |
| --- | --- | --- | --- | --- |
| Hermes Agent | `hermes/` general plugin | Copy to `$HOME/.hermes/plugins/semantic-memory-mcp`, enable it, then use its install tool or `hermes mcp add` | `hermes mcp list`; `hermes mcp test semantic_memory` | Confirm setup tools load; run `hermes mcp configure semantic_memory` in a terminal |
| Claude Code | `claude-plugin/` | `claude --plugin-dir ./integrations/claude-plugin` | `/mcp` or `claude --debug --plugin-dir ...` | `/semantic-memory:semantic-memory-status` |
| Codex | `codex/.agents/skills/semantic-memory/` plus MCP config | `codex mcp add semantic_memory -- semantic-memory-mcp ...` | `codex mcp list` | Invoke `$semantic-memory` from a directory where `.agents/skills` is discoverable |

## Recommended profiles

| Profile | Visible intent | Recommended use |
| --- | --- | --- |
| `lean` / `standard` | Four governed read-only tools: witnessed search, stored replay, assertion decision, action decision | Autonomous or least-privilege recall |
| `agent` | Sixteen bounded recall/capture/provenance/graph tools | Trusted coding agents |
| `full` | Every compiled router tool, including operator/admin surfaces | Interactive operators with explicit approvals |

Do not infer the full profile's count from this repository: compile-time
features alter the router. Query MCP `tools/list` after every build/profile
change.

## Shared prerequisites

Build or install `semantic-memory-mcp` and make the binary available on `PATH`.
Choose one store directory and one embedding configuration for all clients that
need to share memory. A default Candle run may download its model from Hugging
Face; Ollama mode sends text to the configured Ollama endpoint.

The examples use `$HOME/.local/share/semantic-memory` in shell commands, where
the shell expands it. Checked-in JSON and TOML contain no machine-specific home
path. Never put credentials in facts, source fields, command arguments, or
plugin files.

## Validate the packages

From `semantic-memory-mcp`:

```bash
python3 integrations/tests/validate_integrations.py
python3 -m py_compile \
  integrations/hermes/__init__.py \
  integrations/hermes/schemas.py \
  integrations/hermes/tools.py
```

The validator checks JSON, TOML, YAML (with PyYAML), Python compilation, skill
frontmatter, the Claude launcher, path placeholders, and package invariants. It
does not mutate agent configuration.
