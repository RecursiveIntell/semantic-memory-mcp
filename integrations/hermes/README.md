# Hermes plugin

This is a real Hermes general-plugin package. It registers guarded setup tools
that call the public Hermes MCP CLI; it does not duplicate any `sm_*` memory
API. Once installed, Hermes obtains the memory tools from MCP `tools/list`.

Install the directory in the Hermes user plugin location, then enable it:

```bash
cp -R integrations/hermes "$HOME/.hermes/plugins/semantic-memory-mcp"
hermes plugins enable semantic-memory-mcp
```

Restart Hermes. The plugin exposes:

- `semantic_memory_mcp_install` → `hermes mcp add ... --args ...`
- `semantic_memory_mcp_list` → `hermes mcp list`
- `semantic_memory_mcp_test` → `hermes mcp test <name>`
- `semantic_memory_mcp_configure` → `hermes mcp configure <name>`

Hermes' configure command is an interactive checklist. The configure tool
therefore prints the exact command by default; it only launches the checklist
when explicitly requested from a real TTY. This avoids hanging gateway/headless
agent sessions while preserving the supported Hermes configuration path.

For a direct install without the plugin:

```bash
hermes mcp add semantic_memory \
  --command semantic-memory-mcp \
  --args --memory-dir "$HOME/.local/share/semantic-memory" --tool-profile agent
hermes mcp list
hermes mcp test semantic_memory
hermes mcp configure semantic_memory
```

`--args` must be last because it consumes the remaining argv. Prefer `agent`
for a trusted coding agent and `lean` for read-only autonomous use. The `full`
profile is an operator surface.
