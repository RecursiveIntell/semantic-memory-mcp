# Claude Code plugin

The plugin bundles a Claude Agent Skill, two commands, and a stdio MCP launcher.
The launcher resolves `semantic-memory-mcp` from `PATH` unless overridden and
defaults the store below the current user's home directory.

Test from this repository:

```bash
chmod +x integrations/claude-plugin/bin/semantic-memory-server
SEMANTIC_MEMORY_MCP_BIN="$(command -v semantic-memory-mcp)" \
SEMANTIC_MEMORY_DIR="$HOME/.local/share/semantic-memory" \
SEMANTIC_MEMORY_TOOL_PROFILE=agent \
claude --plugin-dir ./integrations/claude-plugin
```

Inside Claude Code:

```text
/mcp
/semantic-memory:semantic-memory-status
/semantic-memory:remember The durable fact and its source
```

Development checks:

```bash
claude --debug --plugin-dir ./integrations/claude-plugin
claude plugin validate ./integrations/claude-plugin
```

If the installed Claude Code version does not provide the CLI validator, use
`--debug` plus the repository validator in `../tests/validate_integrations.py`.
Run `/reload-plugins` after changing the manifest, MCP config, or commands.

Environment variables:

| Variable | Default |
| --- | --- |
| `SEMANTIC_MEMORY_MCP_BIN` | `semantic-memory-mcp` from `PATH` |
| `SEMANTIC_MEMORY_DIR` | `$HOME/.local/share/semantic-memory` |
| `SEMANTIC_MEMORY_TOOL_PROFILE` | `agent` |

Use `lean` for an autonomous read-only surface. Treat `full` as an operator
profile with destructive and maintenance tools.
