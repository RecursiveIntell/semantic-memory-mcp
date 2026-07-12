# Codex integration

Codex needs two independent assets: a configured stdio MCP server supplies live
tools, while the Agent Skill teaches the model when and how to use them.

## Install the MCP server

The CLI route safely expands the current user's home directory before Codex
writes its configuration:

```bash
codex mcp add semantic_memory -- \
  semantic-memory-mcp \
  --memory-dir "$HOME/.local/share/semantic-memory" \
  --tool-profile agent
codex mcp list
```

Run `codex mcp --help` for the commands supported by the installed Codex build.
Alternatively, copy the table from `config.example.toml` into
`~/.codex/config.toml` and replace `/absolute/path/to/semantic-memory`.

Codex user config and trusted project `.codex/config.toml` layers use the same
`[mcp_servers.<id>]` shape. A project config is ignored when the repository is
not trusted.

## Install the skill

For repository-scoped discovery, copy the skill directory to the repository
root:

```bash
mkdir -p .agents/skills
cp -R integrations/codex/.agents/skills/semantic-memory .agents/skills/
```

The supplied integration already has this layout, so launching Codex from
`integrations/codex` or below also discovers it. Codex scans `.agents/skills`
from the working directory toward the repository root. A skill directory must
contain `SKILL.md` with `name` and `description` frontmatter.

## Smoke test

1. Restart Codex after changing global MCP config.
2. Confirm `semantic_memory` appears in the MCP server list.
3. Invoke `$semantic-memory` and ask for a read-only status/recall check.
4. Confirm the visible tools match the configured runtime profile. MCP
   `tools/list` is authoritative.

Use `agent` for trusted coding workflows that need durable capture. Use `lean`
for four governed read-only tools. The `full` profile is an operator surface.

Current format references:

- [Codex MCP](https://developers.openai.com/codex/mcp)
- [Codex skills](https://developers.openai.com/codex/skills)
- [Codex config](https://developers.openai.com/codex/config-reference)
