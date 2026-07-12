"""Model-visible schemas for the semantic-memory-mcp Hermes plugin."""

INSTALL = {
    "name": "semantic_memory_mcp_install",
    "description": (
        "Install or replace a local semantic-memory MCP entry by invoking "
        "`hermes mcp add`. This changes Hermes MCP configuration."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_dir": {
                "type": "string",
                "description": "Memory store directory. Defaults below the current user's home directory.",
            },
            "server_name": {
                "type": "string",
                "description": "Hermes MCP entry name (default: semantic_memory).",
            },
            "binary": {
                "type": "string",
                "description": "Executable name or absolute path (default: semantic-memory-mcp).",
            },
            "tool_profile": {
                "type": "string",
                "enum": ["lean", "standard", "agent", "full"],
                "description": "Runtime MCP tool profile (default: agent).",
            },
        },
        "additionalProperties": False,
    },
}

LIST = {
    "name": "semantic_memory_mcp_list",
    "description": "Invoke `hermes mcp list` and return the configured MCP server list.",
    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
}

TEST = {
    "name": "semantic_memory_mcp_test",
    "description": (
        "Invoke `hermes mcp test` for the semantic-memory entry. The command "
        "starts the server, negotiates MCP, and discovers its current tools."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "server_name": {
                "type": "string",
                "description": "Hermes MCP entry name (default: semantic_memory).",
            }
        },
        "additionalProperties": False,
    },
}

CONFIGURE = {
    "name": "semantic_memory_mcp_configure",
    "description": (
        "Launch `hermes mcp configure` for interactive tool selection when a "
        "TTY is available, or return the exact terminal command otherwise."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "server_name": {
                "type": "string",
                "description": "Hermes MCP entry name (default: semantic_memory).",
            },
            "interactive": {
                "type": "boolean",
                "description": "Actually launch the interactive selector (default: false).",
            },
        },
        "additionalProperties": False,
    },
}
