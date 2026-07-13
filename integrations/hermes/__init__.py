"""Hermes plugin registration for semantic-memory-mcp setup helpers."""

from . import schemas, tools


def register(ctx):
    """Register MCP management helpers; the MCP server supplies all memory APIs."""
    ctx.register_tool(
        name="semantic_memory_mcp_install",
        toolset="semantic_memory_mcp_setup",
        schema=schemas.INSTALL,
        handler=tools.install,
        description="Install semantic-memory-mcp through the Hermes MCP CLI.",
    )
    ctx.register_tool(
        name="semantic_memory_mcp_list",
        toolset="semantic_memory_mcp_setup",
        schema=schemas.LIST,
        handler=tools.list_servers,
        description="List configured Hermes MCP servers.",
    )
    ctx.register_tool(
        name="semantic_memory_mcp_test",
        toolset="semantic_memory_mcp_setup",
        schema=schemas.TEST,
        handler=tools.test,
        description="Test and discover the semantic-memory MCP server.",
    )
    ctx.register_tool(
        name="semantic_memory_mcp_configure",
        toolset="semantic_memory_mcp_setup",
        schema=schemas.CONFIGURE,
        handler=tools.configure,
        description="Open or print the Hermes MCP tool selector command.",
    )
