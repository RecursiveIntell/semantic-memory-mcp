//! semantic-memory-mcp — MCP server for semantic-memory.
//!
//! Library target for integration tests. The main binary entry point
//! is in `main.rs`; this module re-exports the public modules so
//! integration tests can access bridge and http_server.

pub mod bridge;
pub mod http_server;
pub mod server;
mod tools;