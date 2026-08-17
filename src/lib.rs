//! semantic-memory-mcp — MCP server for semantic-memory.
//!
//! Library target for integration tests. The main binary entry point
//! is in `main.rs`; this module re-exports the public modules so
//! integration tests can access bridge and http_server.

pub mod bridge;
#[cfg(feature = "search")]
pub mod http_server;
#[cfg(not(feature = "search"))]
#[path = "http_server_stable.rs"]
pub mod http_server;
pub mod mcp_http_server;
pub mod profile;
#[cfg(feature = "search")]
pub mod server;
#[cfg(not(feature = "search"))]
#[path = "server_stable.rs"]
pub mod server;
pub mod skills;
#[cfg(feature = "search")]
mod tools;
#[cfg(not(feature = "search"))]
#[path = "tools_stable.rs"]
mod tools;
