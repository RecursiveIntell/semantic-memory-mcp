//! semantic-memory-mcp — MCP server for semantic-memory.
//!
//! Library target for integration tests. The main binary entry point
//! is in `main.rs`; this module re-exports the public modules so
//! integration tests can access bridge and http_server.

pub mod bridge;
#[cfg(not(all(feature = "stable", not(feature = "full"))))]
pub mod http_server;
#[cfg(all(feature = "stable", not(feature = "full")))]
#[path = "http_server_stable.rs"]
pub mod http_server;
pub mod profile;
#[cfg(not(all(feature = "stable", not(feature = "full"))))]
pub mod server;
#[cfg(all(feature = "stable", not(feature = "full")))]
#[path = "server_stable.rs"]
pub mod server;
#[cfg(not(all(feature = "stable", not(feature = "full"))))]
mod tools;
#[cfg(all(feature = "stable", not(feature = "full")))]
#[path = "tools_stable.rs"]
mod tools;
