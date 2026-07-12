//! HTTP is intentionally absent from the compile-time stable artifact.

use crate::bridge::MemoryBridge;
use tokio::runtime::Handle;

#[cold]
pub fn start_http_server(_port: u16, _auth_token: &str, _bridge: MemoryBridge, _handle: Handle) {
    panic!("HTTP transport is unavailable in the compile-time stable build; use stdio MCP or rebuild with --features full")
}
