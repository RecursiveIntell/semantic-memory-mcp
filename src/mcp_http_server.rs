//! Authenticated, loopback-only Streamable HTTP MCP transport.
//!
//! This transport is deliberately separate from the auxiliary HTTP API. It
//! exposes the existing rmcp tool server at `/mcp` and is intended to be owned
//! by one warm semantic-memory process.

use axum::{
    body::Bytes,
    extract::State,
    http::{header, HeaderMap, Request, StatusCode},
    response::Response,
    routing::post,
    Router,
};
use http_body_util::{BodyExt, Full};
use rmcp::transport::streamable_http_server::{
    session::local::LocalSessionManager, StreamableHttpServerConfig, StreamableHttpService,
};
use std::{
    io,
    sync::{Arc, Mutex},
    thread,
};
use tokio::sync::oneshot;

use crate::{bridge::MemoryBridge, profile::ToolProfile, server::SemanticMemoryServer};

#[derive(Clone)]
struct AppState {
    service: StreamableHttpService<SemanticMemoryServer, LocalSessionManager>,
    token: Arc<str>,
}

async fn mcp_handler(State(state): State<AppState>, headers: HeaderMap, body: Bytes) -> Response {
    let supplied = headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok());
    let expected = format!("Bearer {}", state.token);
    if supplied != Some(expected.as_str()) {
        return StatusCode::UNAUTHORIZED.into_response();
    }

    let mut builder = Request::builder().method("POST").uri("/mcp");
    for (name, value) in &headers {
        builder = builder.header(name, value);
    }
    let request = match builder
        .header(header::ACCEPT, "application/json")
        .body(Full::new(body))
    {
        Ok(request) => request,
        Err(_) => return StatusCode::BAD_REQUEST.into_response(),
    };

    let response = state.service.handle(request).await;
    let status = response.status();
    let response_headers = response.headers().clone();
    let response_body = match response.into_body().collect().await {
        Ok(body) => body.to_bytes(),
        Err(_) => return StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    };
    let mut output = Response::new(axum::body::Body::from(response_body));
    *output.status_mut() =
        StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    for (name, value) in &response_headers {
        output.headers_mut().insert(name, value.clone());
    }
    output
}

/// Ownership handle for the server thread. Call [`Self::shutdown`] in tests or
/// controlled shutdown paths; the live process retains this guard for its life.
pub struct McpHttpServer {
    shutdown: Mutex<Option<oneshot::Sender<()>>>,
    thread: Mutex<Option<thread::JoinHandle<()>>>,
}

impl McpHttpServer {
    /// Signal graceful shutdown and wait until the TCP listener exits.
    pub fn shutdown(&self) -> Result<(), String> {
        let sender = self
            .shutdown
            .lock()
            .map_err(|_| "MCP HTTP shutdown lock poisoned".to_string())?
            .take()
            .ok_or_else(|| "MCP HTTP server is already shut down".to_string())?;
        let _ = sender.send(());
        let thread = self
            .thread
            .lock()
            .map_err(|_| "MCP HTTP thread lock poisoned".to_string())?
            .take()
            .ok_or_else(|| "MCP HTTP server thread is unavailable".to_string())?;
        thread
            .join()
            .map_err(|_| "MCP HTTP server thread panicked during shutdown".to_string())
    }
}

/// Start authenticated Streamable HTTP MCP at `http://127.0.0.1:<port>/mcp`.
///
/// The server owns its runtime rather than borrowing a caller runtime. That
/// keeps the listener valid for the complete warm-process lifetime and makes
/// test shutdown deterministic.
pub fn start_mcp_http_server(
    port: u16,
    token: &str,
    bridge: MemoryBridge,
    profile: ToolProfile,
) -> io::Result<McpHttpServer> {
    let listener = std::net::TcpListener::bind(("127.0.0.1", port))?;
    listener.set_nonblocking(true)?;
    let address = listener.local_addr()?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .worker_threads(2)
        .build()
        .map_err(io::Error::other)?;
    let token: Arc<str> = Arc::from(token.to_owned());
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let thread = thread::spawn(move || {
        let service = StreamableHttpService::new(
            move || Ok(SemanticMemoryServer::from_profile(bridge.clone(), profile)),
            Arc::new(LocalSessionManager::default()),
            {
                let mut config = StreamableHttpServerConfig::default();
                // The stdio relay is intentionally stateless. It forwards each
                // JSON-RPC request independently and does not create durable
                // transport sessions inside the warm store owner.
                config.stateful_mode = false;
                config.json_response = true;
                config
            },
        );
        let app = Router::new()
            .route("/mcp", post(mcp_handler))
            .with_state(AppState { service, token });

        runtime.block_on(async move {
            eprintln!("MCP Streamable HTTP server listening on {address}");
            let listener = match tokio::net::TcpListener::from_std(listener) {
                Ok(listener) => listener,
                Err(error) => {
                    eprintln!("MCP Streamable HTTP listener setup failed: {error}");
                    return;
                }
            };
            let server = axum::serve(listener, app).with_graceful_shutdown(async move {
                let _ = shutdown_rx.await;
            });
            if let Err(error) = server.await {
                eprintln!("MCP Streamable HTTP server stopped with error: {error}");
            }
        });
    });

    Ok(McpHttpServer {
        shutdown: Mutex::new(Some(shutdown_tx)),
        thread: Mutex::new(Some(thread)),
    })
}

trait IntoResponseExt {
    fn into_response(self) -> Response;
}

impl IntoResponseExt for StatusCode {
    fn into_response(self) -> Response {
        Response::builder()
            .status(self)
            .body(axum::body::Body::empty())
            .expect("valid empty response")
    }
}
