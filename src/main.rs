//! semantic-memory-mcp — MCP stdio server for semantic-memory.
//!
//! Wraps the semantic-memory knowledge management library as an MCP
// (Model Context Protocol) server. Works with Hermes Agent, Claude
//! Desktop, Cursor, and any MCP-compatible client.
//!
//! Usage:
//!   semantic-memory-mcp --memory-dir /path/to/memory-store
//!   semantic-memory-mcp --memory-dir /path --embedder ollama --embedding-url http://localhost:11434

use clap::Parser;
use rmcp::ServiceExt;
#[cfg(not(all(feature = "stable", not(feature = "full"))))]
use std::path::Path;
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;

use semantic_memory_mcp::bridge::{self, EmbedderBackend};
#[cfg(not(all(feature = "stable", not(feature = "full"))))]
use semantic_memory_mcp::http_server;
use semantic_memory_mcp::server;

/// semantic-memory MCP server configuration.
#[derive(Parser, Debug)]
#[command(name = "semantic-memory-mcp", version)]
#[command(about = "MCP server for semantic-memory — local-first knowledge management")]
struct Cli {
    /// Path to the memory store directory (created if it does not exist)
    #[arg(long)]
    memory_dir: String,

    /// Embedding backend: candle (default, in-process CPU, no Ollama needed),
    /// ollama (external Ollama server, GPU-accelerated), or mock (testing).
    ///
    /// Candle: pure-Rust, CPU-only, downloads model from HuggingFace.
    ///   Pros: zero external dependencies, works everywhere.
    ///   Cons: ~138ms per embedding on CPU.
    ///
    /// Ollama: external server with GPU support (ROCm/CUDA/Metal).
    ///   Pros: ~33ms per embedding (4x faster with GPU).
    ///   Cons: requires `ollama serve` running and model pulled.
    ///   Setup: `ollama pull nomic-embed-text` then `--embedder ollama`
    #[arg(long, default_value = "candle")]
    embedder: EmbedderBackend,

    /// Ollama embedding server URL (only used when --embedder ollama).
    /// Default: http://localhost:11434
    #[arg(long)]
    embedding_url: Option<String>,

    /// Embedding model name. For candle: HuggingFace model ID or alias
    /// (default: nomic-embed-text → nomic-ai/nomic-embed-text-v1.5).
    /// For ollama: the Ollama model name (default: nomic-embed-text).
    #[arg(long)]
    embedding_model: Option<String>,

    /// Embedding dimensions (default: 768)
    #[arg(long)]
    embedding_dims: Option<usize>,

    /// Optional HTTP port for warm-server access. When set, starts a
    /// minimal HTTP server alongside stdio MCP so hooks, benchmarks,
    /// and scripts can query the warm process without spawning new ones.
    /// Example: --http-port 1738
    #[arg(long)]
    http_port: Option<u16>,

    /// Authorization token required for HTTP server access.
    /// If --http-port is set, this token must be provided via the
    /// Authorization: Bearer header on all HTTP requests.
    #[arg(long)]
    http_auth_token: Option<String>,

    /// Read the HTTP authorization token from a private file instead of argv.
    /// The file must contain one non-empty token with no internal whitespace.
    #[arg(long)]
    http_auth_token_file: Option<PathBuf>,

    /// Run only the HTTP server (skip stdio MCP). Requires --http-port.
    /// Use this for standalone warm-server mode (benchmarks, hooks).
    #[arg(long)]
    http_only: bool,

    /// Enable TurboQuant compressed vector candidate backend.
    /// When enabled, embeddings are compressed using turbo-quant codecs for
    /// faster candidate generation, with exact f32 rerank for final results.
    /// Requires the `turbo-quant-codec` feature.
    #[arg(long)]
    turbo_quant: bool,

    /// TurboQuant polar angle bits (default: 8). Only used when --turbo-quant is set.
    #[arg(long)]
    turbo_quant_bits: Option<u8>,

    /// TurboQuant QJL projection count (default: 16). Only used when --turbo-quant is set.
    #[arg(long)]
    turbo_quant_projections: Option<usize>,

    /// Tool profile: lean/standard (3 governed read-only tools; lean is default),
    /// agent (15 bounded daily tools), or full (60 operator/admin tools).
    #[cfg_attr(
        all(feature = "stable", not(feature = "full")),
        arg(long, default_value = "stable")
    )]
    #[cfg_attr(
        not(all(feature = "stable", not(feature = "full"))),
        arg(long, default_value = "lean")
    )]
    tool_profile: String,
}

#[cfg(not(all(feature = "stable", not(feature = "full"))))]
fn normalize_http_auth_token(raw: &str, source: &str) -> anyhow::Result<String> {
    let token = raw.trim();
    if token.is_empty() {
        anyhow::bail!("HTTP authorization token from {source} is empty");
    }
    if token.chars().any(char::is_whitespace) {
        anyhow::bail!("HTTP authorization token from {source} contains whitespace");
    }
    Ok(token.to_string())
}

#[cfg(not(all(feature = "stable", not(feature = "full"))))]
fn resolve_http_auth_token(
    explicit: Option<&str>,
    token_file: Option<&Path>,
) -> anyhow::Result<Option<String>> {
    if let Some(token) = explicit {
        return normalize_http_auth_token(token, "--http-auth-token").map(Some);
    }
    if let Some(path) = token_file {
        let raw = std::fs::read_to_string(path)
            .map_err(|error| anyhow::anyhow!("failed to read {}: {error}", path.display()))?;
        return normalize_http_auth_token(&raw, &path.display().to_string()).map(Some);
    }
    Ok(None)
}

fn main() -> anyhow::Result<()> {
    // All logging goes to stderr — stdout is reserved for MCP JSON-RPC.
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();

    eprintln!("semantic-memory-mcp starting...");
    eprintln!("  memory_dir: {}", cli.memory_dir);
    eprintln!("  embedder: {:?}", cli.embedder);
    match cli.embedder {
        EmbedderBackend::Candle => {
            eprintln!(
                "  model: {} ({}d) — in-process Candle, CPU-only",
                cli.embedding_model.as_deref().unwrap_or("nomic-embed-text"),
                cli.embedding_dims.unwrap_or(768)
            );
        }
        EmbedderBackend::Ollama => {
            let url = cli
                .embedding_url
                .as_deref()
                .unwrap_or("http://localhost:11434");
            eprintln!(
                "  embedding: {} @ {} ({}d) — Ollama GPU-accelerated",
                cli.embedding_model.as_deref().unwrap_or("nomic-embed-text"),
                url,
                cli.embedding_dims.unwrap_or(768)
            );
            // Health check: verify Ollama is reachable before starting
            let model = cli.embedding_model.as_deref().unwrap_or("nomic-embed-text");
            if let Err(e) = reqwest::blocking::get(format!("{}/api/tags", url)) {
                eprintln!("  WARNING: Ollama not reachable at {} — {}", url, e);
                eprintln!("  Make sure Ollama is running: `ollama serve`");
                eprintln!("  And the model is pulled: `ollama pull {}`", model);
                eprintln!("  Falling back would require restart with --embedder candle");
                anyhow::bail!("Ollama not reachable at {}", url);
            } else {
                eprintln!("  Ollama health check: OK");
            }
        }
        EmbedderBackend::Mock => {
            eprintln!(
                "  mock embedder ({}d) — for testing only",
                cli.embedding_dims.unwrap_or(768)
            );
        }
    }

    // Open the memory store
    let bridge_config = bridge::BridgeConfig::from_args(
        &cli.memory_dir,
        cli.embedder,
        cli.embedding_url.as_deref(),
        cli.embedding_model.as_deref(),
        cli.embedding_dims,
        cli.turbo_quant,
        cli.turbo_quant_bits,
        cli.turbo_quant_projections,
    );

    let bridge = bridge::MemoryBridge::open(bridge_config)?;

    // Create the tokio runtime first (needed for both HTTP and stdio)
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .worker_threads(4)
        .build()?;

    // Create the MCP server
    let server = server::SemanticMemoryServer::new(bridge.clone(), &cli.tool_profile);

    // Start HTTP server if --http-port was specified.
    // When only HTTP is needed (no MCP client), use --http-only to skip stdio.
    #[cfg(all(feature = "stable", not(feature = "full")))]
    if cli.http_port.is_some() || cli.http_only {
        anyhow::bail!(
            "HTTP transport is unavailable in the compile-time stable build; use stdio MCP or rebuild with --features full"
        );
    }

    #[cfg(not(all(feature = "stable", not(feature = "full"))))]
    if let Some(port) = cli.http_port {
        let auth_token = resolve_http_auth_token(
            cli.http_auth_token.as_deref(),
            cli.http_auth_token_file.as_deref(),
        )?
        .ok_or_else(|| {
            anyhow::anyhow!("--http-port requires --http-auth-token or --http-auth-token-file. Refusing to start HTTP server without authorization.")
        })?;
        http_server::start_http_server(port, &auth_token, bridge, rt.handle().clone());
    }

    // If --http-only was set, skip stdio MCP and just keep the process alive
    // for the HTTP server.
    #[cfg(not(all(feature = "stable", not(feature = "full"))))]
    if cli.http_only {
        eprintln!("HTTP-only mode: stdio MCP disabled, serving HTTP requests.");
        // Park the main thread -- the HTTP server runs in its own thread
        rt.block_on(async {
            // Wait forever -- the HTTP server thread keeps the process alive
            std::future::pending::<()>().await;
        });
        return Ok(());
    }

    // Serve over stdio (MCP transport)
    // rmcp handles the JSON-RPC protocol: initialize, tools/list, tools/call
    // Multi-threaded runtime is required because tool handlers use
    // tokio::task::block_in_place to call async store methods from sync fn.
    // MCP-008: worker_threads(4) as a conservative floor to avoid thread
    // exhaustion under multi-client use.

    rt.block_on(async {
        let service = server.serve(rmcp::transport::stdio()).await?;
        service.waiting().await?;
        Ok::<(), anyhow::Error>(())
    })?;

    Ok(())
}

#[cfg(all(test, not(all(feature = "stable", not(feature = "full")))))]
mod tests {
    use super::*;

    #[test]
    fn token_file_trims_surrounding_whitespace() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let path = directory.path().join("token");
        std::fs::write(&path, "  file-token\n").expect("write token");
        let token = resolve_http_auth_token(None, Some(&path))
            .expect("valid token file")
            .expect("resolved token");
        assert_eq!(token, "file-token");
    }

    #[test]
    fn explicit_token_precedes_token_file() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let path = directory.path().join("token");
        std::fs::write(&path, "file-token\n").expect("write token");
        let token = resolve_http_auth_token(Some("explicit-token"), Some(&path))
            .expect("valid explicit token")
            .expect("resolved token");
        assert_eq!(token, "explicit-token");
    }

    #[test]
    fn token_file_rejects_internal_whitespace() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let path = directory.path().join("token");
        std::fs::write(&path, "first\nsecond\n").expect("write token");
        let error = resolve_http_auth_token(None, Some(&path))
            .expect_err("multiline token must fail")
            .to_string();
        assert!(error.contains("contains whitespace"));
    }

    #[test]
    fn missing_token_sources_resolve_to_none() {
        assert!(resolve_http_auth_token(None, None)
            .expect("missing token is not a parse error")
            .is_none());
    }
}
