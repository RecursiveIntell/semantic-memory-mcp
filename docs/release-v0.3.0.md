# semantic-memory-mcp v0.3.0 — Local-First Knowledge Base for AI Agents

Published on crates.io: https://crates.io/crates/semantic-memory-mcp
GitHub: https://github.com/RecursiveIntell/semantic-memory-mcp
PR to awesome-mcp-servers: https://github.com/punkpeye/awesome-mcp-servers/pull/8676

## What it is

An MCP (Model Context Protocol) server that gives your AI agent a persistent
knowledge base with hybrid search, contradiction detection, and autonomous
memory lifecycle management.

All data stays on your machine. SQLite for storage, in-process Candle embedder
(pure Rust, CPU-only), no cloud, no API keys, no telemetry.

## Why it's different

Every other MCP memory server does some combination of: store text, embed it,
search it. Maybe a knowledge graph. Maybe reranking.

semantic-memory-mcp ships 33 tools that do what no other single MCP memory
server does:

1. Contradiction detection with belief propagation — the decoder identifies
   inconsistent items in your knowledge base and computes minimal corrections.
   No other MCP memory server has this.

2. Adaptive query routing — profiles each query and decides which retrieval
   stages to activate (BM25 only? vector+rerank? full graph reasoning?).

3. Factor graph belief propagation — unifies all four graph edge types
   (semantic, temporal, causal, entity) into a single probabilistic framework
   for reasoning over the knowledge graph.

4. Community detection — Leiden-inspired clustering with within-community
   contradiction scanning and compression-aware recommendations.

5. Topological gap analysis — computes Betti numbers to find structural voids
   in your knowledge graph. Tells you what you don't know.

6. Memory lifecycle GC — lawful subtraction with invariant verification.
   Identifies items safe to forget, compress, or quarantine.

7. Bitemporal truth — every fact has a recorded time and a valid time.
   Supersession tracking with automatic filtering in search results.

8. Blake3-digested receipts — every mutation is receipted and replayable.

## Install

```bash
cargo install semantic-memory-mcp
```

Then add to your MCP client config:

```json
{
  "mcpServers": {
    "semantic-memory": {
      "command": "semantic-memory-mcp",
      "args": ["--memory-dir", "/path/to/memory-store"]
    }
  }
}
```

No Ollama required. The default embedder is Candle — a pure-Rust ML framework
that runs nomic-embed-text-v1.5 in-process on CPU. The model downloads
automatically from HuggingFace on first use (cached after).

## Stats

- 33 MCP tools
- 396 tests passing (381 in semantic-memory + 15 integration in MCP)
- 0 compiler warnings
- Pure Rust, zero non-Rust runtime deps
- SQLite + usearch — single-file storage, no external vector DB

## Architecture

semantic-memory-mcp wraps the semantic-memory crate (also published:
https://crates.io/crates/semantic-memory) as an MCP server using the rmcp
Rust MCP SDK. It works with Hermes Agent, Claude Desktop, Cursor, and any
MCP-compatible client.

The HTTP server mode (--http-port) exposes the same operations over a local
TCP port for hooks, benchmarks, and scripts.

## Published crates

- semantic-memory v0.5.7 — the core library
- semantic-memory-mcp v0.3.0 — the MCP server
- claim-ledger v0.1.0 — claim/evidence/provenance ledger (optional dep)
- llm-output-parser v0.2.0 — robust LLM output parsing (optional dep)
- boundary-compiler v0.1.0 — JSON canonicalization with boundary profiles
- turbo-quant v0.2.1 — vector compression codecs (used by semantic-memory)

All on crates.io. All pure Rust. All local-first.