# semantic-memory-mcp

An MCP (Model Context Protocol) server that gives your AI agent a
local-first knowledge base with hybrid search, evidence-scored
retrieval, contradiction detection, and autonomous memory lifecycle
management.

All data stays on your machine. SQLite for storage, in-process Candle
embedder (pure Rust, CPU-only), no cloud, no API keys, no telemetry.

**No Ollama required.** The default embedder is Candle — a pure-Rust ML
framework that runs nomic-embed-text-v1.5 in-process on CPU. The model
downloads automatically from HuggingFace on first use (cached after).
No external process, no model server, no GPU needed.

**No cloud dependencies.** Every component runs locally: the SQLite
database, the usearch vector index, the Candle embedding model, the
MCP server process. There are no calls to OpenAI, Anthropic, Pinecone,
Weaviase, Supabase, or any hosted service. The only network call is
the one-time model download from HuggingFace (cached after). Your
knowledge base never leaves your machine.

**Ollama still supported.** If you prefer using an external Ollama
instance, pass `--embedder ollama --embedding-url http://localhost:11434`.

[![Architecture](docs/architecture.svg)](docs/architecture.svg)

## What this gives your agent

Your agent gets a persistent knowledge base that:

- **Searches by meaning, not just keywords** — hybrid BM25 + vector
  similarity + Reciprocal Rank Fusion, with the full score breakdown
  exposed via `sm_search_explained`.
- **Tracks evidence confidence** — every item can carry algebraic
  provenance (semiring confidence scores with support counts).
- **Detects and corrects contradictions** — syndrome detection and
  belief propagation on conflict graphs. The decoder identifies
  inconsistent items and computes minimal corrections.
- **Decays old knowledge** — temporal weight factors in age,
  supersession, support, and contradiction signals.
- **Discovers related knowledge** — second-order retrieval through
  graph neighbors (discord search surfaces items related to your
  direct hits but not themselves direct hits).
- **Adapts search strategy per query** — adaptive routing profiles
  each query and decides which retrieval stages to activate.
- **Garbage-collects safely** — lawful subtraction with invariant
  verification. The lifecycle pass identifies items safe to forget,
  compress, or quarantine.
- **Audits every operation** — blake3-digested receipts for every
  mutation, replayable.
- **Tracks causal history** — typed graph edges (semantic, temporal,
  causal, entity) link items into a queryable knowledge graph.
- **Reasons over the graph** — factor graph belief propagation
  unifies all four edge types into a single probabilistic framework.
- **Finds structural gaps** — topological analysis computes Betti
  numbers and identifies voids in the knowledge graph.
- **Detects communities** — Leiden-inspired community detection with
  within-community contradiction scanning and compression-aware
  recommendations.
- **Self-edits memory** — `sm_update_fact` modifies facts in-place
  with re-embedding. `sm_consolidate_facts` merges duplicates with
  automatic supersession edges.
- **Learns from outcomes** — `sm_record_outcome` feeds good/bad/neutral
  signals to the RL routing policy, improving retrieval decisions over
  time.
- **Reranks with LLM** — optional `POST /rerank` endpoint uses an LLM
  (granite4.1:3b via Ollama) to rate query-document relevance 1-5 and
  reorder results for higher precision.
- **Extracts entities** — when `extract_entities: true` is passed to
  `sm_add_fact`, an LLM extracts named entities and auto-creates
  `entity:{name}` graph edges.
- **Generates community summaries** — when `summarize: true` is passed
  to `sm_community`, each community gets an LLM-generated summary
  paragraph.
- **Groups by community** — `group_by_community: true` in
  `sm_search_with_routing` clusters results by knowledge community for
  synthesis queries.
- **Routes adaptively** — `POST /search-routed` endpoint adjusts
  top_k and exactness profile based on query complexity class
  (A/B/C/D/E classification).
- **Serves via HTTP** — `--http-port 1738` starts a warm HTTP server
  alongside stdio MCP. 17 HTTP endpoints: /health, /search,
  /search-routed, /rerank, /stats, /add, /add-edge, /delete-fact,
  /record-outcome, /verify-integrity, /discord, /maintenance/check,
  /maintenance/vacuum, /maintenance/reembed, /maintenance/reconcile,
  /maintenance/compact-hnsw, /maintenance/auto-edge. Hooks,
  benchmarks, and scripts query it directly without spawning new
  processes (4.9x faster).
- **Compresses result content** — `compress_results` in SearchConfig
  shortens search result content to first sentence + key terms,
  reducing token cost by 30-50%.
- **Does 2-stage search** — Matryoshka multi-resolution: 64d truncated
  embeddings for fast candidate generation, 768d exact rerank.
- **Auto-creates graph edges** — `POST /maintenance/auto-edge` scans
  all facts across namespaces and creates entity edges between related
  items. Quality filtering with 300+ stopwords, only proper nouns,
  camelCase, and 5+ character words. Skips social media namespaces.
  Supports rebuild mode (invalidates old edges first). Runs
  automatically via cron (daily 4am) and primer hook (session start).
- **Hard-deletes facts** — `POST /delete-fact` removes a single fact
  and its FTS/vector entries by ID. Irreversible — prefer
  `sm_supersede_fact` for corrections. Useful for KB hygiene and
  removing junk facts.
- **Adds edges via HTTP** — `POST /add-edge` creates a typed graph edge
  between two nodes via the HTTP server. Same semantics as the
  `sm_add_graph_edge` MCP tool but accessible from scripts and hooks.
- **Enriches discord results** — `/discord` and `/search-routed` now
  return fact content and namespace for graph neighbors via `get_fact`
  enrichment. Previously discord results only had IDs — now you get
  the full content of each second-order result.

The combination of hybrid retrieval, provenance-weighted belief
propagation, typed graph edges, and autonomous lifecycle management
in a single local-first Rust substrate is uncommon. This is
knowledge management, not just vector search.

## Installation

### Option 1: Install from crates.io (recommended)

```bash
cargo install semantic-memory-mcp
```

This pulls semantic-memory 0.5.8 and all dependencies from crates.io
automatically. No need to clone any repos.

### Option 2: Build from source

The MCP server depends on [semantic-memory](https://github.com/RecursiveIntell/semantic-memory),
which in turn depends on several crates from the same stack. All of
them are published on crates.io, so `cargo build` from the standalone
repo will resolve them from the registry.

```bash
git clone https://github.com/RecursiveIntell/semantic-memory-mcp.git
cd semantic-memory-mcp
cargo build --release
# Binary: target/release/semantic-memory-mcp
```

If you prefer to build the full stack from source (all repos
cloned as siblings), see the [dependency table](#dependencies)
below for the complete list.

### Dependencies

The MCP server depends on one crate: `semantic-memory`. That crate
in turn depends on several stack crates. All are on both crates.io
and GitHub:

| Crate | crates.io | GitHub | Purpose |
|-------|-----------|--------|---------|
|| [semantic-memory](https://github.com/RecursiveIntell/semantic-memory) | [0.5.8](https://crates.io/crates/semantic-memory) | [GitHub](https://github.com/RecursiveIntell/semantic-memory) | Core search engine, storage, graph, reasoning |
| [stack-ids](https://github.com/RecursiveIntell/stack-ids) | [0.1.1](https://crates.io/crates/stack-ids) | [GitHub](https://github.com/RecursiveIntell/stack-ids) | Typed IDs, scopes, trace context, BLAKE3 digests |
| [bitemporal-runtime](https://github.com/RecursiveIntell/bitemporal-runtime) | [0.1.0](https://crates.io/crates/bitemporal-runtime) | [GitHub](https://github.com/RecursiveIntell/bitemporal-runtime) | Bitemporal truth (valid_time / recorded_time) |
| [boundary-compiler](https://github.com/RecursiveIntell/boundary-compiler) | [0.1.0](https://crates.io/crates/boundary-compiler) | [GitHub](https://github.com/RecursiveIntell/boundary-compiler) | RFC 8785 JSON Canonicalization (JCS) |
| [forge-memory-bridge](https://github.com/RecursiveIntell/forge-memory-bridge) | [0.1.1](https://crates.io/crates/forge-memory-bridge) | [GitHub](https://github.com/RecursiveIntell/forge-memory-bridge) | Projection import transforms |

All of these are published on crates.io. If you install via
`cargo install semantic-memory-mcp`, cargo resolves them
automatically — you do not need to clone anything.

### Building the full stack from source

If you want to modify the underlying library alongside the MCP
server, clone all repos as siblings:

```bash
mkdir semantic-memory-stack && cd semantic-memory-stack

git clone https://github.com/RecursiveIntell/semantic-memory.git
git clone https://github.com/RecursiveIntell/semantic-memory-mcp.git
git clone https://github.com/RecursiveIntell/stack-ids.git
git clone https://github.com/RecursiveIntell/bitemporal-runtime.git
git clone https://github.com/RecursiveIntell/boundary-compiler.git
git clone https://github.com/RecursiveIntell/forge-memory-bridge.git

# The path deps in semantic-memory/Cargo.toml use ../stack-ids, ../bitemporal-runtime, etc.
# With all repos cloned as siblings, these paths resolve correctly.

cd semantic-memory-mcp
cargo build --release
```

The `semantic-memory/Cargo.toml` has `path = "../stack-ids"` (and
similar) with version requirements alongside. Cargo prefers the
path dep when it exists, falls back to crates.io when it doesn't.
So you can clone just `semantic-memory-mcp` for a standalone build,
or clone all siblings for full-stack development.

## Prerequisites

**Default (Candle embedder — no Ollama needed):**

No prerequisites. The model (nomic-embed-text-v1.5) downloads
automatically from HuggingFace on first use and is cached in
`~/.cache/huggingface/hub`. Subsequent runs load from cache with no
network access.

**Ollama alternative:**

If you prefer using Ollama, install it and pull an embedding model:

```bash
ollama pull nomic-embed-text
```

Then pass `--embedder ollama` when starting the server.

## Configuration

### Hermes Agent

Add to `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  semantic_memory:
    command: "semantic-memory-mcp"
    args: ["--memory-dir", "/home/user/.local/share/semantic-memory"]
```

### Claude Desktop

Add to `claude_desktop_config.json` (usually at
`~/Library/Application Support/Claude/claude_desktop_config.json`
on macOS or `%APPDATA%\Claude\claude_desktop_config.json` on Windows):

```json
{
  "mcpServers": {
    "semantic_memory": {
      "command": "semantic-memory-mcp",
      "args": ["--memory-dir", "/home/user/.local/share/semantic-memory"]
    }
  }
}
```

### Cursor / Windsurf

Add to your MCP settings (Settings → MCP):

```json
{
  "mcpServers": {
    "semantic_memory": {
      "command": "semantic-memory-mcp",
      "args": ["--memory-dir", "/home/user/.local/share/semantic-memory"]
    }
  }
}
```

### Remote Ollama

If you prefer Ollama on a different machine:

```json
{
  "mcpServers": {
    "semantic_memory": {
      "command": "semantic-memory-mcp",
      "args": [
        "--memory-dir", "/home/user/.local/share/semantic-memory",
        "--embedder", "ollama",
        "--embedding-url", "http://192.168.1.50:11434",
        "--embedding-model", "nomic-embed-text",
        "--embedding-dims", "768"
      ]
    }
  }
}
```

## CLI options

```
semantic-memory-mcp --memory-dir <DIR> [OPTIONS]

Options:
  --memory-dir <DIR>         Path to the memory store directory (required, created if absent)
  --embedder <BACKEND>       Embedding backend: candle (default), ollama, or mock
  --embedding-url <URL>      Ollama server URL (only used with --embedder ollama, default: http://localhost:11434)
  --embedding-model <NAME>   Embedding model name (default: nomic-embed-text)
  --embedding-dims <N>       Embedding dimensions (default: 768)
```

`--memory-dir` is a directory path, not a SQLite file path. The
SQLite database is created as `memory.db` inside this directory,
alongside the usearch sidecar files (`.hnsw.data`, `.hnsw.graph`,
`.hnsw.manifest.json`).

### Embedder backends

| Backend | Description | Requires |
|---------|-------------|----------|
| `candle` (default) | In-process pure-Rust ML (CPU-only). Downloads nomic-embed-text-v1.5 from HuggingFace on first use, cached after. | Nothing — just `cargo install` |
| `ollama` | External Ollama server. Use if you already run Ollama or want GPU acceleration. | Ollama installed + model pulled |
| `mock` | Deterministic hash-based embeddings for testing. | Nothing |

## How search works

[![Search Pipeline](docs/search-pipeline.svg)](docs/search-pipeline.svg)

When the agent calls `sm_search`, the query flows through:

1. **Embedding** — the query text is embedded by the configured backend
   (Candle in-process by default, or Ollama if specified), producing a
   768-dimensional vector.

2. **Parallel retrieval** — two searches run simultaneously:
   - **BM25 (FTS5)** — SQLite's full-text search ranks results by
     keyword relevance using BM25 scoring.
   - **Vector (usearch)** — the HNSW index finds the nearest neighbors
     by cosine similarity to the query embedding.

3. **Reciprocal Rank Fusion** — the two ranked lists are merged using
   RRF: `score = 1/(k + bm25_rank) + 1/(k + vector_rank)`. This
   doesn't require score calibration — it works off ranks alone,
   which is why it's robust across different embedding models and
   corpus sizes.

4. **Optional advanced stages** — when `sm_search_with_routing` is
   used, the query is profiled and additional stages may activate:
   - **Routing** — decides whether to run the decoder, discord, or
     graph expansion based on query characteristics.
   - **Decoder** — detects contradictions in the results and computes
     corrections via belief propagation.
   - **Factor graph** — runs belief propagation over stored graph
     edges to refine confidence scores using the knowledge graph's
     structure.

5. **Results + receipt** — returns ranked results with scores, source
   types, and (optionally) a content-addressed receipt for audit.

## Tools

The server exposes 38 MCP tools. Use `tools/list` as the source of
truth for the available tool surface on your build.

### Core tools (always available)

#### sm_search

Hybrid BM25 + vector + RRF semantic search over the knowledge base.
By default, results targeted by `supersedes` graph edges are filtered
when non-superseded alternatives exist. Queries that explicitly ask for
stale, old, historical, or superseded facts keep those results available.

```json
{
  "query": "rust async runtime tokio",
  "top_k": 5,
  "namespaces": ["general", "coding"]
}
```

Returns ranked results with content, scores, and stable result IDs
(`result_id` field) for downstream tool chaining (e.g., passing to
`sm_graph_path` or `sm_set_provenance`).

#### sm_search_explained

Same as `sm_search` but with the full per-signal score breakdown:
BM25 score, vector score, recency score, RRF score, weights, and
contribution percentages. Useful for debugging retrieval quality.
It applies the same superseded-result filtering as `sm_search`.

#### sm_add_fact

Add a fact to the knowledge base. The fact is embedded by the configured
backend (Candle by default) and indexed for both BM25 and vector search.

```json
{
  "content": "Rust 1.75 stabilized async fn in traits",
  "namespace": "rust-facts",
  "source": "https://blog.rust-lang.org/2023/12/21/async-fn-rpit-in-traits.html"
}
```

#### sm_supersede_fact

Create a replacement fact and link it to an older stale fact with a
durable entity edge using `relation: "supersedes"`. Use this for verified
corrections so old facts remain auditable but no longer stand alone as
unmarked stale context.

```json
{
  "old_fact_id": "fact:a1b2c3d4-...",
  "content": "The current verified fact as of 2026-06-21 is ...",
  "namespace": "codex",
  "source": "repo:/path/or/url",
  "reason": "verified against current repository state"
}
```

#### sm_ingest_document

Ingest a longer document with automatic text chunking. Each chunk
is embedded and indexed independently. Returns the document ID and
chunk count.

```json
{
  "title": "Tokio Tutorial",
  "content": "Tokio is an asynchronous runtime for the Rust programming language...",
  "namespace": "docs"
}
```

#### sm_stats

Get knowledge base statistics: fact count, chunk count, document
count, session count, message count, graph edge count, database size,
embedding model and dimensions.

#### sm_graph_path

Find the shortest path between two items in the knowledge graph.
Traverses semantic, temporal, causal, entity, and stored graph
edges. Returns the path as a list of node IDs with per-hop edge
evidence (edge type, weight, metadata).

```json
{
  "from_id": "fact:a1b2c3d4-...",
  "to_id": "fact:e5f6g7h8-...",
  "max_depth": 5
}
```

#### sm_set_provenance

Set evidence confidence for an item using the ConfidenceSemiring:
confidence in [0.0, 1.0] with a support count of independent
observations. Returns a provenance receipt.

```json
{
  "item_id": "fact:a1b2c3d4-...",
  "confidence": 0.92,
  "support_count": 3
}
```

#### sm_add_graph_edge

Add a durable, typed graph edge between two nodes. Nodes use
prefixed IDs (`fact:<uuid>`, `namespace:<name>`, `document:<id>`).
Edge types: `semantic` (cosine_similarity), `temporal` (delta_secs),
`causal` (confidence + evidence_ids), `entity` (relation name).
Insertion is idempotent — same edge returns existing ID.

```json
{
  "source": "fact:a1b2c3d4-...",
  "target": "fact:e5f6g7h8-...",
  "edge_type": "causal",
  "confidence": 0.85,
  "evidence_ids": ["fact:ev1-...", "fact:ev2-..."],
  "weight": 1.0
}
```

```json
{
  "source": "fact:a1b2c3d4-...",
  "target": "namespace:rust-facts",
  "edge_type": "entity",
  "relation": "belongs_to",
  "weight": 1.0
}
```

#### sm_list_graph_edges

List graph edges for a specific node (as source or target), or all
stored graph edges if no node_id is provided. Returns non-invalidated
edges only.

```json
{ "node_id": "fact:a1b2c3d4-..." }
```

#### sm_invalidate_graph_edge

Invalidate a stored graph edge by ID. Append-only — the edge row is
never deleted, only marked invalidated with a reason.

```json
{
  "edge_id": "edge:abc123-...",
  "reason": "superseded by newer evidence"
}
```

### Advanced tools (full feature)

#### sm_route_query

Profile a query and get an adaptive routing decision. Determines
which retrieval stages (BM25, vector, rerank, graph, decoder,
discord) should be activated for this query. Useful for
understanding why certain stages fire or don't.

```json
{ "query": "what changed between v0.4 and v0.5" }
```

#### sm_search_with_routing

Adaptive search: profiles the query, routes to appropriate stages,
and applies factor graph belief propagation if the decoder stage is
activated. Returns results with routing decision, decoder status,
factor graph analysis, and matryoshka multi-resolution routing
payload.

```json
{
  "query": "what changed between v0.4 and v0.5",
  "top_k": 10,
  "contradictions": [["fact:old-claim-...", "fact:new-claim-..."]]
}
```

#### sm_decoder_analyze

Detect contradictions and inconsistencies in search results. Runs
syndrome detection, computes corrections, and applies belief
propagation to refine confidence scores. Operates on caller-supplied
results — does not require graph edges from the store.

```json
{
  "results": [
    ["fact:item-a-...", 0.9],
    ["fact:item-b-...", 0.7]
  ],
  "contradictions": [["fact:item-a-...", "fact:item-b-..."]]
}
```

#### sm_discord_search

Second-order retrieval: find items related to your search results
through the knowledge graph, but NOT themselves direct hits. Loads
graph edges from the store automatically — caller supplies only the
direct result IDs.

```json
{
  "direct_result_ids": [
    "fact:a1b2c3d4-...",
    "fact:e5f6g7h8-..."
  ]
}
```

Returns items connected to your direct hits via graph edges, scored by
relationship strength. Useful for discovering adjacent knowledge you
didn't think to search for.

#### sm_run_lifecycle

Autonomous memory health check. Runs in one call:
- Syndrome detection on the supplied items
- Correction computation
- Subtraction candidate identification (items safe to forget/compress)
- Compression recompression trigger check
- Topological analysis (Betti numbers + voids)
- Community detection with contradiction scanning
- Subgraph pruning assessment
- Compression governor quantization assessment

```json
{
  "item_ids": [
    "fact:a1b2c3d4-...",
    "fact:e5f6g7h8-...",
    "fact:i9j0k1l2-..."
  ]
}
```

#### sm_factor_graph

Run factor graph belief propagation on heterogeneous graph edges
stored in the knowledge base. Models all 4 edge types (semantic,
temporal, causal, entity) as factors in a single probabilistic
reasoning framework. Loads edges from the store automatically — caller
supplies only node initial beliefs and optional config overrides.

```json
{
  "nodes": [
    { "item_id": "fact:a1b2-...", "initial_belief": 0.8 },
    { "item_id": "fact:e5f6-...", "initial_belief": 0.6 },
    { "item_id": "fact:i9j0-...", "initial_belief": 0.3 }
  ],
  "semantic_weight": 0.35,
  "causal_weight": 0.30,
  "max_iterations": 100
}
```

Returns unified confidence scores after message propagation
converges, with per-edge-type factor counts and convergence metadata.

#### sm_topology

Find topological voids in the knowledge graph. Computes Betti numbers
(connected components and independent cycles) and detects structural
gaps. Loads edges from the store automatically.

Returns Betti numbers, void descriptions with nearby items and
suggested connections, and a gap report summary.

#### sm_community

Detect communities in the knowledge graph using a Leiden-inspired
algorithm. Loads edges from the store automatically. Returns community
assignments with member lists, optional within-community contradiction
scans, and community-aware compression recommendations.

```json
{
  "resolution": 1.0,
  "seed": 42,
  "contradictions": [["fact:a1b2-...", "fact:e5f6-..."]]
}
```

## Tool chaining

The tools are designed to chain. The `result_id` field returned by
`sm_search` is a stable prefixed node ID (`fact:<uuid>`,
`chunk:<uuid>`, etc.) that can be passed directly to downstream tools:

```
sm_search("tokio async runtime")
  → results[0].result_id = "fact:abc123-..."

sm_graph_path("fact:abc123-...", "fact:def456-...")
  → path through the knowledge graph

sm_set_provenance("fact:abc123-...", confidence=0.9, support_count=2)
  → confidence recorded

sm_add_graph_edge("fact:abc123-...", "namespace:rust", "entity", relation="belongs_to")
  → edge added

sm_discord_search(["fact:abc123-...", "fact:def456-..."])
  → second-order neighbors discovered
```

## Feature flags

| Feature | Default | Description |
|---------|---------|-------------|
| `full` | yes | All features — the full 38-tool surface + Candle embedder + late-interaction + TurboQuant codec. This is the default build. |
| `search` | no | Core search only (BM25 + vector + RRF, add facts, stats, graph path, graph edges, provenance) + Candle embedder. Minimal build with no external codec deps. |
| `candle-embedder` | yes (via full/search) | In-process pure-Rust Candle embedder (CPU-only, no Ollama required). |

Build with `--no-default-features --features search` for the minimal
profile. The `full` feature enables all semantic-memory sub-features
(provenance, temporal, decoder, discord, routing, subtraction,
compression governor, integration, topology, community) plus the
Candle embedder.

The `full` feature does NOT pull in the `turbo-quant-codec` or
`poly-kv-pool` features — those are experimental codec integrations
that remain opt-in in the underlying library and are not needed for
the MCP server's functionality.

## Architecture

[![Architecture](docs/architecture.svg)](docs/architecture.svg)

```
semantic-memory-mcp (MCP stdio server, rmcp SDK)
  └── semantic-memory (Rust library, 0.5.8)
        ├── Candle embedder (pure-Rust, CPU-only, default — no Ollama required)
        ├── SQLite (authoritative storage, FTS5, WAL)
        ├── usearch 2.25 (vector sidecar, default backend)
        ├── Provenance (Boolean, Tropical, Probability, Confidence semirings)
        ├── Temporal weight (age + supersession + support + contradiction)
        ├── Decoder (syndromes + corrections + belief propagation)
        ├── Subtraction (lawful forgetting + invariant verification)
        ├── Compression governor (importance-driven quantization level decisions)
        ├── Routing (query profiling + adaptive stage selection)
        ├── Discord (second-order graph-neighbor retrieval)
        ├── Stored graph edges (durable, typed, append-only with invalidation)
        ├── Factor graph (unified probabilistic reasoning over all edge types)
        ├── Topology (Betti numbers, void detection)
        ├── Community detection (Leiden-inspired, contradiction-aware)
        └── Integration (cross-feature wiring: routing → decoder → subtraction → compression → discord → provenance)
```

The server uses rmcp's `#[tool_router]` macro to auto-generate JSON
Schema for each tool's parameters. All tool handlers return
`Result<String, ErrorData>` — errors are protocol-level MCP errors,
not string-encoded error messages.

Graph edges and factor inputs are loaded from the store automatically
— the caller never needs to supply edge arrays. This is a design
decision: the store is the single source of truth for graph state.

## Underlying crate

This server wraps the [semantic-memory](https://crates.io/crates/semantic-memory)
crate (0.5.8), which provides the storage engine, search pipeline,
and all feature modules. See the
[semantic-memory crate documentation](https://github.com/RecursiveIntell/Libraries/tree/main/semantic-memory)
for the full library API, including direct usage without an MCP
server.

### Dependency crates

The underlying library depends on several crates from the same
stack, all published on crates.io:

- [stack-ids](https://crates.io/crates/stack-ids) — typed IDs, scopes,
  trace context, BLAKE3 content digests.
- [bitemporal-runtime](https://crates.io/crates/bitemporal-runtime) —
  bitemporal truth primitives (valid_time / recorded_time tracking,
  append-supersede, as-of queries).
- [boundary-compiler](https://crates.io/crates/boundary-compiler) —
  RFC 8785 JSON Canonicalization (JCS) with strict duplicate-key
  rejection.
- [forge-memory-bridge](https://crates.io/crates/forge-memory-bridge)
  — transformation layer from Forge export envelopes to memory import
  batches.

## Scope and limits

- The default embedder (Candle) downloads nomic-embed-text-v1.5 from
  HuggingFace on first use (~550MB, cached after). No Ollama required.
  If you prefer Ollama, pass `--embedder ollama`.
- The Candle embedder is CPU-only and pure-Rust (no C++ runtime, no
  heap corruption risk). It processes embeddings one at a time to keep
  memory bounded. First-run latency is higher (model download + load);
  subsequent runs load from cache in ~1-2 seconds.
- The `search`-only build (no `full` feature) does not expose advanced
  tools (routing, decoder, discord, lifecycle, factor graph, topology,
  community). The `full` feature is the default.
- Graph-based tools (discord, factor graph, topology, community) load
  edges from the store. With zero stored edges, these tools return
  empty results — they are not broken, they have no graph to work
  with. Add edges with `sm_add_graph_edge` to give them something to
  traverse.
- The `decoder_executed` field in `sm_search_with_routing` is
  currently always `false` — the routing decision is computed and
  reported, but the decoder does not yet re-rank search results in the
  live search path. The factor graph analysis runs independently when
  the decoder stage is planned. This is a known gap, not a bug.
- All state is local. There is no sync, no federation, no network
  calls beyond the one-time HuggingFace model download (cached after).
  This is a feature, not a limitation — local-first is the design goal.

## License

Apache-2.0

## Links

- [semantic-memory crate](https://crates.io/crates/semantic-memory)
- [GitHub repository](https://github.com/RecursiveIntell/Libraries/tree/main/semantic-memory-mcp)
- [MCP Protocol](https://modelcontextprotocol.io/)
- [rmcp Rust SDK](https://github.com/modelcontextprotocol/rust-sdk)
- [Ollama](https://ollama.ai/)
