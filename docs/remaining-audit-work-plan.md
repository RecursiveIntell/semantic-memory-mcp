# Remaining Audit Work Plan

## Current state

Both repos have the containment patch committed and all tests passing:
- semantic-memory: commit 60c064e, 376 tests pass
- semantic-memory-mcp: commit 0cccc30, 43 tests pass

This plan covers all remaining audit findings that were deferred because they require code changes beyond the containment patch. Each item has exact file paths, line numbers, code snippets, and step-by-step instructions.

---

## ITEM 1: Remove idempotent_hint from 11 mutation tools

### Problem

MCP defines `idempotentHint` as "repeated calls with the same arguments have no additional environmental effect." These tools create new records or mutate state on every call. The annotation is materially wrong.

### Exact locations in `/home/sikmindz/Coding/Libraries/semantic-memory-mcp/src/server.rs`

Each of these has `annotations(idempotent_hint = true)` or `annotations(read_only_hint = false, idempotent_hint = true)` in the `#[tool()]` attribute preceding the function:

| Line | Tool | Current annotation | Action |
|------|------|--------------------|--------|
| 2220 | `sm_ingest_document` | `annotations(idempotent_hint = true)` | Remove the annotations line |
| 2521 | `sm_supersede_fact` | `annotations(idempotent_hint = true)` | Remove the annotations line |
| 2591 | `sm_create_session` | `annotations(idempotent_hint = true)` | Already commented out (DEPRECATED block) — leave as-is |
| 3382 | `sm_set_provenance` | `annotations(idempotent_hint = true)` | Remove the annotations line |
| 3671 | `sm_add_graph_edge` | `annotations(idempotent_hint = true)` | Remove the annotations line |
| 3799 | `sm_invalidate_graph_edge` | `annotations(idempotent_hint = true)` | Remove the annotations line |
| 4227 | `sm_update_fact` | `annotations(idempotent_hint = true)` | Remove the annotations line |
| 4538 | `sm_create_claim` | `annotations(read_only_hint = false, idempotent_hint = true)` | Change to `annotations(read_only_hint = false)` |
| 5028 | `sm_reconcile` | `annotations(idempotent_hint = true)` | Remove the annotations line |
| 5073 | `sm_vacuum` | `annotations(idempotent_hint = true)` | Remove the annotations line |
| 5093 | `sm_reembed_all` | `annotations(idempotent_hint = true)` | Remove the annotations line |
| 5273 | `sm_import_envelope` | `annotations(idempotent_hint = true)` | Remove the annotations line |

### Steps

1. For each tool above (except `sm_create_session` which is already commented out), find the `#[tool( ... )]` block.
2. If the annotations line is `annotations(idempotent_hint = true)` on its own line — delete that line.
3. If the annotations line is `annotations(read_only_hint = false, idempotent_hint = true)` — change to `annotations(read_only_hint = false)`.
4. Do NOT touch any `#[tool( description = "..." )]` lines that have no annotations.
5. Do NOT touch `annotations(read_only_hint = true)` on read-only tools — those are correct.

### Pitfall

The `#[tool()]` attribute is multi-line. The `annotations(...)` line is a separate line within it. Removing only the annotations line is safe. Do NOT remove the entire `#[tool()]` block or the description line.

Example of correct edit for `sm_supersede_fact`:
```
Before:
    #[tool(
        description = "Create a replacement fact and link it to a stale fact via 'supersedes' edge. Use this when a fact has been corrected or updated.",
        annotations(idempotent_hint = true)
    )]

After:
    #[tool(
        description = "Create a replacement fact and link it to a stale fact via 'supersedes' edge. Use this when a fact has been corrected or updated."
    )]
```

### Verification

```bash
cd /home/sikmindz/Coding/Libraries/semantic-memory-mcp
grep -n "idempotent_hint" src/server.rs
# Expected: zero results (all removed)
cargo check --features full
cargo test --features full
```

### Estimated effort

15 minutes. Mechanical find-and-replace. No logic changes.

---

## ITEM 2: Fix sm_search_as_of_preview response message

### Problem

The tool was renamed to `sm_search_as_of_preview` and the description was updated to say "PREVIEW: Temporal filtering is not yet implemented." But the response JSON still says `"Found N facts valid as of DATE"`, which claims the results were temporally filtered when they were not.

### Exact location

`/home/sikmindz/Coding/Libraries/semantic-memory-mcp/src/server.rs`, in `fn sm_search_as_of_preview` (around line 4800):

```rust
"message": format!("Found {} facts valid as of {}", filtered.len(), as_of_date),
```

### Fix

Change to:
```rust
"message": format!("PREVIEW: {} results returned. Temporal filtering NOT applied — results may include facts not valid as of {}.", filtered.len(), as_of_date),
```

### Verification

```bash
grep -A2 "sm_search_as_of_preview" src/server.rs | grep "message"
cargo test --features full
```

### Estimated effort

2 minutes.

---

## ITEM 3: Fix HTTP routed-search hardcoded verification

### Problem

The HTTP routed-search endpoint in `http_server.rs` hardcodes `"widening_occurred": false` and verification/provenance fields instead of deriving them from actual execution.

### Exact locations

`/home/sikmindz/Coding/Libraries/semantic-memory-mcp/src/http_server.rs`:

- Line 308: `"widening_occurred": false,`
- Line 677: `"widening_occurred": false,`

Search for `widening_occurred` and `verification_status` and `verified` in the routed-search response construction.

### Fix

For each hardcoded field, either:
(a) Derive it from the actual search execution result if the data is available, OR
(b) Remove the field if it cannot be derived, OR
(c) Change the value to `null` with a comment: `// TODO: derive from execution receipt`

Specifically:
- `"widening_occurred": false` → `"widening_occurred": null` with TODO comment
- Any `"verification_status": "verified"` or similar → `null` with TODO comment
- Any hardcoded `"provenance"` fields that are not derived from the search receipt → `null`

### Steps

1. Find `handle_search_routed` function in `http_server.rs`
2. Find all hardcoded boolean/string fields in the response JSON
3. For each, check if the search execution result provides the data
4. If not available, set to `null` with a TODO comment
5. If available, wire it through

### Verification

```bash
grep -n "widening_occurred\|verification_status\|verified.*true" src/http_server.rs
# Should show null or derived values, not hardcoded true/false
cargo test --features full
```

### Estimated effort

30 minutes. Need to trace the search execution path and determine which fields are available.

---

## ITEM 4: Claim tool persistence

### Problem

`sm_create_claim`, `sm_add_evidence`, and `sm_judge_support` instantiate `Claim::new`, `EvidenceBundle::new`, and `SupportJudgment` objects, return IDs, and say "created"/"added"/"recorded" — but never write these objects to durable storage. They disappear on restart.

### Current architecture

The server has a `ClaimLedgerStore` struct:
```rust
#[cfg(feature = "claim-integration")]
struct ClaimLedgerStore {
    entries: Vec<claim_ledger::LedgerEntry>,
    path: std::path::PathBuf,
    // ...
}
```

The claim-ledger crate at `/home/sikmindz/Coding/Libraries/claim-ledger/src/ledger.rs` has methods:
- `add_claim` — appends to an in-memory `Vec<LedgerEntry>`
- `add_support_judgment`
- `add_support_admission`
- `add_contradiction_candidate`
- `add_contradiction_resolved`
- `compact_ledger`
- `verify_ledger`

The `ClaimLedgerStore` in the MCP server wraps this but the tools are not calling the write methods.

### Fix approach

This requires three changes:

#### Step 1: Wire ClaimLedgerStore writes into the tools

In `sm_create_claim` (line 4540):
- After `Claim::new(...)` succeeds, call `self.claim_ledger_store.lock().unwrap().add_claim(claim)` (or whatever the ClaimLedgerStore API expects)
- The `ClaimLedgerStore` needs a method to persist entries to its `path` (a JSONL file)

In `sm_add_evidence` (line 4617):
- After `EvidenceBundle::new(...)` succeeds, persist it

In `sm_judge_support` (line 4653):
- After `SupportJudgment` construction, persist it

#### Step 2: Add persistence to ClaimLedgerStore

The `ClaimLedgerStore` currently holds entries in memory. It needs a `flush()` or `persist()` method that writes entries to the JSONL file at `self.path`. Check the existing `ClaimLedgerStore` implementation for what it already does:

```bash
grep -n "fn " src/server.rs | grep -i "claim_ledger"
```

Look for methods like `load`, `save`, `flush`, `persist` on `ClaimLedgerStore`.

#### Step 3: Add restart-roundtrip test

In `tests/integration.rs`, add a test that:
1. Creates a claim via `sm_create_claim`
2. Drops the server (simulating restart)
3. Creates a new server with the same `--memory-dir`
4. Verifies the claim is still present

### Pitfalls

- The `ClaimLedgerStore` may need file-system persistence (JSONL append). Check if `claim-ledger` crate already provides this.
- Thread safety: `ClaimLedgerStore` is behind a `Mutex`. Writes must be short.
- The claim-ledger `LedgerEntry` type must be serializable. Check if it derives `Serialize`.

### Verification

```bash
cargo test --features full -- --claim_persistence
# New test should pass: create claim, restart, verify claim exists
```

### Estimated effort

2-4 hours. Need to understand ClaimLedgerStore internals, wire persistence, add test.

---

## ITEM 5: Consolidation atomic transaction

### Problem

`sm_consolidate_facts` (line 4259) performs three operations without a transaction:
1. `store.update_fact(&keep_bare, &final_content)` — updates kept fact in place
2. `store.add_fact(&namespace, &final_content, None, None)` — creates a duplicate
3. `store.add_graph_edge(...)` — adds supersession edge

If step 3 fails, there are two equivalent live facts. The edge result is assigned to `let _edge` (discarded).

### Core transaction API

The core `MemoryStore` has `with_write_conn`:
```rust
fn with_write_conn<F, T>(&self, f: F) -> Result<T, MemoryError>
where
    F: FnOnce(&rusqlite::Connection) -> Result<T, MemoryError> + Send + 'static,
    T: Send + 'static,
```

This provides a single `Connection` for atomic operations. Any `f` that executes multiple SQL statements through this connection runs in an implicit SQLite transaction if the connection has not been committed.

However, the current `update_fact`, `add_fact`, and `add_graph_edge` are separate async methods on `MemoryStore` — each opens its own write connection. There is no public API to run all three in one transaction.

### Fix approach

#### Option A: Add a `consolidate_facts` method to core MemoryStore

In `/home/sikmindz/Coding/Libraries/semantic-memory/src/lib.rs`, add:

```rust
pub async fn consolidate_facts(
    &self,
    keep_id: &str,
    supersede_id: &str,
    merged_content: &str,
) -> Result<ConsolidationReceipt, MemoryError> {
    self.with_write_conn(move |conn| {
        // 1. Update kept fact content
        conn.execute("UPDATE facts SET content = ?1 WHERE id = ?2", params![merged_content, keep_id])?;
        // 2. Add supersession edge
        conn.execute(
            "INSERT INTO graph_edges (source, target, edge_type, recorded_at, is_invalidated)
             VALUES (?1, ?2, ?3, datetime('now'), 0)",
            params![
                format!("fact:{}", keep_id),
                format!("fact:{}", supersede_id),
                r#"{"Entity":{"relation":"supersedes"}}"#
            ]
        )?;
        // 3. Do NOT create a duplicate fact — the kept fact already has the merged content
        Ok(ConsolidationReceipt { kept_id: keep_id.to_string(), superseded_id: supersede_id.to_string() })
    })
}
```

Key insight: the current code creates a duplicate fact AND updates the kept fact in place. The correct append-plus-supersession approach is:
- Create ONE new canonical fact with the merged content
- Supersede BOTH source facts
- Do NOT update any fact in place

But the simpler fix is: update the kept fact in place and supersede the other — all in one transaction. No duplicate.

#### Option B: Keep it hidden from stable

If the transaction work is too risky, keep `sm_consolidate_facts` excluded from the stable profile (already done) and add a compile-time `#[cfg(feature = "admin-ops")]` gate so it only compiles with the admin-ops feature.

### Steps for Option A

1. Add `ConsolidationReceipt` struct to core types
2. Add `consolidate_facts` method to `MemoryStore` in `lib.rs`
3. Change `sm_consolidate_facts` in `server.rs` to call `store.consolidate_facts()` instead of doing three separate operations
4. Remove the `let _edge` discard — check the result
5. Add test: consolidate, then verify only one live fact exists
6. Add test: inject failure during edge insertion, verify database unchanged

### Verification

```bash
cd /home/sikmindz/Coding/Libraries/semantic-memory
cargo test --all-features -- consolidate
cd /home/sikmindz/Coding/Libraries/semantic-memory-mcp
cargo test --features full -- consolidate
```

### Estimated effort

2-3 hours. Requires core API addition and MCP rewiring.

---

## ITEM 6: Compile-time stable feature (cfg gates)

### Problem

`server.rs` unconditionally references modules behind `full` features. Building with `--no-default-features --features stable` produces 70+ errors. The `stable` feature in `Cargo.toml` only includes `usearch-backend`, `candle-embedder`, `provenance`, and `temporal` — but `server.rs` references `discord`, `routing`, `decoder`, `integration`, `late_interaction`, `turbo_quant`, `rl_routing`, `claim_ledger`, `llm_output_parser`, `projection_import`.

### Current cfg gates

server.rs already has 45 cfg gates:
- `claim-integration`: 31 gates
- `llm-parser`: 7 gates
- `full`: 5 gates
- `integration`: 1 gate
- `subgraph-pruning`: 1 gate

### Modules needing gates

| Module | References in server.rs | Feature flag |
|--------|------------------------|-------------|
| `discord` | 41 | `semantic-memory/discord` |
| `routing` | 72 | `semantic-memory/routing` |
| `decoder` | 27 | `semantic-memory/decoder` |
| `integration` | 51 | `semantic-memory/integration` |
| `turbo_quant` | 40 | `semantic-memory/turbo-quant-codec` |
| `rl_routing` | 6 | `semantic-memory/rl-routing` |
| `late_interaction` | (in HTTP server) | `semantic-memory/late-interaction` |
| `multiscale` | (references) | `semantic-memory/multiscale` |
| `subtraction` | (references) | `semantic-memory/subtraction` |
| `compression_governor` | (references) | `semantic-memory/compression-governor` |

### Fix approach

For each module reference in `server.rs` that is behind a `full` feature:

1. Find every `use semantic_memory::discord::*` or `semantic_memory::discord::` import
2. Wrap each in `#[cfg(feature = "semantic-memory/discord")]` or the local equivalent
3. Find every function that uses `discord` types and wrap the entire function in `#[cfg(feature = "discord")]`
4. Add a local `discord` feature forwarding in `Cargo.toml`: `discord = ["semantic-memory/discord"]`
5. Do the same for each module listed above

The `stable` feature should then be:
```toml
stable = [
    "semantic-memory/usearch-backend",
    "semantic-memory/candle-embedder",
    "semantic-memory/provenance",
    "semantic-memory/temporal",
    "candle-embedder",
]
```

And `server.rs` should compile with only those features because all other module references are cfg-gated out.

### Steps

1. Add local feature forwarders in `Cargo.toml` for each module:
   ```toml
   discord = ["semantic-memory/discord"]
   routing = ["semantic-memory/routing"]
   decoder = ["semantic-memory/decoder"]
   # ... etc
   ```
2. In `server.rs`, wrap every `use semantic_memory::discord::*` with `#[cfg(feature = "discord")]`
3. Wrap every function that uses `discord` types with `#[cfg(feature = "discord")]`
4. Repeat for each module
5. Wrap the HTTP server functions that use `late_interaction` or `turbo_quant`
6. Build with `--no-default-features --features stable` and fix remaining errors
7. Iterate until clean

### Pitfall

This is the largest single task. There are ~300 references across 10 modules. Each needs a cfg gate. The `tool_router()` macro registration also needs gating — tools behind features that aren't compiled must not be registered.

The `#[tool_router]` macro generates a `tool_router()` method that registers all `#[tool]`-annotated methods. If a method is cfg-gated out, it won't be registered. This should work automatically — but verify that the macro doesn't generate references to missing methods.

### Verification

```bash
cd /home/sikmindz/Coding/Libraries/semantic-memory-mcp
cargo check --no-default-features --features stable 2>&1 | tail -5
# Expected: "Finished" with no errors
cargo test --no-default-features --features stable 2>&1 | tail -5
# Expected: tests pass
cargo check --features full 2>&1 | tail -5
# Expected: still compiles with full
cargo test --features full 2>&1 | tail -5
# Expected: all tests pass
```

### Estimated effort

4-8 hours. Tedious but mechanical. Each module is independent. Can be done incrementally — gate one module at a time, build, fix errors, repeat.

---

## ITEM 7: Structured MCP outputs (outputSchema)

### Problem

All 76 tool functions return `Result<String, ErrorData>` where the String is JSON serialized via `json_to_string`. MCP supports `structuredContent` with `outputSchema` for protocol-level validation and typed client consumption.

### Current state

- 76 `json_to_string` calls in server.rs
- 0 `structuredContent` usages
- 0 `outputSchema` usages
- rmcp dependency: `rmcp = { version = "1", features = ["server", "transport-io"] }`

### Fix approach

This requires understanding rmcp's typed output API. The rmcp crate (v1) supports `#[tool(output_schema = ...)]` or similar. Check the rmcp documentation:

```bash
cd /home/sikmindz/Coding/Libraries/semantic-memory-mcp
grep -r "output" target/doc/rmcp/ 2>/dev/null || cargo doc --features full --open 2>/dev/null
```

For each tool function:

1. Define a typed output struct:
   ```rust
   #[derive(JsonSchema, Serialize)]
   pub struct SearchOutput {
       pub ok: bool,
       pub results: Vec<SearchResultJson>,
       pub count: usize,
       pub query: String,
       pub warnings: Vec<String>,
   }
   ```

2. Change the return type from `Result<String, ErrorData>` to `Result<SearchOutput, ErrorData>` (or `Result<StructuredToolOutput<SearchOutput>, ErrorData>` if rmcp requires a wrapper)

3. Remove the `json_to_string` call — rmcp will serialize the struct

4. Add `#[tool(output_schema = SearchOutput)]` or the rmcp-equivalent annotation

### Prioritization

Start with the most-used tools:
1. `sm_search` — most called tool by agents
2. `sm_add_fact` — second most called
3. `sm_search_witnessed` — used by autonomous profiles
4. `sm_stats` — simple output, good test case

Then batch the remaining 72 tools.

### Pitfall

This changes the wire format. Any client parsing the JSON string will need to adapt. The MCP `structuredContent` field is separate from the text result — rmcp may still send a text representation alongside the structured content. Check rmcp docs for backward compatibility.

### Verification

```bash
cargo check --features full
cargo test --features full
# Add a test that verifies structuredContent is present in the MCP response
```

### Estimated effort

6-12 hours. 76 tools, each needs a typed struct. Can be done incrementally.

---

## ITEM 8: Clean-clone reproducibility

### Problem

Both repos use `path = "../sibling"` dependencies. A clean clone of just the repo fails `cargo metadata` because siblings aren't present.

### Current path dependencies

**MCP Cargo.toml** (6 path deps):
- `../semantic-memory`, `../stack-ids`, `../claim-ledger`, `../llm-output-parser`, `../knowledge-runtime`, `../boundary-compiler`

**Core Cargo.toml** (9 path deps):
- `../stack-ids`, `../forge-memory-bridge`, `../boundary-compiler`, `../bitemporal-runtime`, `../turbo-quant`, `../quant-governor`, `../scr-runtime-compression`, `../poly-kv`, `../semantic-memory-forge`

### Fix approach

#### Option A: Publish to crates.io (ideal but high effort)

1. Publish each sibling crate to crates.io
2. Change `path = "../sibling"` to `version = "X.Y"` in Cargo.toml
3. Clean clone works

#### Option B: Git dependencies with local patch overrides

1. Add git dependencies pointing to the GitHub repos:
   ```toml
   semantic-memory = { git = "https://github.com/RecursiveIntell/semantic-memory", branch = "main" }
   ```
2. Keep `[patch]` overrides for local development:
   ```toml
   [patch."https://github.com/RecursiveIntell/semantic-memory"]
   semantic-memory = { path = "../semantic-memory" }
   ```
3. Clean clone uses git deps; local dev uses path overrides

#### Option C: Workspace root with all siblings

1. Create a `Cargo.toml` workspace root at `/home/sikmindz/Coding/Libraries/Cargo.toml` that includes all sibling crates as workspace members
2. Remove path deps from individual Cargo.tomls — the workspace resolver handles it
3. Clean clone = clone the workspace root

### Recommended: Option B

Least disruptive. Local dev still works with patch overrides. Clean clone resolves from git.

### Steps for Option B

1. For each path dep in MCP Cargo.toml:
   - Add a git dependency as the base
   - Add a `[patch]` section pointing to the local path
2. For each path dep in core Cargo.toml:
   - Same pattern
3. Test clean clone:
   ```bash
   cd /tmp && git clone https://github.com/RecursiveIntell/semantic-memory-mcp
   cd semantic-memory-mcp
   cargo metadata --format-version 1 --all-features
   # Should succeed
   ```
4. Test local dev still works:
   ```bash
   cd /home/sikmindz/Coding/Libraries/semantic-memory-mcp
   cargo check --features full
   # Should still work with patch overrides
   ```

### Pitfall

All sibling crates must be pushed to GitHub first. Some may not have public repos yet.

### Verification

```bash
# Clean clone test
cd /tmp && rm -rf test-clone && mkdir test-clone && cd test-clone
git clone https://github.com/RecursiveIntell/semantic-memory-mcp
cd semantic-memory-mcp
cargo metadata --format-version 1 --all-features
# Expected: succeeds
```

### Estimated effort

2-4 hours. Mostly mechanical, but requires all siblings to have GitHub repos.

---

## Execution order

Recommended priority by ROI:

| Priority | Item | Effort | Risk | Return |
|----------|------|--------|------|--------|
| 1 | ITEM 1: idempotent_hint | 15min | Zero | Protocol correctness |
| 2 | ITEM 2: as_of message | 2min | Zero | Contract honesty |
| 3 | ITEM 3: HTTP hardcoded fields | 30min | Low | Contract honesty |
| 4 | ITEM 5: Consolidation tx | 2-3hr | Medium | Data integrity |
| 5 | ITEM 4: Claim persistence | 2-4hr | Medium | Data integrity |
| 6 | ITEM 6: cfg gates | 4-8hr | Low | Compile-time containment |
| 7 | ITEM 8: Clean clone | 2-4hr | Low | Reproducibility |
| 8 | ITEM 7: Structured outputs | 6-12hr | Medium | Protocol conformance |

Items 1-3 are quick wins. Items 4-5 are the highest-impact remaining defects. Items 6-8 are infrastructure improvements.

---

## Test plan for each item

| Item | Test to add |
|------|------------|
| 1 | `grep -c idempotent_hint src/server.rs` returns 0 |
| 2 | Response message contains "PREVIEW" and "NOT applied" |
| 3 | HTTP routed-search response has `null` for unimplemented fields |
| 4 | Create claim → restart → claim still exists |
| 5 | Consolidate → verify one live fact → inject failure → DB unchanged |
| 6 | `cargo check --no-default-features --features stable` succeeds |
| 7 | MCP `tools/call` response includes `structuredContent` field |
| 8 | Clean clone `cargo metadata` succeeds without siblings present |