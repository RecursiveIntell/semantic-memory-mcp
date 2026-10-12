# @recursiveintell/semantic-memory-mcp

A Node launcher for a **separate Rust executable**, not the Rust server itself. The npm version does not identify the installed Rust generation or certify a tool inventory.

## Executable selection

The default verified asset is currently Linux x64 only:

- GitHub release tag: `v1.1.0`
- Executable's reported version: `semantic-memory-mcp 0.5.6`
- SHA-256: `5089f22f1e0f3f423b9086b018036c509b74579ff49bc844dab05062b8c629a3`
- Source: https://github.com/RecursiveIntell/semantic-memory-mcp/releases/tag/v1.1.0

These are distinct from the current Rust source crate's `0.5.8` version. Do not infer current-source capabilities from this older executable. The digest was matched against the existing GitHub release API asset digest and downloaded bytes; this is not an independent build reproduction or security certification.

**Observed identity limitation:** in an isolated `--embedder mock --tool-profile agent` check, this executable reported CLI version `0.5.6` but MCP `serverInfo.version` `0.5.4`. Both values are retained; the launcher does not rewrite server metadata. That configuration listed 11 tools and successfully returned `sm_stats` for an empty fixture store. This is protocol/read-operation evidence, not real-embedding, populated-store, provider, current-source-parity or cross-platform qualification. The cause of the version disagreement has not been established.

The launcher requires HTTPS, checks HTTP status and the complete download digest, verifies the executable version, and atomically installs into a platform/version/digest-specific cache below `~/.cache/recursiveintell/`. Downloads have time and size bounds. Cache bytes are rechecked on every launch. It does not execute a partial download or silently fall back to an arbitrary PATH binary or unpinned Cargo installation.

Unsupported platforms fail explicitly. Linux x64 selection is not a promise that every Linux distribution satisfies the Rust binary's runtime dependencies. Other platforms remain unqualified.

## Explicit operator-admitted executable

For a different executable, supply all three non-secret admission values:

```bash
export SEMANTIC_MEMORY_MCP_BINARY=/absolute/path/to/semantic-memory-mcp
export SEMANTIC_MEMORY_MCP_SHA256=the_independently_verified_64_character_lowercase_digest
export SEMANTIC_MEMORY_MCP_VERSION=the_exact_reported_Rust_version
```

The placeholders above must be replaced with actual independently verified values. A symlink, relative path, malformed value, digest mismatch or version mismatch is rejected. Merely computing a hash of an unknown binary does not establish publisher provenance. Version probing executes only after digest admission. These checks are not an OS sandbox and do not prevent an authorized filesystem owner from changing files concurrently.

The server still needs its documented arguments, including an explicit memory directory. The launcher forwards arguments without shell interpolation. Provider credentials and server configuration remain the caller's responsibility; launching the server does not authorize remote model calls or certify provider availability.

## Failures and validation

Download/admission errors fail nonzero with explicit reasons. Child exit status and termination signals are preserved; operator termination is forwarded with bounded escalation. Importing the resolver module does not start a download or server.

```bash
node --test test/launcher.test.js
npm pack --dry-run --json
```

The regression tests use local executable and loopback HTTP fixtures. They test selection and failure boundaries, not real registry installation, model quality, cross-platform support or current-source server capabilities. A packed README is not successful onboarding. Source, tarball, binary and actual negotiated MCP inventory must be bound separately before a release/onboarding claim.

Wrapper license declaration: Apache-2.0. The separately downloaded executable remains subject to its own source/release terms.
