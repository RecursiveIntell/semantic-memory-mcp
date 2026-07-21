#!/usr/bin/env python3
"""
pre_tool_call hook for semantic-memory dedup guard.

Intercepts sm_add_fact and sm_ingest_document calls and checks if the
content already exists in the DB. If it does, blocks the call with a
message telling the agent to use the existing fact/document instead.

Uses the warm HTTP server when available (350ms), falls back to spawning
(1700ms). Fails open: any error -> no block, lets the call through.

Wire protocol (stdin JSON from Hermes shell-hooks bridge):
{
  "hook_event_name": "pre_tool_call",
  "tool_name": "sm_add_fact",
  "tool_input": {"content": "...", "namespace": "..."},
  "session_id": "...",
  "cwd": "...",
  "extra": {...}
}

Returns on stdout:
  {"decision": "block", "reason": "..."}  -> blocks the tool call
  {} or empty                            -> allows the tool call
"""
import json, os, re, subprocess, sys
from pathlib import Path

# Add the agent-hooks directory to the path for the HTTP client
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sm_http_client import search as http_search, http_available

BLOCK_MATCHER = re.compile(r"^sm_(add_fact|ingest_document)$")


def extract_content(payload):
    """Extract the content to check for duplicates from the tool input."""
    tool_name = payload.get("tool_name", "")
    tool_input = payload.get("tool_input", {})
    if not isinstance(tool_input, dict):
        return None, None, None
    content = tool_input.get("content") or tool_input.get("text") or ""
    namespace = tool_input.get("namespace", "general")
    title = tool_input.get("title", "")
    if not content and title:
        content = title
    if not content:
        return None, None, None
    return tool_name, content, namespace


def check_duplicate_http(content, namespace):
    """Check if content already exists via the warm HTTP server."""
    # Search using the first 200 chars as a fingerprint
    fingerprint = content[:200].strip()
    if len(fingerprint) < 10:
        return None

    results = http_search(fingerprint, top_k=3, namespaces=[namespace] if namespace else None, timeout=8)
    if not results:
        return None

    if isinstance(results, dict):
        results = results.get("results") or []
    if not isinstance(results, list):
        return None

    for r in results:
        if not isinstance(r, dict):
            continue
        existing = r.get("content", "")
        # Check if the existing content is substantially the same
        if existing and content_similarity(existing, content) > 0.85:
            return r

    return None


def content_similarity(a, b):
    """Quick Jaccard similarity on word sets."""
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def main():
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            return 0
        payload = json.loads(raw)
    except Exception:
        return 0

    tool_name, content, namespace = extract_content(payload)
    if not content or not BLOCK_MATCHER.match(tool_name or ""):
        return 0

    # Try HTTP first (warm server, ~350ms)
    if http_available():
        existing = check_duplicate_http(content, namespace or "general")
        if existing:
            print(json.dumps({
                "decision": "block",
                "reason": f"Duplicate content already exists in semantic memory "
                          f"(ID: {existing.get('id', 'unknown')}, namespace: {namespace}). "
                          f"Use the existing fact instead of adding a duplicate. "
                          f"Search for it with sm_search."
            }, separators=(",", ":")))
            return 0
    else:
        # Fall back to spawning a new MCP process (slow but correct)
        # Try a quick search via the binary
        binary = os.environ.get("SEMANTIC_MEMORY_MCP_BIN", "semantic-memory-mcp")
        which = subprocess.run(["which", binary], capture_output=True, text=True)
        if which.returncode != 0:
            return 0  # fail open

        memdir = os.environ.get("SEMANTIC_MEMORY_DB", os.path.expanduser("~/.hermes/semantic-memory.db"))
        fingerprint = content[:200].strip()
        try:
            proc = subprocess.run(
                [binary, "--memory-dir", memdir, "--embedder", "mock",
                 "--embedding-model", "test", "--embedding-dims", "768",
                 "--http-only", "--http-port", "0"],
                input=json.dumps({"jsonrpc": "2.0", "method": "tools/call",
                                  "params": {"name": "sm_search",
                                             "arguments": {"query": fingerprint, "top_k": 3}}},
                                stop=5)
            )
            # This fallback is unreliable -- if HTTP isn't available, just fail open
            # The HTTP server should always be running when Hermes is active
            return 0
        except Exception:
            return 0  # fail open

    return 0


if __name__ == "__main__":
    sys.exit(main())