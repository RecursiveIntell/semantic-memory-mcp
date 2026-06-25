#!/usr/bin/env python3
"""
HTTP client for semantic-memory hooks.

Instead of spawning a new semantic-memory-mcp process per hook invocation
(~1.2s overhead just for model loading), this client queries the warm
HTTP server that runs alongside the main MCP stdio process.

The HTTP server is started automatically by the MCP server when
--http-port is passed (configured in ~/.hermes/config.yaml).

Falls back to spawning a new process if the HTTP server is not available.
"""
import json, os, subprocess, sys
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError

DEFAULT_PORT = 1738
DEFAULT_TIMEOUT = 10

def get_http_port():
    return int(os.environ.get("SEMANTIC_MEMORY_HTTP_PORT", str(DEFAULT_PORT)))

def http_available(port=None):
    """Check if the warm HTTP server is running."""
    port = port or get_http_port()
    try:
        req = Request(f"http://127.0.0.1:{port}/health")
        with urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read())
            return data.get("ok", False)
    except Exception:
        return False

def http_search(query, top_k=5, namespaces=None, port=None, timeout=DEFAULT_TIMEOUT):
    """Search via the warm HTTP server. Returns None on failure."""
    port = port or get_http_port()
    payload = {"query": query, "top_k": top_k}
    if namespaces:
        payload["namespaces"] = namespaces
    body = json.dumps(payload).encode()
    req = Request(
        f"http://127.0.0.1:{port}/search",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception:
        return None

def http_search_routed(query, top_k=12, query_class="A", namespaces=None, port=None, timeout=DEFAULT_TIMEOUT):
    """Routing-aware search via the warm HTTP server. Returns None on failure."""
    port = port or get_http_port()
    payload = {"query": query, "top_k": top_k, "query_class": query_class}
    if namespaces:
        payload["namespaces"] = namespaces
    body = json.dumps(payload).encode()
    req = Request(
        f"http://127.0.0.1:{port}/search-routed",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception:
        return None

def http_stats(port=None, timeout=5):
    """Get DB stats via the warm HTTP server."""
    port = port or get_http_port()
    req = Request(
        f"http://127.0.0.1:{port}/stats",
        data=b"{}",
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception:
        return None

def http_add_fact(content, namespace="general", source=None, port=None, timeout=DEFAULT_TIMEOUT):
    """Add a fact via the warm HTTP server."""
    port = port or get_http_port()
    payload = {"content": content, "namespace": namespace}
    if source:
        payload["source"] = source
    body = json.dumps(payload).encode()
    req = Request(
        f"http://127.0.0.1:{port}/add",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception:
        return None

# ── Fallback: spawn a new MCP process (slow, ~1.2s overhead) ────────────

def resolve_binary():
    env = os.environ.get("SEMANTIC_MEMORY_MCP_BIN")
    if env and os.access(os.path.expanduser(env), os.X_OK):
        return os.path.expanduser(env)
    which = subprocess.run(["which", "semantic-memory-mcp"], capture_output=True, text=True)
    if which.returncode == 0 and which.stdout.strip():
        return which.stdout.strip()
    cargo = Path.home() / ".cargo/bin/semantic-memory-mcp"
    if cargo.exists() and os.access(cargo, os.X_OK):
        return str(cargo)
    return None

def memory_dir():
    return os.environ.get("SEMANTIC_MEMORY_DIR",
                          str(Path.home() / ".hermes/semantic-memory.db"))

def rpc_call_fallback(tool, arguments, timeout=15):
    """Spawn a new MCP process for a single RPC call (fallback)."""
    binary = resolve_binary()
    if not binary:
        return None
    memdir = memory_dir()
    reqs = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                    "clientInfo": {"name": "hermes-sm-hook", "version": "1"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
         "params": {"name": tool, "arguments": arguments}},
    ]
    stdin = "\n".join(json.dumps(r) for r in reqs) + "\n"
    try:
        proc = subprocess.run(
            [binary, "--memory-dir", memdir, "--embedder", "candle"],
            input=stdin, text=True, capture_output=True, timeout=timeout, check=False)
    except Exception:
        return None
    for line in proc.stdout.splitlines():
        try:
            msg = json.loads(line)
        except Exception:
            continue
        if msg.get("id") != 2:
            continue
        try:
            return json.loads(msg["result"]["content"][0]["text"])
        except Exception:
            return None
    return None

# ── Unified API: try HTTP first, fall back to spawn ─────────────────────

def search(query, top_k=5, namespaces=None, timeout=DEFAULT_TIMEOUT):
    """Search semantic memory. Tries warm HTTP first, falls back to spawn."""
    if http_available():
        return http_search(query, top_k=top_k, namespaces=namespaces, timeout=timeout)
    # Fallback: spawn a new process
    args = {"query": query, "top_k": top_k}
    if namespaces:
        args["namespaces"] = namespaces
    return rpc_call_fallback("sm_search", args, timeout=timeout + 10)

def search_routed(query, top_k=12, query_class="A", namespaces=None, timeout=DEFAULT_TIMEOUT):
    """Routing-aware search. Uses /search-routed when HTTP is available, falls back to plain search."""
    if http_available():
        result = http_search_routed(query, top_k=top_k, query_class=query_class,
                                    namespaces=namespaces, timeout=timeout)
        if result is not None:
            return result
    return search(query, top_k=top_k, namespaces=namespaces, timeout=timeout)

def stats(timeout=5):
    """Get DB stats. Tries warm HTTP first, falls back to spawn."""
    if http_available():
        return http_stats(timeout=timeout)
    return rpc_call_fallback("sm_stats", {}, timeout=timeout + 10)

def add_fact(content, namespace="general", source=None, timeout=DEFAULT_TIMEOUT):
    """Add a fact. Tries warm HTTP first, falls back to spawn."""
    if http_available():
        return http_add_fact(content, namespace=namespace, source=source, timeout=timeout)
    args = {"content": content, "namespace": namespace}
    if source:
        args["source"] = source
    return rpc_call_fallback("sm_add_fact", args, timeout=timeout + 10)