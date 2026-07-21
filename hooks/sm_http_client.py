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
DEFAULT_POOLED_MEMORY_URL = "http://127.0.0.1:1748"
DEFAULT_TIMEOUT = 10


def pooled_url():
    return os.environ.get("POOLED_MEMORY_URL", DEFAULT_POOLED_MEMORY_URL).rstrip("/")


def _pooled_json(path, payload=None, method="POST", timeout=DEFAULT_TIMEOUT):
    data = None if method == "GET" else json.dumps(payload or {}).encode()
    req = Request(
        f"{pooled_url()}{path}", data=data, method=method,
        headers={"Content-Type": "application/json", **_pooled_auth_headers()},
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def _pooled_auth_headers():
    credential = os.environ.get("POOLED_MEMORY_CREDENTIAL", "")
    env_file = Path(os.environ.get("POOLED_MEMORY_ENV_FILE", str(Path.home() / ".config/pooled-memory/client.env")))
    if not credential:
        try:
            for line in env_file.read_text().splitlines():
                if line.startswith("POOLED_MEMORY_CREDENTIAL="):
                    credential = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
        except Exception:
            pass
    return {"Authorization": f"Bearer {credential}"} if credential else {}


def _pooled_mcp(name, arguments, timeout=DEFAULT_TIMEOUT):
    """Call pooled-memory MCP tool through the pooled proxy (handles auth)."""
    import subprocess, tempfile
    payload = json.dumps({
        "jsonrpc": "2.0", "id": "hermes-sm-hook",
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments}
    })
    try:
        proc = subprocess.run(
            ["python3", str(Path.home() / ".local/bin/pooled-memory-mcp-proxy.py")],
            input=payload + "\n", capture_output=True, text=True,
            timeout=timeout,
            env={**os.environ, "PATH": str(Path.home() / ".local/bin") + ":" + os.environ.get("PATH", "")}
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return None
        result = json.loads(proc.stdout.strip())
        if "result" not in result:
            return None
        return result["result"]
    except Exception:
        return None


def pooled_available(timeout=2):
    return pooled_health(timeout=timeout) is not None


def pooled_health(timeout=2):
    """Return pooled health, falling back to the local HTTP health probe."""
    result = _pooled_json("/v1/health", method="GET", timeout=timeout)
    if result is not None:
        return result
    return {"ok": http_available()} if http_available() else None


def pooled_search(query, limit=5, namespaces=None, timeout=DEFAULT_TIMEOUT, top_k=None):
    """Search pooled memory first; flatten its witnessed response envelope."""
    if top_k is not None:
        limit = top_k
    payload = {"query": query, "limit": limit}
    if namespaces:
        payload["namespaces"] = namespaces
    result = _pooled_mcp("sm_search_witnessed", payload, timeout=timeout)
    if result is not None:
        return result.get("results", result) if isinstance(result, dict) else result
    return search(query, top_k=limit, namespaces=namespaces, timeout=timeout)


def pooled_stats(timeout=5):
    """Return pooled stats, falling back to local stats."""
    result = _pooled_mcp("sm_stats", {}, timeout=timeout)
    return result if result is not None else stats(timeout=timeout)


def get_http_port():
    return int(os.environ.get("SEMANTIC_MEMORY_HTTP_PORT", str(DEFAULT_PORT)))

def http_available(port=None):
    """Check if the warm HTTP server is running."""
    port = port or get_http_port()
    try:
        headers = headers_for("/health")
        req = Request(f"http://127.0.0.1:{port}/health", headers=headers)
        with urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read())
            return data.get("ok", False)
    except Exception:
        return False

ADMIN_ENDPOINTS = {"/stats", "/search-routed", "/maintenance/check", "/rerank", "/discord", "/add", "/delete", "/supersede", "/reembed", "/vacuum", "/reconcile", "/compact-claim-ledger", "/run-lifecycle", "/list-facts"}

def admin_token():
    """Read the admin auth token. Priority: SM_HTTP_ADMIN_TOKEN env var, then token file."""
    env_token = os.environ.get("SM_HTTP_ADMIN_TOKEN")
    if env_token:
        return env_token
    try:
        return Path(os.environ.get("SEMANTIC_MEMORY_HTTP_TOKEN_FILE", "/home/sikmindz/.hermes/semantic-memory-http-admin.token")).read_text().strip()
    except Exception:
        return ""

def headers_for(path):
    """Headers for an HTTP request. /health needs auth on the full-profile server."""
    headers = {"Content-Type": "application/json"}
    token = admin_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers

def http_json(path, payload=None, method="POST", port=None, timeout=DEFAULT_TIMEOUT):
    """Generic HTTP JSON request to the warm semantic-memory server."""
    port = port or get_http_port()
    data = None if method == "GET" else json.dumps(payload or {}).encode()
    req = Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        method=method,
        headers=headers_for(path),
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception:
        return None

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
        headers=headers_for("/search"),
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
        headers=headers_for("/stats"),
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
        headers=headers_for("/stats"),
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
        headers=headers_for("/stats"),
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
    """Search semantic memory, pooled-first with the existing local fallback."""
    pooled = _pooled_json("/search", {"query": query, "limit": top_k, **({"namespaces": namespaces} if namespaces else {})}, timeout=timeout)
    if pooled is not None:
        return pooled.get("results", pooled) if isinstance(pooled, dict) else pooled
    if http_available():
        result = http_search(query, top_k=top_k, namespaces=namespaces, timeout=timeout)
        if result is not None:
            return result
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
    """Get DB stats, pooled-first with the existing local fallback."""
    pooled = _pooled_json("/stats", {}, timeout=timeout)
    if pooled is not None:
        return pooled
    if http_available():
        result = http_stats(timeout=timeout)
        if result is not None:
            return result
    return rpc_call_fallback("sm_stats", {}, timeout=timeout + 10)

def add_fact(content, namespace="general", source=None, timeout=DEFAULT_TIMEOUT):
    """Add a fact. Tries warm HTTP first, falls back to spawn."""
    if http_available():
        return http_add_fact(content, namespace=namespace, source=source, timeout=timeout)
    args = {"content": content, "namespace": namespace}
    if source:
        args["source"] = source
    return rpc_call_fallback("sm_add_fact", args, timeout=timeout + 10)


def http_record_outcome(query, outcome="good", query_class="A", timeout=3):
    """Record a routing feedback outcome via the warm HTTP server.

    Calls the sm_record_outcome MCP tool through the HTTP JSON-RPC endpoint.
    This is a fire-and-forget telemetry call — failures are silently ignored
    so they never block the recall path.
    """
    try:
        port = get_http_port()
        if not http_available(port=port):
            return None
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "sm_record_outcome",
                "arguments": {
                    "query": query[:200],
                    "outcome": outcome,
                },
            },
        }
        return http_json("/v1/mcp", payload=payload, method="POST", port=port, timeout=timeout)
    except Exception:
        return None