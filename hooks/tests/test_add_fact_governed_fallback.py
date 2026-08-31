"""Behavior contract: governed add_fact falls through dead HTTP admission.

The warm HTTP /add endpoint is fail-closed by design (503 body with ok=false).
add_fact must treat a non-ok body as a failure and reach the governed MCP
write path instead of returning the failure to the caller.
"""
import importlib.util
import json
import sys
from pathlib import Path

HOOK_DIR = Path(__file__).resolve().parent.parent
MOD_PATH = HOOK_DIR / "sm_http_client.py"


def load_module():
    spec = importlib.util.spec_from_file_location("sm_http_client_fallback_test", MOD_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_add_fact_falls_through_non_ok_http_body_to_mcp(monkeypatch):
    mod = load_module()
    calls = []

    def fake_http_add(content, namespace="general", source=None, port=None, timeout=10):
        calls.append("http")
        return {"ok": False, "error": "HTTP evidence admission is disabled: ..."}

    def fake_mcp_http_call(tool, arguments, timeout=10):
        calls.append(("mcp", tool, arguments["namespace"]))
        return {"ok": True, "fact_id": "test-1"}

    monkeypatch.setattr(mod, "http_add_fact", fake_http_add)
    monkeypatch.setattr(mod, "mcp_http_call", fake_mcp_http_call)

    result = mod.add_fact("behavior contract probe", namespace="contract-ns")

    assert result == {"ok": True, "fact_id": "test-1"}
    assert calls[0] == "http"
    assert calls[1] == ("mcp", "sm_add_fact", "contract-ns")


def test_add_fact_ok_http_body_short_circuits(monkeypatch):
    mod = load_module()

    def fake_http_add(content, namespace="general", source=None, port=None, timeout=10):
        return {"ok": True, "fact_id": "direct-1"}

    def fail_mcp(*a, **k):
        raise AssertionError("mcp path must not be reached when HTTP write succeeds")

    monkeypatch.setattr(mod, "http_add_fact", fake_http_add)
    monkeypatch.setattr(mod, "mcp_http_call", fail_mcp)

    assert mod.add_fact("direct probe")["fact_id"] == "direct-1"


def test_add_fact_passes_source_through_fallback(monkeypatch):
    mod = load_module()
    seen = {}

    def fake_http_add(content, namespace="general", source=None, port=None, timeout=10):
        return {"ok": False}

    def fake_mcp(tool, arguments, timeout=10):
        seen.update(arguments)
        return {"ok": True, "fact_id": "s"}

    monkeypatch.setattr(mod, "http_add_fact", fake_http_add)
    monkeypatch.setattr(mod, "mcp_http_call", fake_mcp)

    mod.add_fact("probe", namespace="ns-x", source="unit-test")
    assert seen["source"] == "unit-test"
    assert seen["namespace"] == "ns-x"


def test_mcp_http_call_env_overrides_select_store(monkeypatch, tmp_path):
    """Port + token env overrides must drive which warm server is addressed."""
    mod = load_module()
    token_file = tmp_path / "bots-mcp.token"
    token_file.write_text("bots-token-123\n")
    monkeypatch.setenv("SEMANTIC_MEMORY_MCP_HTTP_PORT", "1759")
    monkeypatch.setenv("SEMANTIC_MEMORY_MCP_TOKEN_FILE", str(token_file))

    assert mod.mcp_http_token() == "bots-token-123"

    captured = {}

    class FakeResponse:
        def __init__(self, body, status=200, sid=None):
            self._body, self.status, self._sid = body, status, sid

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def getheader(self, name):
            return self._sid if name == "MCP-Session-Id" else None

        headers = property(lambda self: {"MCP-Session-Id": self._sid})

    bodies = iter([
        json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}}).encode(),  # initialize
        b"",  # initialized notification (202, empty)
        json.dumps({"jsonrpc": "2.0", "id": 2, "result": {"content": [
            {"type": "text", "text": json.dumps({"ok": True, "fact_id": "x"})}
        ]}}).encode(),  # tools/call
    ])
    urls = []

    def fake_urlopen(request, timeout=None):
        urls.append(request.full_url)
        return FakeResponse(next(bodies))

    monkeypatch.setattr(mod, "urlopen", fake_urlopen)
    result = mod.mcp_http_call("sm_add_fact", {"content": "c", "namespace": "n"})

    assert result == {"ok": True, "fact_id": "x"}
    assert urls and urls[0] == "http://127.0.0.1:1759/mcp"
