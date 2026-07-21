import importlib.util
import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError

HOOK_DIR = Path(__file__).resolve().parent.parent
MOD_PATH = HOOK_DIR / "sm_http_client.py"


def load_module():
    spec = importlib.util.spec_from_file_location("sm_http_client_under_test", MOD_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_admin_headers_include_bearer_token(monkeypatch):
    mod = load_module()
    monkeypatch.setenv("SM_HTTP_ADMIN_TOKEN", "tok-test")
    headers = mod.headers_for("/record-outcome")
    assert headers["Authorization"] == "Bearer tok-test"


def test_authenticated_service_headers_include_bearer_token_for_read_routes(monkeypatch):
    mod = load_module()
    monkeypatch.setenv("SM_HTTP_ADMIN_TOKEN", "tok-test")
    headers = mod.headers_for("/search")
    assert headers["Authorization"] == "Bearer tok-test"


def test_add_fact_uses_admin_endpoint(monkeypatch):
    mod = load_module()
    monkeypatch.setenv("SM_HTTP_ADMIN_TOKEN", "tok-test")

    class Response:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b'{"ok": true, "fact_id": "f1"}'

    captured = {}
    def fake_urlopen(req, timeout=10):
        captured["url"] = req.full_url
        captured["data"] = req.data
        captured["headers"] = dict(req.headers)
        return Response()

    monkeypatch.setattr(mod, "urlopen", fake_urlopen)
    monkeypatch.setattr(mod, "http_available", lambda port=None: True)
    result = mod.http_add_fact("durable fact", namespace="general")
    assert result["ok"] is True
    assert "/add" in captured["url"]
    assert b'"content": "durable fact"' in captured["data"]
    assert b'"namespace": "general"' in captured["data"]


def test_record_outcome_uses_admin_endpoint(monkeypatch):
    mod = load_module()
    calls = []

    def fake_http_json(path, payload=None, method="POST", port=None, timeout=10):
        calls.append((path, payload))
        return {"ok": True}

    monkeypatch.setattr(mod, "http_json", fake_http_json)
    monkeypatch.setattr(mod, "http_available", lambda port=None: True)
    result = mod.http_record_outcome("query", outcome="good", query_class="A")
    assert result["ok"] is True
    # http_record_outcome calls /v1/mcp with sm_record_outcome MCP tool
    assert len(calls) == 1
    assert calls[0][0] == "/v1/mcp"
    assert calls[0][1]["params"]["name"] == "sm_record_outcome"
    assert calls[0][1]["params"]["arguments"]["query"] == "query"
    assert calls[0][1]["params"]["arguments"]["outcome"] == "good"


def test_http_available_authenticates_health_probe(monkeypatch):
    mod = load_module()
    monkeypatch.setenv("SM_HTTP_ADMIN_TOKEN", "tok-test")
    seen = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"ok": true}'

    def fake_urlopen(req, timeout=2):
        seen["authorization"] = req.headers.get("Authorization")
        seen["url"] = req.full_url
        return Response()

    monkeypatch.setattr(mod, "urlopen", fake_urlopen)
    assert mod.http_available(port=1738) is True
    assert seen == {
        "authorization": "Bearer tok-test",
        "url": "http://127.0.0.1:1738/health",
    }
