#!/usr/bin/env python3
"""Quick health check for the semantic-memory HTTP server."""
import json, sys, urllib.request

BASE = "http://127.0.0.1:1738"

def health():
    try:
        with urllib.request.urlopen(f"{BASE}/health", timeout=3) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"ok": False, "error": str(e)}

def search(query, top_k=5, namespaces=None):
    payload = {"query": query, "top_k": top_k}
    if namespaces:
        payload["namespaces"] = namespaces
    body = json.dumps(payload).encode()
    req = urllib.request.Request(f"{BASE}/search", data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"ok": False, "error": str(e)}

if __name__ == "__main__":
    h = health()
    print(f"Health: {h}")
    if h.get("ok"):
        r = search("turbo-quant", top_k=3)
        for result in r.get("results", [])[:3]:
            ns = result.get("namespace", "?")
            score = float(result.get("score", 0))
            content = result.get("content", "")[:80]
            print(f"  [{score:.4f}] [{ns}] {content}")