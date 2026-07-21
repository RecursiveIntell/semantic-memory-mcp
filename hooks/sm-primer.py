#!/usr/bin/env python3
"""
SessionStart hook for semantic-memory.

Primes the session with:
1. HTTP server health check (warns if warm server isn't running)
2. Read-only maintenance check: /maintenance/check for integrity + embedding health
   - If embeddings are dirty: warns (re-embed is expensive, don't auto-run)
   - If integrity fails: warns loudly; repair is left to maintenance cron/manual operator action
   - If integrity is fine but DB is large: suggests vacuum
3. KB stats (fact count, edge count, namespace count)
4. Project-scoped recall — facts relevant to the current git repo or cwd

Uses the warm HTTP server when available, falls back to spawning.
The session-start primer is intentionally read-only. Maintenance runs via HTTP endpoints (not MCP tools)
from guarded scripts/cron so it works
regardless of the --tool-profile setting.

Fails open: any error -> exit 0, no output.
"""
import json, os, subprocess, sys
from pathlib import Path
from urllib.request import Request, urlopen

# Add the agent-hooks directory to the path so we can import the HTTP client
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sm_http_client import stats as http_stats, search as http_search, http_available, get_http_port

def http_post(path, body=None, timeout=10):
    """POST to a maintenance endpoint. Returns parsed JSON or None."""
    port = get_http_port()
    data = json.dumps(body or {}).encode()
    req = Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception:
        return None

def http_get(path, timeout=10):
    """GET from an endpoint. Returns parsed JSON or None."""
    port = get_http_port()
    req = Request(f"http://127.0.0.1:{port}{path}")
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception:
        return None

def run_auto_management():
    """Check DB health without mutating the store. Returns warning strings."""
    warnings = []

    # Combined health check: embeddings_are_dirty + verify_integrity
    health = http_post("/maintenance/check", {}, timeout=15)
    if not health or not health.get("ok"):
        # /maintenance/check might not exist on older binaries -- fall back to /verify-integrity
        health = http_get("/verify-integrity", timeout=10)
        if not health:
            return []  # fail open, can't check
        health = {"ok": health.get("ok"), "integrity": health.get("integrity", True),
                  "embeddings_dirty": None, "issues": health.get("issues", [])}

    # Integrity check
    if not health.get("integrity", True):
        issues = health.get("issues", [])
        warnings.append(
            f"DB INTEGRITY CHECK FAILED: {len(issues)} issue(s): "
            + "; ".join(str(i) for i in issues[:3])
        )
        warnings.append(
            "Session-start primer is read-only; run guarded maintenance to reconcile."
        )

    # Embedding health
    dirty = health.get("embeddings_dirty")
    if dirty is True:
        warnings.append(
            "WARNING: Embeddings are DIRTY (some facts lack embeddings). "
            "Search quality may be degraded. Run /maintenance/reembed to fix "
            "(expensive: ~138ms per fact on CPU)."
        )
        # Don't auto-reembed -- too expensive for session start. Just warn.

    # WAL checkpoint — prevents WAL from growing unboundedly between sessions.
    # The semantic-memory-mcp server uses WAL mode, and with multiple connections
    # the WAL can grow to hundreds of MB without checkpointing. This is a cheap
    # PASSIVE checkpoint (doesn't block readers, just merges committed pages).
    try:
        import sqlite3
        db_path = str(Path.home() / ".hermes" / "semantic-memory.db" / "memory.db")
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path, timeout=5)
            conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
            conn.close()
    except Exception:
        pass  # Checkpoint is best-effort — don't block session start

    return warnings

def project_name(cwd):
    path = Path(cwd or ".")
    try:
        proc = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            text=True, capture_output=True, timeout=2, check=False)
        root = Path(proc.stdout.strip()) if proc.returncode == 0 and proc.stdout.strip() else None
    except Exception:
        root = None
    if root and root != Path.home():
        return root.name, True
    return path.name or "workspace", False

def main():
    payload = {}
    try:
        payload = json.load(sys.stdin)
    except Exception:
        pass
    cwd = payload.get("cwd") or str(Path.cwd())

    # Health check: warn if warm HTTP server isn't running
    if not http_available():
        print(
            "WARNING: semantic-memory HTTP server not running on port 1738. "
            "Hooks will fall back to spawning new MCP processes (~1.7s per call "
            "instead of ~350ms). Start the standalone semantic-memory-http "
            "user service; stdio MCP should stay lean and should not own "
            "--http-port 1738.",
            file=sys.stderr)
    else:
        # Read-only maintenance check via HTTP (works regardless of tool profile)
        warnings = run_auto_management()
        for w in warnings:
            print(f"semantic-memory: {w}", file=sys.stderr)

    # Get stats via HTTP (warm) or fallback
    stats_data = http_stats(timeout=5)
    if not stats_data or not stats_data.get("ok"):
        return 0

    lines = [
        f"Persistent semantic memory is ACTIVE: {stats_data.get('facts', 0)} facts, "
        f"{stats_data.get('documents', 0)} docs, {stats_data.get('chunks', 0)} chunks. "
        f"This is your primary long-term memory across all projects and sessions."
    ]

    # Check for stale DB -- warn if no recent activity
    try:
        import sqlite3
        db_path = os.path.expanduser("~/.hermes/semantic-memory.db/memory.db")
        if os.path.exists(db_path):
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            latest = conn.execute("SELECT created_at FROM facts ORDER BY created_at DESC LIMIT 1").fetchone()
            conn.close()
            if latest and latest[0]:
                from datetime import datetime, timezone
                latest_dt = datetime.fromisoformat(latest[0].replace('Z', '+00:00'))
                age_days = (datetime.now(timezone.utc) - latest_dt).days
                if age_days > 7:
                    lines.append(f"WARNING: Last fact added {age_days} days ago. Hooks may not be capturing correctly.")
    except Exception:
        pass

    proj, do_proj = project_name(cwd)
    # For git repos: search with project name + "codebase project overview"
    # For non-git dirs: fallback to directory name search with lower threshold
    if do_proj:
        query = f"{proj} codebase project overview"
    else:
        query = f"{proj} project config hooks tools"

    # Search via HTTP (warm) or fallback
    # Use priority namespaces to avoid social media noise
    PRIORITY_NAMESPACES = ["projects", "research", "semantic-memory", "libraries",
                           "libraries-crates", "doctrine", "agent-setup", "infrastructure",
                           "personal", "behavioral", "codex", "recursiveintell", "general"]
    result = http_search(query, top_k=10, namespaces=PRIORITY_NAMESPACES, timeout=8)
    if result and result.get("ok"):
        hits = result.get("results") or []
        # Dedup by content
        seen = set()
        hits = [h for h in hits if not (h.get("content","")[:80] in seen or seen.add(h.get("content","")[:80]))]
        if hits:
            label = f"Project-scoped recall for {proj}" if do_proj else f"Context recall for {proj}"
            lines.append(f"\n{label} (verify against current artifacts):")
            for h in hits[:3]:
                c = " ".join(str(h.get("content") or "").split())
                if len(c) > 300:
                    c = c[:299] + "..."
                if c:
                    lines.append(f"- {c}")

    lines.extend([
        "\n- RECALL: search semantic memory (sm_search) before starting substantial work.",
        "- PERSIST: store durable verified facts with sm_add_fact after dedupe.",
        "- DISCIPLINE: never let stored memory outrank current artifacts; "
        "record corrections by append/supersede.",
    ])

    print(json.dumps({"context": "\n".join(lines)}, separators=(",", ":")))
    return 0

if __name__ == "__main__":
    sys.exit(main())