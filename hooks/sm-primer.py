#!/usr/bin/env python3
"""
SessionStart hook for semantic-memory.

Primes the session with:
1. HTTP server health check (warns if warm server isn't running)
2. KB stats (fact count, edge count, namespace count)
3. Project-scoped recall — facts relevant to the current git repo or cwd

Uses the warm HTTP server when available, falls back to spawning.

Fails open: any error -> exit 0, no output.
"""
import json, os, subprocess, sys
from pathlib import Path

# Add the agent-hooks directory to the path so we can import the HTTP client
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sm_http_client import stats as http_stats, search as http_search, http_available

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
            "instead of ~350ms). Start Hermes with the semantic_memory MCP server "
            "configured to use --http-port 1738 for best performance.",
            file=sys.stderr)
    else:
        # Integrity check: verify DB integrity on session start
        try:
            import urllib.request
            with urllib.request.urlopen("http://127.0.0.1:1738/verify-integrity", timeout=3) as r:
                integrity = json.loads(r.read())
                if not integrity.get("integrity", False):
                    print(f"WARNING: semantic-memory integrity check FAILED: {integrity.get('checks', {})}", file=sys.stderr)
        except Exception:
            pass  # Non-fatal -- don't block session start on integrity check
    
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