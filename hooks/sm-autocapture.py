#!/usr/bin/env python3
"""
post_llm_call hook for semantic-memory auto-capture.

After each turn completes, scans the user message + assistant response for
durable facts worth persisting. Uses lightweight heuristics to filter out
ephemeral conversation, tool output, and questions. Only captures clear
declarative statements about:

- User preferences and corrections
- Project/config facts (paths, versions, decisions)
- Technical discoveries (workarounds, gotchas, architecture)

Fails open: any error -> exit 0, no output.

Wire protocol (stdin JSON from Hermes shell-hooks bridge):
{
  "hook_event_name": "post_llm_call",
  "session_id": "...",
  "cwd": "/home/user",
  "extra": {
    "user_message": "...",
    "assistant_response": "...",
    "conversation_history": [...],
    "session_id": "...",
    "model": "...",
    "platform": "cli"
  }
}
"""
import json, os, re, subprocess, sys
from pathlib import Path

# Add the agent-hooks directory to the path for the HTTP client
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sm_http_client import search as http_search, add_fact as http_add_fact, http_available

# ── Config ──────────────────────────────────────────────────────────────

# Min confidence to even consider capturing
MIN_INTEREST_SIGNAL = 3

# Max facts to capture per turn
MAX_CAPTURES_PER_TURN = 2

# Max content length per fact
MAX_FACT_LEN = 500

# ── Helpers ─────────────────────────────────────────────────────────────

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

def rpc_call(binary, memdir, tool, arguments, timeout=10):
    reqs = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                    "clientInfo": {"name": "hermes-sm-autocapture", "version": "1"}}},
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

# ── Fact extraction heuristics ──────────────────────────────────────────

# Patterns that signal a durable fact worth capturing
DURABLE_PATTERNS = [
    (r"\bprefer(?:s|red|ring)?\b", "preference"),
    (r"\balways\b.*\b(not|never|use|avoid)\b", "preference"),
    (r"\bnever\b.*\b(use|do|run)\b", "preference"),
    (r"\bdon'?t\b.*\b(use|do|run)\b", "preference"),
    (r"\bcorrection\b|\bactually\b.*\bnot\b", "correction"),
    (r"\bby default\b|\bthe default\b", "config"),
    (r"\bconfig(?:ured|uration)?\b.*\b(?:is|to|at)\b", "config"),
    (r"\bversion\b.*\d+\.\d+", "version"),
    (r"\bpath\b.*(?:/|~)", "path"),
    (r"\binstalled\b.*(?:at|in|to)\b", "install"),
    (r"\bworkaround\b|\bfix(?:ed)?\b.*\bby\b", "workaround"),
    (r"\bgotcha\b|\bpitfall\b|\bcaveat\b", "gotcha"),
    (r"\bdecided\b.*\bto\b", "decision"),
    (r"\barchitecture\b.*\b(?:is|uses)\b", "architecture"),
    (r"\bthe\s+\w+\s+(?:project|repo|crate)\b", "project"),
    (r"\bcargo\b.*\b(?:add|feature|toml)\b", "rust_config"),
    (r"\bmodel\b.*\b(?:is|uses|provider)\b", "model_config"),
]

# Patterns that indicate ephemeral/non-durable content
SKIP_PATTERNS = [
    r"^\s*what\s",
    r"^\s*how\s+(do|to|can)\s",
    r"^\s*can\s+you\s",
    r"^\s*could\s+you\s",
    r"^\s*w(i|e)ll\s+you\s",
    r"^\s*are\s+you\s",
    r"^\s*is\s+(?:this|that|it)\s",
    r"^\s*let'?s\s",
    r"^\s*i\s+(?:think|guess|feel|believe)\s",
    r"^\s*maybe\s",
    r"^\s*testing\s",
    r"^\s*trying\s",
    r"^\s*(?:ok|okay|sure|yes|no|done|thanks|thank you)\s*",
    r"^\s*\{.*\}\s*$",  # JSON blobs
    r"^\s*\[.*\]\s*$",  # JSON arrays
    r"Traceback|Error|Exception|stderr",
    r"^\s*(?:git|cargo|npm|pip)\s+\w+",  # command invocations
    # Session status updates -- NOT durable knowledge
    r"^\s*(?:everything|all)\s+(?:from|the)\s.*(?:done|complete|finished|pushed)",
    r"^\s*(?:here'?s|this is)\s+(?:what|the|a)\s+(?:summary|status|update|result|report)",
    r"^\s*(?:the\s+)?\d+\s+(?:crate\s+)?(?:integrations?|phases?|tasks?|changes?)\s+(?:are|is)\s+(?:done|complete|finished)",
    r"^\s*(?:completed?|finished?|done|pushed|committed)",
    r"^\s*new\s+(?:reference|file|tool|endpoint|feature)",
    r"^\s*added\s+(?:reference|file|tool|endpoint|feature)",
    r"^\s*patched\s+",
    r"^\s*\d+\s+(?:new\s+)?(?:MCP\s+)?tools?",
    r"^\s*(?:tool count|test count|compile)",
    r"^\s*(?:needs?|requires?|blocked|waiting)",
    r"\[IMPORTANT:.*Background process",
    r"^\s*(?:the\s+)?(?:key\s+)?session\s+learnings?",
    r"^\s*(?:compiles?|pushed|installed)",
]

# Patterns that indicate high-value technical content
TECH_SIGNALS = [
    r"\b(?:struct|enum|fn|impl|trait|pub)\b",  # Rust
    r"\b(?:def|class|import|async)\b",  # Python
    r"\b(?:export|const|interface|type)\b",  # TypeScript
    r"\b(?:yaml|toml|json|xml|sql)\b",
    r"\b(?:config|setting|option|flag)\b",
    r"\b(?:hook|plugin|skill|tool)\b",
    r"\b(?:model|provider|api|endpoint)\b",
    r"\b(?:build|compile|test|deploy|install)\b",
    r"\b(?:namespace|crate|module|package)\b",
    r"\b(?:memory|semantic|embedding|vector)\b",
    r"\b(?:error|bug|fix|patch|regression)\b",
    r"\b(?:permission|approval|security|allowlist)\b",
    r"\b(?:profile|session|gateway|platform)\b",
]


def score_content(text):
    """Return (score, category) for a piece of text."""
    score = 0
    category = "general"
    text_lower = text.lower()

    for pattern, cat in DURABLE_PATTERNS:
        if re.search(pattern, text_lower):
            score += 2
            if category == "general":
                category = cat

    for pattern in TECH_SIGNALS:
        if re.search(pattern, text_lower):
            score += 1

    return score, category


def should_skip(text):
    text_stripped = text.strip()
    if len(text_stripped) < 20:
        return True
    for pattern in SKIP_PATTERNS:
        if re.search(pattern, text_stripped, re.IGNORECASE):
            return True
    return False


def extract_candidate_facts(user_msg, assistant_resp):
    """Extract candidate durable facts from the conversation turn."""
    candidates = []

    # Check user message for preferences/corrections (high signal)
    if user_msg and not should_skip(user_msg):
        score, cat = score_content(user_msg)
        if score >= MIN_INTEREST_SIGNAL:
            # User messages are strong signal for preferences
            candidates.append({
                "content": user_msg.strip()[:MAX_FACT_LEN],
                "score": score + 2,  # boost user-stated facts
                "category": cat,
                "source": "user_message",
            })

    # Extract declarative sentences from assistant response
    if assistant_resp:
        # Split into sentences/paragraphs
        sentences = re.split(r'\n\n+|\.\s+(?=[A-Z])', assistant_resp)
        for sent in sentences:
            sent = sent.strip()
            if should_skip(sent):
                continue
            score, cat = score_content(sent)
            if score >= MIN_INTEREST_SIGNAL:
                candidates.append({
                    "content": sent[:MAX_FACT_LEN],
                    "score": score,
                    "category": cat,
                    "source": "assistant_response",
                })

    # Sort by score, dedup by content similarity
    candidates.sort(key=lambda c: c["score"], reverse=True)
    seen = set()
    unique = []
    for c in candidates:
        # Simple dedup: skip if first 60 chars match something already seen
        key = c["content"][:60].lower()
        if key not in seen:
            seen.add(key)
            unique.append(c)

    return unique[:MAX_CAPTURES_PER_TURN]


def dedupe_against_db(candidates):
    """Check if candidates already exist in the DB by searching for them."""
    if not candidates:
        return []

    fresh = []
    for c in candidates:
        # Search for the candidate content using HTTP (warm) or fallback
        result = http_search(c["content"][:200], top_k=3, timeout=8)
        if not result or not result.get("ok"):
            # If search fails, be conservative and skip
            continue

        hits = result.get("results") or []
        is_dup = False
        for h in hits:
            existing = (h.get("content") or "").strip()
            if existing and (
                existing[:100] == c["content"][:100]
                or c["content"][:100] in existing
                or existing[:100] in c["content"]
            ):
                is_dup = True
                break

        if not is_dup:
            fresh.append(c)

    return fresh


def namespace_for(cwd, category):
    """Determine namespace based on cwd and category."""
    path = Path(cwd or ".")
    # Check if we're in a known project
    try:
        name = path.name
        if name and path.parent != path:  # not root
            return f"projects"
    except Exception:
        pass
    return "general"


# ── Main ────────────────────────────────────────────────────────────────

def main():
    payload = {}
    try:
        payload = json.load(sys.stdin)
    except Exception:
        pass

    extra = payload.get("extra") or {}
    if not isinstance(extra, dict):
        extra = {}

    user_msg = str(extra.get("user_message") or "")
    assistant_resp = str(extra.get("assistant_response") or "")

    # Need both to be meaningful
    if not user_msg or not assistant_resp:
        return 0

    # Skip very short turns
    if len(user_msg) < 15:
        return 0

    # Skip slash commands
    if user_msg.lstrip().startswith("/"):
        return 0

    cwd = payload.get("cwd") or ""

    # Extract candidates
    candidates = extract_candidate_facts(user_msg, assistant_resp)
    if not candidates:
        return 0

    # Dedupe against existing DB (uses warm HTTP server automatically)
    fresh = dedupe_against_db(candidates)
    if not fresh:
        return 0

    # Capture
    ns = namespace_for(cwd, "general")
    captured = []
    for c in fresh:
        # Build a descriptive fact content
        content = c["content"]
        # Prefix with source context if from user message (preference/correction)
        if c["source"] == "user_message" and c["category"] in ("preference", "correction"):
            content = f"User stated: {content}"

        result = http_add_fact(content, namespace=ns, timeout=10)
        if result and result.get("ok"):
            fact_id = result.get("fact_id") or result.get("id") or ""
            if fact_id:
                captured.append(f"{c['category']}: {content[:80]}")
                # Record outcome for RL feedback
                try:
                    import urllib.request
                    from sm_http_client import get_http_port
                    port = get_http_port()
                    if port:
                        body_out = json.dumps({"query": content[:200], "outcome": "good", "query_class": "A"}).encode()
                        req = urllib.request.Request(f"http://127.0.0.1:{port}/record-outcome",
                                                     data=body_out, headers={"Content-Type": "application/json"})
                        urllib.request.urlopen(req, timeout=3).read()
                except Exception:
                    pass

    if captured:
        # Log to stderr (appears in logs, doesn't interfere with output)
        print(
            f"Auto-captured {len(captured)} fact(s) to semantic memory "
            f"(namespace={ns}): " + " | ".join(captured),
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())