#!/usr/bin/env python3
"""
pre_llm_call hook for semantic-memory auto-recall.

Searches semantic memory for the user's prompt and injects relevant
facts as context. Uses adaptive routing:
  - Class A (simple): flat /search (fast, ~140ms)
  - Class B/C/D/E (complex): /search-routed (adjusts top_k, exactness)

Self-RAG gate: skips retrieval for greetings, confirmations, and trivial queries.

After search, auto-records the outcome to feed the RL routing policy:
  - Records "good" if results were found with decent scores
  - Records "bad" if no results or very low scores
  - This happens silently, never blocks the hook

Uses the warm HTTP server when available, falls back to spawning.
Fails open: any error -> exit 0, no output.
"""
import json, os, re, subprocess, sys
from pathlib import Path

# Add the agent-hooks directory to the path for the HTTP client
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sm_http_client import (
    search as http_search,
    http_search_routed,
    http_available,
    get_http_port,
)

STOPWORDS = {
    "about", "after", "again", "agent", "agentic", "and", "best", "code",
    "coding", "does", "doing", "everything", "examine", "for", "from",
    "function", "have", "how", "improve", "into", "look", "looks", "make",
    "optimize", "perform", "performance", "possible", "research", "seem",
    "seems", "than", "that", "the", "this", "through", "using", "well",
    "what", "when", "where", "with", "your",
}

# Self-RAG gate: skip retrieval for these
SKIP_PHRASES = {
    "ok", "yes", "no", "thanks", "done", "sure", "yeah", "right",
    "correct", "agreed", "ok thanks", "got it", "sounds good",
    "that works", "makes sense", "i see", "understood", "gotcha",
    "sounds good, do it", "do it", "fix it", "go for it", "continue",
}


def terms(text):
    found = set()
    for m in re.finditer(r"[a-zA-Z][a-zA-Z0-9_-]+", text.lower()):
        w = m.group()
        if w not in STOPWORDS and len(w) > 1:
            found.add(w)
    return found


def classify_query(query):
    """Classify query complexity (A=simple, B=multi-hop, C=contradiction, D=synthesis, E=temporal)."""
    q = query.lower()
    q_terms = terms(query)

    # Contradiction signals
    contradiction_words = {"contradiction", "conflict", "disagree", "but", "vs", "versus", "is it true", "wrong"}
    if any(w in q for w in contradiction_words):
        return "C"

    # Synthesis signals
    synthesis_words = {"summarize", "overview", "all about", "themes", "landscape", "everything", "compare", "comparison"}
    if any(w in q for w in synthesis_words):
        return "D"

    # Temporal signals
    temporal_words = {"when", "before", "after", "changed", "current", "latest", "updated", "how old", "timeline"}
    if any(w in q for w in temporal_words):
        return "E"

    # Multi-hop signals
    relation_words = {"connects", "between", "depends on", "relates to", "relationship",
                       "how did", "how does", "work with", "lead to", "link", "integrate"}
    if len(q_terms) >= 2 and any(w in q for w in relation_words):
        return "B"

    # Default: simple
    return "A"


def auto_record_outcome(query, results, query_class):
    """Record search outcome for RL routing feedback via HTTP /record-outcome."""
    try:
        import urllib.request
        port = get_http_port()
        if not port:
            return

        has_results = len(results) > 0
        top_score = float(results[0].get("score", 0)) if has_results else 0.0
        outcome = "good" if (has_results and top_score > 0.3) else "bad"

        body = json.dumps({
            "query": query[:200],
            "outcome": outcome,
            "query_class": query_class,
        }).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/record-outcome",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=3).read()
    except Exception:
        pass  # fail open, never block on outcome recording


def main():
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            return 0
        payload = json.loads(raw)
    except Exception:
        return 0

    extra = payload.get("extra") or {}
    user_message = extra.get("user_message") or ""
    prompt = (user_message or "").strip()

    # Self-RAG gate: skip retrieval for greetings and confirmations
    if len(prompt) < 12 or prompt.lstrip().startswith("/"):
        return 0

    prompt_lower = prompt.strip().lower()
    if prompt_lower in SKIP_PHRASES:
        return 0
    if any(prompt_lower.startswith(p) and len(prompt_lower) <= len(p) + 10
           for p in ["can you", "could you", "would you", "will you"]):
        return 0

    # Classify query complexity
    query_class = classify_query(prompt)

    # Search via HTTP (warm) or fallback
    # Priority namespaces: technical knowledge first, social media excluded
    PRIORITY_NAMESPACES = ["projects", "research", "semantic-memory", "libraries",
                           "libraries-crates", "doctrine", "agent-setup", "infrastructure",
                           "personal", "behavioral", "codex", "recursiveintell", "general"]

    if http_available():
        if query_class == "A":
            # Simple query: flat search, fast, priority namespaces only
            data = http_search(prompt, top_k=5, namespaces=PRIORITY_NAMESPACES, timeout=10)
        elif query_class in ("C", "D"):
            # Complex query: routed search + LLM rerank for higher precision
            top_k = 10 if query_class == "D" else 5
            data = http_search_routed(prompt, top_k=top_k, query_class=query_class,
                                      namespaces=PRIORITY_NAMESPACES, timeout=12)
            # Chain through /rerank for contradiction/synthesis queries
            if data and isinstance(data, list) and len(data) > 2:
                try:
                    import urllib.request
                    port = get_http_port()
                    rerank_body = json.dumps({
                        "query": prompt[:200],
                        "results": [{"id": r.get("id", ""), "content": r.get("content", "")[:500]}
                                    for r in data[:10]],
                        "model": "granite4.1:3b",
                    }).encode()
                    req = urllib.request.Request(
                        f"http://127.0.0.1:{port}/rerank",
                        data=rerank_body,
                        headers={"Content-Type": "application/json"},
                    )
                    reranked = json.loads(urllib.request.urlopen(req, timeout=15).read())
                    if reranked and isinstance(reranked, list):
                        # Merge rerank scores back into original results
                        rerank_map = {r.get("id", ""): r.get("rerank_score", 0) for r in reranked}
                        for r in data:
                            rid = r.get("id", "")
                            if rid in rerank_map:
                                r["score"] = float(rerank_map[rid]) / 5.0  # normalize 1-5 to 0.2-1.0
                        data.sort(key=lambda x: float(x.get("score", 0)), reverse=True)
                except Exception:
                    pass  # fail open, use unrouted results
        else:
            # Class B/E: routed search without rerank
            top_k = 10 if query_class == "D" else 5
            data = http_search_routed(prompt, top_k=top_k, query_class=query_class,
                                      namespaces=PRIORITY_NAMESPACES, timeout=12)
    else:
        # Fallback: spawn a new MCP process (slow)
        data = None

    if not data:
        return 0

    results = data if isinstance(data, list) else data.get("results", [])
    if not results:
        return 0

    # Auto-record outcome for RL feedback (silent, never blocks)
    auto_record_outcome(prompt, results, query_class)

    # Filter: only show results with decent RRF scores
    # RRF scores are naturally low (0.01-0.03), so use relative threshold
    results_with_scores = []
    for r in results:
        score = float(r.get("score", 0))
        results_with_scores.append((score, r))
    
    if not results_with_scores:
        return 0
    
    # Relative threshold: top result * 0.4, with absolute floor of 0.005
    top_score = results_with_scores[0][0]
    threshold = max(top_score * 0.4, 0.005)
    
    seen_content = set()
    lines = []
    for score, r in results_with_scores:
        content = (r.get("content") or "").strip()
        if not content or content in seen_content:
            continue
        if score < threshold:
            continue
        # Truncate long content
        if len(content) > 300:
            content = content[:297] + "..."
        # Extract namespace from the result
        ns = r.get("namespace", "")
        ns_tag = f" [{ns}]" if ns else ""
        lines.append(f"[{score:.4f}]{ns_tag} {content}")
        seen_content.add(content)
        if len(lines) >= 5:
            break

    if not lines:
        return 0

    route_tag = f" (routed: class {query_class})" if query_class != "A" else ""
    header = f"## Semantic memory recall{route_tag}"
    print(json.dumps({"context": header + "\n" + "\n".join(lines)}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())