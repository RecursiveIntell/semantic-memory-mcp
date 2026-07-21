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
    http_record_outcome,
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

# Content markers that indicate session artifacts, templates, or status prose.
JUNK_MARKERS = [
    "PHASE_", "PHASE-", "INJECTION", "INJECT-", "GUARDRAIL", "PREFLIGHT",
    "AFTER_PHASE_", "CODEX_PHASED", "CODEX_CONTROL_PACK", "MANUAL_PHASE",
    "OPERATOR_PASTE", "COPY_PASTE_SEQUENCE", "MASTER_PROMPT",
    "benchmark_release_grade_super_", "stage1_intake", "stage2_dossier",
    "P00_", "P01_", "P02_", "P03_", "P04_", "P05_", "P06_", "P07_",
    "P08_", "P09_", "P10_", "P11_", "P12_", "P13_",
    "P14_", "P15_", "P16_", "P17_", "P18_", "P19_", "P20_", "P21_",
    "BUILD_ORDER_DAG", "RELEASE_BAR", "CONFORMANCE_PLAN",
    "Grok conversation", "ChatGPT Conversations",
    "CONCEPTUAL_BIRTH", "kernel-oracles", "kernel-execution",
    "EXECUTIVE_INTAKE", "PROMOTION_CANDIDATES", "IMPLEMENTATION_TIMELINE",
    "FINAL_AUDITOR", "RECURSIVEINTELL_SYSTEM_MAP",
    "execution-is-evidence", "recall-linux",
    "TURBO_QUANT_0_2_RELEASE_PROMPT",
    "00_SOURCE_BASIS", "00_RISK_REGISTER", "OPEN_QUESTIONS",
    "canonical_ownership_source_dri", "hostile_audit_docset",
    "constitutional_combat_strength",
    "NEXT_CODEX_RUN", "CLAUDE.md", "claude_code_run", "CODEX.md",
    "OPERATOR_DECISION", "fixpack", "finish_pack", "convergence_spec",
    "closing_hardening", "finish-line", "hardening pass",
    "Non-negotiable", "closeout", "super_pass",
    "codex prompt", "operator decision brief",
    "CODEX_PHASED_PROMPT", "CODEX_EXEC_PHASE", "CODEX_WORKFLOW",
    "CODEX_OUTPUT_INTEGRATION", "START_HERE_PHASED",
    "00_OPERATOR", "00_README", "00_START", "00_AFTER",
    "01_EXECUTIVE", "00_CODEX",
]

STATUS_MARKERS = [
    "what's missing", "whats missing", "what is missing", "missing:",
    "things that are missing", "missing are", "final report", "final auditor",
    "async delegation complete", "not implemented", "not wired", "stub", "todo",
    "blocked", "gap", "completion summary", "status update", "phase complete",
    "added pitfall", "new pitfall", "session summary", "everything is done",
    "done and shipped", "shipped", "active config remains", "completed",
]

TEMPLATE_MARKERS = [
    "template", "prompt template", "implementation prompt", "operator paste",
    "copy/paste", "copy-paste", "start_here", "readme template",
]

SPECULATIVE_MARKERS = [
    "likely", "would likely", "potentially", "may indicate", "could imply",
    "seems to", "intended to", "probably", "could build", "could add",
    "candidate", "proposal", "proposed", "should build", "would build",
]

VERIFICATION_MARKERS = [
    "source:", "verified", "passed", "cargo test", "cargo check",
    "arxiv:", "github api", "crates.io", "receipt", "evidence:",
]

LIVE_VERIFICATION_QUERY_MARKERS = [
    "changed", "change", "current", "latest", "now", "done", "shipped",
    "implemented", "how did", "what changed", "right now", "active", "running",
    "status", "is it done", "did it ship", "did we ship",
]

AUTHORITATIVE_DURABLE_NAMESPACES = {
    "research", "semantic-memory", "libraries", "libraries-crates",
    "doctrine", "infrastructure", "preferences", "recursiveintell",
}

LOW_TRUST_NAMESPACES = {"autonomous", "tool-receipts", "general", "test"}


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


def classify_recall_quality(result, query):
    """Return candidate quality metadata. Recall is evidence only after this gate."""
    content = (result.get("content") or "")
    content_l = content.lower()
    source_l = (result.get("source") or "").lower()
    namespace = (result.get("namespace") or "").lower()
    query_l = (query or "").lower()
    reasons = []

    if result.get("_graph_discovery"):
        reasons.append("graph-discovery-not-proof")

    def has_any(markers):
        return any(marker.lower() in content_l or marker.lower() in source_l for marker in markers)

    if has_any(JUNK_MARKERS):
        reasons.append("artifact-marker")
    if has_any(TEMPLATE_MARKERS):
        reasons.append("template-or-prompt-artifact")
    if has_any(STATUS_MARKERS):
        reasons.append("status-or-completion-language")
    if has_any(SPECULATIVE_MARKERS):
        reasons.append("speculative-language")
    if has_any(VERIFICATION_MARKERS):
        reasons.append("verification-signal")
    if namespace in LOW_TRUST_NAMESPACES:
        reasons.append(f"low-trust-namespace:{namespace}")
    if any(marker in query_l for marker in LIVE_VERIFICATION_QUERY_MARKERS):
        reasons.append("needs-live-verification")

    unsafe = any(
        reason.startswith("artifact-marker")
        or reason.startswith("template-or-prompt-artifact")
        or reason.startswith("low-trust-namespace")
        for reason in reasons
    )
    stale_status = "status-or-completion-language" in reasons and "needs-live-verification" in reasons
    speculative = "speculative-language" in reasons and "verification-signal" not in reasons

    if "graph-discovery-not-proof" in reasons:
        label = "graph_discovery"
        safe_as_evidence = False
    elif stale_status:
        label = "stale_status"
        safe_as_evidence = False
    elif unsafe:
        label = "artifact_template"
        safe_as_evidence = False
    elif speculative:
        label = "speculative"
        safe_as_evidence = False
    elif "verification-signal" in reasons and namespace in AUTHORITATIVE_DURABLE_NAMESPACES:
        label = "authoritative_durable"
        safe_as_evidence = True
    elif namespace == "preferences" and any(word in content_l for word in ("working agreement", "preference", "prefers")):
        label = "authoritative_durable"
        safe_as_evidence = True
    else:
        label = "background"
        safe_as_evidence = False

    return {
        "label": label,
        "safe_as_evidence": safe_as_evidence,
        "reasons": reasons or ["no-quality-flags"],
    }


def apply_quality_gate(results, query, maxhits):
    kept = []
    filtered = []
    for result in results:
        quality = classify_recall_quality(result, query)
        namespace = (result.get("namespace") or "").lower()
        # Tool receipts are audit/debug traces, not durable facts.  Keep them
        # out of default recall unless a future explicit receipt/debug mode is
        # added, otherwise they dominate the KB and look like evidence.
        if namespace == "tool-receipts":
            filtered.append((result, quality))
        elif quality["safe_as_evidence"]:
            kept.append((result, quality))
        else:
            filtered.append((result, quality))
        if len(kept) >= maxhits:
            break
    return kept, filtered


def format_recall_line(result, quality, score_val):
    content = (result.get("content") or "").strip()
    if len(content) > 300:
        content = content[:297] + "..."
    ns = result.get("namespace", "")
    ns_tag = f" [{ns}]" if ns else ""
    safe = "yes" if quality.get("safe_as_evidence") else "no"
    reasons = ";".join(quality.get("reasons") or [])
    return (
        f"[{score_val:.4f}]{ns_tag} "
        f"[quality={quality.get('label')} safe_evidence={safe} reasons={reasons}] "
        f"{content}"
    )




def is_current_state_query(query):
    query_l = (query or "").lower()
    return any(marker in query_l for marker in LIVE_VERIFICATION_QUERY_MARKERS)


def live_verification_guidance():
    return (
        "[live-verification] Current-state/status query: semantic memory is discovery, not proof. "
        "Verify live config/process/repo/API state before final claims."
    )


def auto_record_outcome(query, results, query_class):
    """Record search outcome for RL routing feedback via HTTP /record-outcome."""
    try:
        has_results = bool(results and len(results) > 0)
        top_score = float(results[0].get("score", 0) or results[0].get("rrf_score", 0) or 0) if has_results else 0.0
        outcome = "good" if (has_results and top_score > 0.3) else "bad"
        http_record_outcome(query[:200], outcome=outcome, query_class=query_class, timeout=3)
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
                           "personal", "behavioral", "codex", "recursiveintell", "general",
                           "autonomous", "gloss", "feut", "preferences"]

    # Content markers that indicate session artifacts, not durable knowledge.
    # If a search result's content contains any of these, skip it.
    JUNK_MARKERS = [
        "PHASE_", "PHASE-", "INJECTION", "INJECT-", "GUARDRAIL", "PREFLIGHT",
        "AFTER_PHASE_", "CODEX_PHASED", "CODEX_CONTROL_PACK", "MANUAL_PHASE",
        "OPERATOR_PASTE", "COPY_PASTE_SEQUENCE", "MASTER_PROMPT",
        "benchmark_release_grade_super_", "stage1_intake", "stage2_dossier",
        "P00_", "P01_", "P02_", "P03_", "P04_", "P05_", "P06_", "P07_",
        "P08_", "P09_", "P10_", "P11_", "P12_", "P13_",
        "P14_", "P15_", "P16_", "P17_", "P18_", "P19_", "P20_", "P21_",
        "BUILD_ORDER_DAG", "RELEASE_BAR", "CONFORMANCE_PLAN",
        "Grok conversation", "ChatGPT Conversations",
        "CONCEPTUAL_BIRTH", "kernel-oracles", "kernel-execution",
        "EXECUTIVE_INTAKE", "PROMOTION_CANDIDATES", "IMPLEMENTATION_TIMELINE",
        "FINAL_AUDITOR", "RECURSIVEINTELL_SYSTEM_MAP",
        "execution-is-evidence", "recall-linux",
        "TURBO_QUANT_0_2_RELEASE_PROMPT",
        "00_SOURCE_BASIS", "00_RISK_REGISTER", "OPEN_QUESTIONS",
        "canonical_ownership_source_dri", "hostile_audit_docset",
        "constitutional_combat_strength",
        # Codex/Claude bundle artifacts
        "NEXT_CODEX_RUN", "CLAUDE.md", "claude_code_run", "CODEX.md",
        "OPERATOR_DECISION", "fixpack", "finish_pack", "convergence_spec",
        "closing_hardening", "finish-line", "hardening pass",
        "Non-negotiable", "closeout", "super_pass",
        "codex prompt", "operator decision brief",
        # Codex control packs and bundles
        "CODEX_PHASED_PROMPT", "CODEX_EXEC_PHASE", "CODEX_WORKFLOW",
        "CODEX_OUTPUT_INTEGRATION", "START_HERE_PHASED",
        # More session artifacts
        "00_OPERATOR", "00_README", "00_START", "00_AFTER",
        "01_EXECUTIVE", "00_CODEX",
    ]
    use_cosine_gate = False  # warm HTTP returns RRF scores; cold returns cosine

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
                        "model": os.environ.get("SEMANTIC_MEMORY_LLM_MODEL", "deepseek-v4-flash:cloud"),
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
        # Cold fallback: spawn the MCP binary over stdio (slower, but correct)
        from sm_http_client import resolve_binary, memory_dir, rpc_call_fallback
        args = {"query": prompt, "top_k": 5}
        if PRIORITY_NAMESPACES:
            args["namespaces"] = PRIORITY_NAMESPACES
        data = rpc_call_fallback("sm_search", args, timeout=15)
        use_cosine_gate = True  # cold stdio returns cosine_similarity

    if not data:
        return 0

    results = data if isinstance(data, list) else data.get("results", [])
    if not results:
        return 0

    # Auto-record outcome for RL feedback (silent, never blocks)
    auto_record_outcome(prompt, results, query_class)

    # Dual gating: warm HTTP returns fused RRF scores (0.01-0.03 range);
    # cold stdio returns cosine_similarity (0.0-1.0 range with nomic's high ~0.5 baseline)
    if use_cosine_gate:
        # Cosine gate (cold path): relative band + absolute floor
        MINTOP = float(os.environ.get("SM_RECALL_MINTOP", "0.58"))
        BAND = float(os.environ.get("SM_RECALL_BAND", "0.12"))
        ABSFLOOR = float(os.environ.get("SM_RECALL_ABSFLOOR", "0.54"))
        MAXHITS = int(os.environ.get("SM_RECALL_MAXHITS", "5"))

        results.sort(key=lambda r: float(r.get("cosine_similarity") or 0), reverse=True)
        top_cos = float(results[0].get("cosine_similarity") or 0)
        if top_cos < MINTOP:
            return 0
        floor = max(ABSFLOOR, top_cos - BAND)
        keep = [r for r in results if float(r.get("cosine_similarity") or 0) >= floor][:MAXHITS]
    else:
        # RRF score gate (warm path): relative threshold + absolute floor
        SCOREREL = float(os.environ.get("SM_RECALL_SCOREREL", "0.4"))
        ABSFLOOR = float(os.environ.get("SM_RECALL_ABSFLOOR_RRF", "0.005"))
        MAXHITS = int(os.environ.get("SM_RECALL_MAXHITS", "5"))

        results.sort(key=lambda r: float(r.get("score") or 0), reverse=True)
        top_score = float(results[0].get("score") or 0)
        if top_score <= 0:
            return 0
        threshold = max(top_score * SCOREREL, ABSFLOOR)
        keep = [r for r in results if float(r.get("score") or 0) >= threshold][:MAXHITS]

    # Quality gate: label recall candidates and omit stale/status/template/artifact
    # snippets from primary injected context unless they are safe as evidence.
    keep_pairs, filtered_pairs = apply_quality_gate(keep, prompt, MAXHITS)
    keep = [r for r, _quality in keep_pairs]
    quality_by_key = {
        (r.get("id") or r.get("result_id") or r.get("content") or ""): quality
        for r, quality in keep_pairs
    }

    seen_content = set()
    lines = []
    if is_current_state_query(prompt):
        lines.append(live_verification_guidance())
    if filtered_pairs:
        labels = {}
        for _result, quality in filtered_pairs:
            labels[quality["label"]] = labels.get(quality["label"], 0) + 1
        label_summary = ", ".join(f"{label}={count}" for label, count in sorted(labels.items()))
        lines.append(
            f"[recall-quality] filtered {len(filtered_pairs)} unsafe/background-risk candidates "
            f"({label_summary}); use live verification before trusting stale status snippets"
        )
    for r in keep:
        content = (r.get("content") or "").strip()
        if not content or content in seen_content:
            continue
        key = r.get("id") or r.get("result_id") or r.get("content") or ""
        quality = quality_by_key.get(key) or classify_recall_quality(r, prompt)
        if use_cosine_gate:
            score_val = float(r.get("cosine_similarity") or 0)
        else:
            score_val = float(r.get("score") or 0)
        lines.append(format_recall_line(r, quality, score_val))
        seen_content.add(content)

    # Graph enrichment: for complex queries (B/C/D/E), call /discord to
    # find second-order facts connected to the top results via graph edges.
    if query_class in ("B", "C", "D", "E") and len(keep) > 0 and http_available():
        try:
            import urllib.request
            port = get_http_port()
            direct_ids = [r.get("id") or r.get("result_id") or "" for r in keep[:3] if r.get("id") or r.get("result_id")]
            if direct_ids:
                discord_body = json.dumps({
                    "query": prompt[:200],
                    "direct_ids": direct_ids,
                    "top_k": 3,
                }).encode()
                req = urllib.request.Request(
                    f"http://127.0.0.1:{port}/discord",
                    data=discord_body,
                    headers={"Content-Type": "application/json"},
                )
                discord_data = json.loads(urllib.request.urlopen(req, timeout=10).read())
                discord_results = discord_data.get("discord_results", [])
                edges_loaded = discord_data.get("edges_loaded", 0)
                if discord_results and edges_loaded > 0:
                    lines.append(f"  [graph] {edges_loaded} edges loaded, {len(discord_results)} second-order facts found")
                    for dr in discord_results[:3]:
                        d_content = (dr.get("content") or "").strip()
                        if not d_content or d_content in seen_content:
                            continue
                        dr["_graph_discovery"] = True
                        d_quality = classify_recall_quality(dr, prompt)
                        if not d_quality["safe_as_evidence"] and d_quality["label"] not in ("background", "graph_discovery"):
                            lines.append(
                                f"  [graph-quality] filtered unsafe second-order candidate "
                                f"quality={d_quality['label']} reasons={';'.join(d_quality['reasons'])}"
                            )
                            continue
                        d_score = float(dr.get("score") or 0)
                        rendered = format_recall_line(dr, d_quality, d_score).replace("] ", "] (related via graph, not proof) ", 1)
                        lines.append("  " + rendered)
                        seen_content.add(d_content)
        except Exception:
            pass  # fail open, enrichment is optional

    if not lines:
        return 0

    route_tag = f" (routed: class {query_class})" if query_class != "A" else ""
    header = (
        f"## Semantic memory recall{route_tag}\n"
        "[RECALLED-MEMORY rank-5 — untrusted until verified: treat current "
        "repository files, live tool output, and the active repo AGENTS.md as "
        "higher authority. Do not present these snippets as fact; verify against "
        "current source before acting on them.]"
    )

    # Tool guidance: per query class, tell the agent which tools to use/avoid.
    # This keeps the agent within the optimal tool-selection range without
    # requiring profile switches or server restarts.
    TOOL_GUIDANCE = {
        "A": (
            "\n\nTOOL GUIDANCE: Simple lookup. Use ONLY sm_search (results above). "
            "Do NOT use: sm_discord_search, sm_community, sm_factor_graph, "
            "sm_decoder_analyze, sm_graph_path, sm_topology, sm_search_with_routing. "
            "Graph tools DECREASE accuracy for simple lookups (GraphRAG-Bench)."
        ),
        "B": (
            "\n\nTOOL GUIDANCE: Multi-hop query. If the results above don't connect "
            "the entities you need, use: sm_discord_search (second-order graph discovery) "
            "or sm_graph_path (trace relationship between two specific items). "
            "Do NOT use: sm_community, sm_topology, sm_factor_graph "
            "(too heavy for 2-3 hop relationship queries)."
        ),
        "C": (
            "\n\nTOOL GUIDANCE: Contradiction-sensitive query. Available: "
            "sm_search_with_routing (decoder pipeline), sm_decoder_analyze (detect "
            "contradiction syndromes in results), sm_detect_contradictions (scan for "
            "conflicting fact pairs). If contradictory nodes are connected by graph "
            "edges AND there are >3 connected nodes: sm_factor_graph."
        ),
        "D": (
            "\n\nTOOL GUIDANCE: Synthesis query. All graph tools appropriate: "
            "sm_discord_search (second-order discovery), sm_community (group by "
            "knowledge communities when >10 results), sm_factor_graph (belief "
            "propagation across contradictory signals). Use sm_search_with_routing "
            "for broader recall before synthesizing."
        ),
        "E": (
            "\n\nTOOL GUIDANCE: Temporal query. Use sm_search_as_of for date-scoped "
            "retrieval. sm_graph_path with temporal edges can surface chronological "
            "ordering. sm_list_facts can enumerate a namespace over time. "
            "Do NOT use: sm_community, sm_factor_graph (timeline queries don't "
            "benefit from these)."
        ),
    }
    guidance = TOOL_GUIDANCE.get(query_class, "")
    if guidance:
        lines.append(guidance)

    print(json.dumps({"context": header + "\n" + "\n".join(lines)}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())