"""Shared pure recall presentation policy; no native truth/permission authority.

Eligibility for untrusted context is NOT evidence verification. A namespace,
source string, keyword or discovery route cannot mint proof or execute a task.
"""
import json
import re

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

LOW_TRUST_NAMESPACES = {"autonomous", "tool-receipts", "general", "test",
    "quarantine-artifacts",
}


_HISTORY = re.compile(r"\b(?:prior|previous|old|historical|earlier)\b.{0,40}\b(?:plans?|instructions?|tasks?)\b", re.I | re.S)
_TASK = re.compile(r"^\s*(?:you are (?:council|agent)|complete (?:these|the following) tasks|your (?:task|role) is|implement (?:everything|the plan)|you need to\b)", re.I)
_HISTORY_RECOVERY = re.compile(
    r"^\s*(?:please(?:\s+|,\s*))?(?:(?:can|could|would|will)\s+you\s+)?"
    r"(?:recover|recall|retrieve|show|quote|display|find|look\s+up|pull\s+up)\b", re.I)
_HISTORY_QUESTION = re.compile(r"^\s*what\s+(?:did|was|were)\b", re.I)
_HISTORY_OPTOUT = re.compile(
    r"\b(?:(?:do\s+not|don['’]t|never)\s+"
    r"(?:use|follow|recover|recall|retrieve|show|quote|display|include|load|inject|consult|find|search|look\s+up|pull\s+up)\b"
    r"|(?:avoid|ignore|exclude|omit|skip|leave\s+out)\b"
    r"|without\s+(?:using|following|recovering|recalling|retrieving|showing|quoting|including|loading|injecting|consulting)\b)", re.I)
_HISTORY_PASSIVE_OPTOUT = re.compile(
    r"\b(?:(?:must|should|shall|will|can|may|could|would|is|are|was|were)\s+not"
    r"|(?:must|should|could|would|is|are|was|were)n['’]t|can['’]t|won['’]t|cannot)"
    r"\s+(?:to\s+)?(?:be\s+)?"
    r"(?:used|followed|recovered|recalled|retrieved|shown|quoted|displayed|included|loaded|injected|consulted)\b", re.I)
_HISTORY_RETRACTION = re.compile(
    r"^\s*(?:no\b[,\s:]*)?(?:skip|omit|exclude|ignore|leave\s+out)\s+"
    r"(?:them|that|those|it)\b", re.I)


def _history_recovery_requested(query):
    """Conservative English intent gate, not general language/security parsing.

    A historical mention alone is insufficient. An explicit opt-out in a
    history-bearing clause takes precedence over another recovery clause.
    Non-execution wording does not cancel an explicit request to quote data.
    """
    requested = False
    for clause in re.split(r"[.!?;\n]", query):
        # A clear later retraction can refer back to the requested historical
        # material. Unrelated commands such as "skip unit tests" do not match.
        if requested and _HISTORY_RETRACTION.search(clause):
            return False
        matches = list(_HISTORY.finditer(clause))
        if not matches:
            continue
        for match in matches:
            if (_HISTORY_OPTOUT.search(clause[:match.end()])
                    or _HISTORY_PASSIVE_OPTOUT.search(clause[match.end():])):
                return False
        if _HISTORY_RECOVERY.search(clause) or _HISTORY_QUESTION.search(clause):
            requested = True
    return requested


def quote_history(content, max_chars):
    """Largest JSON-quoted prefix within the declared content limit."""
    if type(max_chars) is not int or max_chars < 2:
        raise ValueError("quoted history requires at least two characters")
    lo, hi = 0, len(content)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(json.dumps(content[:mid], ensure_ascii=False)) <= max_chars:
            lo = mid
        else:
            hi = mid - 1
    return json.dumps(content[:lo], ensure_ascii=False)


def classify_recall_quality(result, query):
    content = result.get("content")
    if not isinstance(content, str) or not content.strip():
        return {"label": "invalid_candidate", "safe_as_evidence": False,
                "allow_context": False, "quoted_history": False, "reasons": ["missing-or-nontext-content"]}
    namespace = result.get("namespace")
    namespace = namespace.lower() if isinstance(namespace, str) else ""
    source = result.get("source")
    source = source.lower() if isinstance(source, str) else ""
    content_l = content.lower()
    query_l = query.lower() if isinstance(query, str) else ""
    reasons = []
    for markers, reason in ((JUNK_MARKERS, "artifact-marker"),
                            (TEMPLATE_MARKERS, "template-or-prompt-artifact"),
                            (STATUS_MARKERS, "status-or-completion-language"),
                            (SPECULATIVE_MARKERS, "speculative-language"),
                            (VERIFICATION_MARKERS, "verification-signal")):
        if any(marker.lower() in content_l or marker.lower() in source for marker in markers):
            reasons.append(reason)
    if namespace in LOW_TRUST_NAMESPACES:
        reasons.append("low-trust-namespace:" + namespace)
    if any(marker in query_l for marker in LIVE_VERIFICATION_QUERY_MARKERS):
        reasons.append("needs-live-verification")
    task = bool(_TASK.search(content))
    if task:
        reasons.append("historical-task-not-current-instruction")
    if result.get("_graph_discovery"):
        reasons.append("graph-discovery-not-proof")
    artifact = "artifact-marker" in reasons or "template-or-prompt-artifact" in reasons
    stale = "status-or-completion-language" in reasons and "needs-live-verification" in reasons
    speculative = "speculative-language" in reasons and "verification-signal" not in reasons
    restricted = namespace in LOW_TRUST_NAMESPACES
    recoverable = namespace not in {"tool-receipts", "test", "quarantine-artifacts"}
    history_requested = _history_recovery_requested(query_l)
    if _HISTORY.search(query_l) and not history_requested:
        reasons.append("historical-recovery-not-requested-or-opted-out")
    history = history_requested and recoverable and (artifact or task or stale or restricted)
    eligible = history or not (artifact or task or stale or speculative or restricted)
    if history:
        label = "quoted_historical"
    elif stale:
        label = "stale_status"
    elif artifact or (task and content_l.lstrip().startswith("you are ")) or restricted:
        label = "artifact_template"
    elif speculative:
        label = "speculative"
    elif result.get("_graph_discovery"):
        label = "graph_discovery"
    elif "verification-signal" in reasons and namespace in {
        "research", "semantic-memory", "libraries", "libraries-crates", "doctrine",
        "infrastructure", "preferences", "recursiveintell", "behavioral", "agent-setup", "personal",
    }:
        # Presentation label only: neither this namespace nor the marker is proof.
        label = "untrusted_recall"
    else:
        label = "background"
    return {"label": label, "safe_as_evidence": False, "allow_context": eligible,
            "quoted_history": history, "reasons": reasons or ["no-quality-flags"]}


def apply_quality_gate(results, query, maxhits):
    if type(maxhits) is not int or maxhits < 0:
        raise ValueError("maxhits must be a nonnegative integer")
    kept, filtered, seen = [], [], set()
    for result in results:
        quality = classify_recall_quality(result, query)
        content = (result.get("content") or "").strip() if isinstance(result.get("content"), str) else ""
        if not quality["allow_context"]:
            filtered.append((result, quality))
        elif content in seen:
            filtered.append((result, {**quality, "label": "duplicate_context", "reasons": quality["reasons"] + ["duplicate-context"]}))
        elif len(kept) >= maxhits:
            filtered.append((result, {**quality, "label": "output_cap", "reasons": quality["reasons"] + ["output-cap"]}))
        else:
            seen.add(content)
            kept.append((result, quality))
    return kept, filtered
