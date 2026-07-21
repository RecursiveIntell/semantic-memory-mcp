#!/usr/bin/env python3
import importlib.util
import pathlib
import sys

HOOK = pathlib.Path(__file__).resolve().parent.parent / "sm-recall.py"
spec = importlib.util.spec_from_file_location("sm_recall", HOOK)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def assert_eq(left, right, label):
    if left != right:
        raise AssertionError(f"{label}: expected {right!r}, got {left!r}")


def assert_in(needle, haystack, label):
    if needle not in haystack:
        raise AssertionError(f"{label}: expected {needle!r} in {haystack!r}")


def assert_not_in(needle, haystack, label):
    if needle in haystack:
        raise AssertionError(f"{label}: expected {needle!r} not in {haystack!r}")


def test_quality_labels_status_as_not_evidence():
    q = mod.classify_recall_quality(
        {
            "namespace": "projects",
            "content": "WHAT'S MISSING: async delegation complete but TUI missing routes. FINAL REPORT template follows.",
            "source": "Fact { namespace: \"projects\" }",
        },
        "how did all this change things?",
    )
    assert_eq(q["label"], "stale_status", "label")
    assert_eq(q["safe_as_evidence"], False, "safe_as_evidence")
    assert_in("status-or-completion-language", q["reasons"], "reasons")
    assert_in("needs-live-verification", q["reasons"], "reasons")


def test_quality_labels_verified_research_as_authoritative_durable():
    q = mod.classify_recall_quality(
        {
            "namespace": "research",
            "content": "arXiv:2606.27027v1 ShareLock describes multi-tool poisoning against MCP. Source: arXiv API sweep 2026-06-27.",
            "source": "Fact { namespace: \"research\" }",
        },
        "ShareLock MCP poisoning",
    )
    assert_eq(q["label"], "authoritative_durable", "label")
    assert_eq(q["safe_as_evidence"], True, "safe_as_evidence")
    assert_in("verification-signal", q["reasons"], "reasons")


def test_format_line_includes_quality_metadata():
    result = {
        "namespace": "research",
        "content": "arXiv:2606.27027v1 Source: arXiv API sweep.",
        "score": 0.25,
    }
    q = mod.classify_recall_quality(result, "MCP poisoning")
    line = mod.format_recall_line(result, q, score_val=0.25)
    assert_in("quality=authoritative_durable", line, "quality metadata")
    assert_in("safe_evidence=yes", line, "safe evidence metadata")


def test_filter_omits_unsafe_by_default_and_reports_count():
    results = [
        {"namespace": "projects", "content": "WHAT'S MISSING: final report template", "score": 0.9},
        {"namespace": "research", "content": "Source: arXiv API sweep. Verified finding.", "score": 0.8},
    ]
    kept, filtered = mod.apply_quality_gate(results, "what changed?", maxhits=5)
    assert_eq(len(kept), 1, "kept count")
    assert_eq(len(filtered), 1, "filtered count")
    assert_eq(filtered[0][1]["label"], "stale_status", "filtered label")


def test_background_is_not_safe_evidence_without_verification_signal():
    result = {
        "namespace": "projects",
        "content": "Current project facts and prior work summary without receipts.",
        "source": "Fact { namespace: \"projects\" }",
    }
    q = mod.classify_recall_quality(result, "what do you remember?")
    assert_eq(q["label"], "background", "label")
    assert_eq(q["safe_as_evidence"], False, "safe_as_evidence")

    kept, filtered = mod.apply_quality_gate([result], "what do you remember?", maxhits=5)
    assert_eq(len(kept), 0, "unsafe background kept count")
    assert_eq(len(filtered), 1, "unsafe background filtered count")


def test_speculative_cargo_graph_proposal_is_not_safe_evidence():
    q = mod.classify_recall_quality(
        {
            "namespace": "projects",
            "content": "We could build a tool that ingests all Cargo.toml files and builds a knowledge graph of crate relationships.",
            "source": "Fact { namespace: \"projects\" }",
        },
        "is the cargo graph implemented right now?",
    )
    assert_eq(q["label"], "speculative", "label")
    assert_eq(q["safe_as_evidence"], False, "safe_as_evidence")
    assert_in("speculative-language", q["reasons"], "reasons")


def test_plugin_separation_instruction_is_background_not_project_state_evidence():
    q = mod.classify_recall_quality(
        {
            "namespace": "projects",
            "content": "You need to seperate out the claude plugin from the hermes and codex integrations.",
            "source": "Fact { namespace: \"projects\" }",
        },
        "what is the current plugin state?",
    )
    assert_eq(q["label"], "background", "label")
    assert_eq(q["safe_as_evidence"], False, "safe_as_evidence")
    assert_in("needs-live-verification", q["reasons"], "reasons")


def test_compaction_request_is_background_not_completion_evidence():
    q = mod.classify_recall_quality(
        {
            "namespace": "projects",
            "content": "implement everything and when you're done, package it up as a reusable compaction crate",
            "source": "Fact { namespace: \"projects\" }",
        },
        "is the compaction crate done right now?",
    )
    assert_eq(q["label"], "background", "label")
    assert_eq(q["safe_as_evidence"], False, "safe_as_evidence")
    assert_in("needs-live-verification", q["reasons"], "reasons")


def test_completion_language_requires_live_verification():
    q = mod.classify_recall_quality(
        {
            "namespace": "projects",
            "content": "Everything is done and shipped. Active config remains context_governor.",
            "source": "Fact { namespace: \"projects\" }",
        },
        "is it done right now?",
    )
    assert_eq(q["label"], "stale_status", "label")
    assert_eq(q["safe_as_evidence"], False, "safe_as_evidence")


def test_recalled_integrity_report_is_background_not_live_evidence():
    q = mod.classify_recall_quality(
        {
            "namespace": "projects",
            "content": "/verify-integrity returned facts_missing_embeddings=145 in a previous session.",
            "source": "Fact { namespace: \"projects\" }",
        },
        "what is integrity right now?",
    )
    assert_eq(q["label"], "background", "label")
    assert_eq(q["safe_as_evidence"], False, "safe_as_evidence")
    assert_in("needs-live-verification", q["reasons"], "reasons")


def test_graph_discovery_is_related_not_proof():
    q = mod.classify_recall_quality(
        {
            "namespace": "semantic-memory",
            "content": "Source: verified prior graph edge note.",
            "source": "Fact { namespace: \"semantic-memory\" }",
            "_graph_discovery": True,
        },
        "how are graph edges connected?",
    )
    assert_eq(q["label"], "graph_discovery", "label")
    assert_eq(q["safe_as_evidence"], False, "safe_as_evidence")
    assert_in("graph-discovery-not-proof", q["reasons"], "reasons")


def test_live_verification_guidance_for_current_state_queries():
    assert_eq(mod.is_current_state_query("is context governor active right now?"), True, "current query")
    guidance = mod.live_verification_guidance()
    assert_in("semantic memory is discovery, not proof", guidance, "guidance")
    assert_in("Verify live config", guidance, "guidance")


def test_tool_receipts_filtered_from_default_recall():
    results = [
        {"namespace": "tool-receipts", "content": "Tool receipt: terminal ran cargo test and exited 0", "score": 0.99},
        {"namespace": "research", "content": "Source: arXiv API sweep. Verified finding.", "score": 0.8},
    ]
    kept, filtered = mod.apply_quality_gate(results, "cargo test receipt", maxhits=5)
    assert_eq(len(kept), 1, "kept count")
    assert_eq(kept[0][0]["namespace"], "research", "kept namespace")
    assert_eq(len(filtered), 1, "filtered count")
    assert_eq(filtered[0][0]["namespace"], "tool-receipts", "filtered namespace")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
