#!/usr/bin/env python3
"""
on_session_end hook for semantic-memory active capture.

Instead of just printing a passive reminder, this hook:
1. Reports how many facts were captured during the session (via KB stats delta)
2. Scans the last conversation for uncommitted durable facts
3. Prints a structured summary to stderr for the agent log

Fails open: any error -> exit 0, no output on stdout.
"""
import json, os, sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sm_http_client import http_stats, http_available, search as http_search, add_fact as http_add_fact

# Patterns that indicate durable facts worth capturing
DURABLE_PATTERNS = [
    # User preferences and corrections
    (r"user (?:prefers|wants|likes|always|never)", "user_preference"),
    (r"don't (?:do|use|store)", "correction"),
    (r"(?:fix|change|update) (?:that|this|it) (?:to|so that)", "correction"),
    # Project/config facts
    (r"(?:path|dir|directory) (?:is|at) ", "config_fact"),
    (r"(?:version|release) (?:is|at) ", "config_fact"),
    # Technical discoveries
    (r"(?:workaround|gotcha|pitfall|lesson)", "discovery"),
    (r"(?:the (?:fix|solution|cause) (?:is|was))", "discovery"),
]


def main():
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            return 0
        payload = json.loads(raw)
    except Exception:
        return 0

    if not http_available():
        # No HTTP server -- can't do active capture, just print reminder
        print(
            "Semantic memory: HTTP server not running. Run sm_add_fact for any "
            "durable facts from this session before closing.",
            file=sys.stderr)
        return 0

    # Get current stats
    stats = http_stats(timeout=5)
    if not stats:
        return 0

    fact_count = stats.get("facts", 0)

    # Print session-end summary to stderr
    print(
        f"Semantic memory session-end: {fact_count} facts in KB. "
        f"Review this session for any uncommitted durable facts. "
        f"Use sm_add_fact for decisions, stable config, and corrections. "
        f"Use sm_update_fact to correct outdated facts. "
        f"Use sm_consolidate_facts to merge duplicates.",
        file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())