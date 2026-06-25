#!/usr/bin/env python3
"""
post_tool_call hook for semantic-memory tool receipt recording.

Records a lightweight receipt for every tool call the agent makes.
This creates an auditable trail of agent actions without grepping logs.
Receipts are stored as facts in semantic memory with memory_kind=observation.
"""

import sys
import json
import re

# Import the HTTP client
sys.path.insert(0, "/home/sikmindz/.hermes/agent-hooks")
from sm_http_client import http_available, http_add_fact, get_http_port


def main():
    payload = {}
    try:
        payload = json.load(sys.stdin)
    except Exception:
        pass

    tool_name = payload.get("tool_name", "unknown")
    tool_input = payload.get("tool_input", {})
    session_id = payload.get("session_id", "")
    cwd = payload.get("cwd", "")

    # Skip semantic-memory's own tools (avoid recursive receipts)
    if tool_name.startswith("sm_"):
        return 0

    # Skip low-value tools
    SKIP_TOOLS = {"browser_snapshot", "browser_console", "browser_vision",
                  "browser_get_images", "browser_scroll", "todo"}
    if tool_name in SKIP_TOOLS:
        return 0

    # Build a compact receipt
    # Extract key info from tool_input based on tool type
    summary = ""
    if tool_name == "terminal":
        cmd = tool_input.get("command", "")
        summary = f"terminal: {cmd[:100]}"
    elif tool_name == "read_file":
        path = tool_input.get("path", "")
        summary = f"read_file: {path}"
    elif tool_name == "write_file":
        path = tool_input.get("path", "")
        content_len = len(tool_input.get("content", ""))
        summary = f"write_file: {path} ({content_len} bytes)"
    elif tool_name == "patch":
        path = tool_input.get("path", "")
        summary = f"patch: {path}"
    elif tool_name == "search_files":
        pattern = tool_input.get("pattern", "")
        summary = f"search_files: {pattern}"
    elif tool_name == "browser_navigate":
        url = tool_input.get("url", "")
        summary = f"browser_navigate: {url}"
    elif tool_name == "browser_click":
        ref = tool_input.get("ref", "")
        summary = f"browser_click: {ref}"
    elif tool_name == "browser_type":
        ref = tool_input.get("ref", "")
        text = tool_input.get("text", "")[:50]
        summary = f"browser_type: {ref} = {text}"
    elif tool_name == "delegate_task":
        goal = tool_input.get("goal", "")[:80]
        summary = f"delegate_task: {goal}"
    elif tool_name == "execute_code":
        code_len = len(tool_input.get("code", ""))
        summary = f"execute_code: {code_len} chars"
    elif tool_name == "memory":
        action = tool_input.get("action", "")
        target = tool_input.get("target", "")
        summary = f"memory_{action}: {target}"
    else:
        # Generic receipt
        input_summary = json.dumps(tool_input)[:100]
        summary = f"{tool_name}: {input_summary}"

    # Store as observation in semantic memory
    if http_available():
        receipt_content = f"Tool receipt: {summary}"
        if cwd:
            receipt_content += f" (cwd: {cwd})"
        if session_id:
            receipt_content += f" (session: {session_id[:8]})"
        
        http_add_fact(
            receipt_content,
            namespace="tool-receipts",
            timeout=5
        )

    return 0


if __name__ == "__main__":
    main()