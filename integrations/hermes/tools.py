"""Deterministic wrappers around the public Hermes MCP CLI.

This plugin intentionally does not mirror any `sm_*` API. Hermes discovers
those APIs directly from the MCP server after installation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from typing import Any, Sequence

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_PROFILES = {"lean", "standard", "agent", "full"}
_TIMEOUT_SECONDS = 180


def _response(*, success: bool, **values: Any) -> str:
    return json.dumps({"success": success, **values}, sort_keys=True)


def _server_name(value: Any) -> str:
    name = str(value or "semantic_memory")
    if not _NAME.fullmatch(name):
        raise ValueError("server_name must match [A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
    return name


def _executable(value: Any, default: str) -> str:
    candidate = os.path.expanduser(str(value or default))
    if os.path.isabs(candidate):
        path = Path(candidate)
        if not path.is_file() or not os.access(path, os.X_OK):
            raise ValueError(f"executable is not runnable: {candidate}")
        return str(path)
    resolved = shutil.which(candidate)
    if resolved is None:
        raise ValueError(f"executable is not on PATH: {candidate}")
    return resolved


def _run(argv: Sequence[str], *, interactive: bool = False) -> str:
    if not argv:
        return _response(success=False, error="empty command")
    try:
        if interactive:
            completed = subprocess.run(list(argv), check=False, timeout=_TIMEOUT_SECONDS)
            stdout = ""
            stderr = ""
        else:
            completed = subprocess.run(
                list(argv),
                check=False,
                capture_output=True,
                text=True,
                timeout=_TIMEOUT_SECONDS,
            )
            stdout = completed.stdout
            stderr = completed.stderr
        return _response(
            success=completed.returncode == 0,
            command=shlex.join(argv),
            exit_code=completed.returncode,
            stdout=stdout,
            stderr=stderr,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return _response(success=False, command=shlex.join(argv), error=str(error))


def install(params: dict[str, Any], **_: Any) -> str:
    """Add the stdio MCP entry with `--args` in its required final position."""
    try:
        hermes = _executable(None, "hermes")
        binary = _executable(params.get("binary"), "semantic-memory-mcp")
        name = _server_name(params.get("server_name"))
        profile = str(params.get("tool_profile") or "agent")
        if profile not in _PROFILES:
            raise ValueError(f"tool_profile must be one of {sorted(_PROFILES)}")
        memory_dir = Path(
            os.path.expanduser(
                str(params.get("memory_dir") or "~/.local/share/semantic-memory")
            )
        ).resolve()
        memory_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        argv = [
            hermes,
            "mcp",
            "add",
            name,
            "--command",
            binary,
            "--args",
            "--memory-dir",
            str(memory_dir),
            "--tool-profile",
            profile,
        ]
        return _run(argv)
    except (OSError, ValueError) as error:
        return _response(success=False, error=str(error))


def list_servers(params: dict[str, Any], **_: Any) -> str:
    del params
    try:
        return _run([_executable(None, "hermes"), "mcp", "list"])
    except ValueError as error:
        return _response(success=False, error=str(error))


def test(params: dict[str, Any], **_: Any) -> str:
    try:
        name = _server_name(params.get("server_name"))
        return _run([_executable(None, "hermes"), "mcp", "test", name])
    except ValueError as error:
        return _response(success=False, error=str(error))


def configure(params: dict[str, Any], **_: Any) -> str:
    try:
        name = _server_name(params.get("server_name"))
        argv = [_executable(None, "hermes"), "mcp", "configure", name]
        interactive = bool(params.get("interactive", False))
        if not interactive:
            return _response(
                success=True,
                launched=False,
                interactive=True,
                command=shlex.join(argv),
                message="Run this command in a terminal to choose the exposed MCP tools.",
            )
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            return _response(
                success=False,
                launched=False,
                command=shlex.join(argv),
                error="Hermes tool selection requires an interactive terminal.",
            )
        return _run(argv, interactive=True)
    except ValueError as error:
        return _response(success=False, error=str(error))
