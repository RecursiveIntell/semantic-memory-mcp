#!/usr/bin/env python3
"""Read-only structural validator for semantic-memory agent integrations."""

from __future__ import annotations

import json
from pathlib import Path
import py_compile
import re
import sys
import tempfile
import tomllib

try:
    import yaml
except ImportError as error:  # pragma: no cover - explicit dependency message
    raise SystemExit("PyYAML is required to validate integrations/hermes/plugin.yaml") from error


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
ERRORS: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        ERRORS.append(message)


def load_json(relative: str) -> dict:
    path = ROOT / relative
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        ERRORS.append(f"{relative}: invalid JSON: {error}")
        return {}
    check(isinstance(value, dict), f"{relative}: top level must be an object")
    return value


def load_yaml(relative: str) -> dict:
    path = ROOT / relative
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        ERRORS.append(f"{relative}: invalid YAML: {error}")
        return {}
    check(isinstance(value, dict), f"{relative}: top level must be a mapping")
    return value or {}


def load_toml(relative: str) -> dict:
    path = ROOT / relative
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        ERRORS.append(f"{relative}: invalid TOML: {error}")
        return {}


def frontmatter(relative: str) -> dict[str, str]:
    path = ROOT / relative
    text = path.read_text(encoding="utf-8")
    match = re.match(r"\A---\n(.*?)\n---\n", text, re.DOTALL)
    if match is None:
        ERRORS.append(f"{relative}: missing YAML frontmatter")
        return {}
    try:
        value = yaml.safe_load(match.group(1))
    except yaml.YAMLError as error:
        ERRORS.append(f"{relative}: invalid frontmatter: {error}")
        return {}
    check(isinstance(value, dict), f"{relative}: frontmatter must be a mapping")
    return value or {}


hermes = load_yaml("hermes/plugin.yaml")
check(hermes.get("name") == "semantic-memory-mcp", "Hermes manifest name mismatch")
check(bool(hermes.get("version")), "Hermes manifest needs a version")
with tempfile.TemporaryDirectory(prefix="semantic-memory-integrations-") as temp_dir:
    for index, module in enumerate(
        ["hermes/__init__.py", "hermes/schemas.py", "hermes/tools.py"]
    ):
        try:
            py_compile.compile(
                str(ROOT / module),
                cfile=str(Path(temp_dir) / f"module-{index}.pyc"),
                doraise=True,
            )
        except py_compile.PyCompileError as error:
            ERRORS.append(f"{module}: Python compile failed: {error.msg}")

claude_manifest = load_json("claude-plugin/.claude-plugin/plugin.json")
check(claude_manifest.get("name") == "semantic-memory", "Claude plugin name mismatch")
claude_mcp = load_json("claude-plugin/.mcp.json")
command = (
    claude_mcp.get("mcpServers", {})
    .get("semantic-memory", {})
    .get("command", "")
)
check(command.startswith("${CLAUDE_PLUGIN_ROOT}/"), "Claude MCP command must use CLAUDE_PLUGIN_ROOT")
launcher = ROOT / "claude-plugin/bin/semantic-memory-server"
check(launcher.is_file(), "Claude MCP launcher is missing")
check(launcher.stat().st_mode & 0o111 != 0, "Claude MCP launcher is not executable")

for skill in [
    "claude-plugin/skills/semantic-memory/SKILL.md",
    "codex/.agents/skills/semantic-memory/SKILL.md",
]:
    metadata = frontmatter(skill)
    check(metadata.get("name") == "semantic-memory", f"{skill}: name mismatch")
    check(bool(metadata.get("description")), f"{skill}: description is required")

codex = load_toml("codex/config.example.toml")
server = codex.get("mcp_servers", {}).get("semantic_memory", {})
check(server.get("command") == "semantic-memory-mcp", "Codex command mismatch")
check("--memory-dir" in server.get("args", []), "Codex args need --memory-dir")
check("--tool-profile" in server.get("args", []), "Codex args need --tool-profile")

for path in ROOT.rglob("*"):
    if not path.is_file() or "__pycache__" in path.parts:
        continue
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        continue
    unix_home = "/" + "home" + "/"
    mac_home = "/" + "Users" + "/"
    check(unix_home not in text, f"{path.relative_to(REPO)}: hard-coded Unix home path")
    check(mac_home not in text, f"{path.relative_to(REPO)}: hard-coded macOS home path")

if ERRORS:
    for message in ERRORS:
        print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)

print("integration assets: valid")
