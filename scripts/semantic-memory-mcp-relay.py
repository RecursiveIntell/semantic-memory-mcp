#!/usr/bin/env python3
"""
semantic-memory-mcp-relay.py — stdio MCP relay to the warm TCP MCP agent port.

The semantic-memory binary can run in --http-only mode with separate TCP MCP
ports (--mcp-agent-port, --mcp-admin-port). This relay bridges a Hermes stdio
MCP connection to the agent TCP port (newline-delimited JSON-RPC), so there is
a single store owner (the warm process) and Hermes gets its stdio MCP interface.

Usage by Hermes config.yaml:
  mcp_servers:
    semantic_memory:
      command: /path/to/semantic-memory-mcp-relay.py
      args: ["--port", "17540"]

Protocol translation:
  - stdin (from Hermes): newline-delimited JSON-RPC (the current MCP Python
    SDK transport) or legacy Content-Length framing → raw JSON + newline to
    TCP socket
  - stdout (to Hermes): read newline-delimited JSON from TCP socket and emit
    the framing style detected on stdin

Fail-open: if the TCP port is unreachable, exits non-zero so Hermes reports
the failure rather than silently providing no tools.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import threading
from typing import Optional

DEFAULT_PORT = 17540
DEFAULT_HOST = "127.0.0.1"
# Bound legacy Content-Length frames so a malformed stdio peer cannot force
# unbounded buffering before the relay forwards a complete JSON-RPC message.
MAX_CONTENT_LENGTH = 16 * 1024 * 1024


class Framing:
    """The MCP stdio framing selected by the first complete request."""

    def __init__(self) -> None:
        self.mode: Optional[str] = None
        self.ready = threading.Event()

    def select(self, mode: str) -> None:
        if self.mode is None:
            self.mode = mode
            self.ready.set()


def parse_args() -> tuple[str, int]:
    host = os.environ.get("SEMANTIC_MEMORY_MCP_HOST", DEFAULT_HOST)
    port = DEFAULT_PORT
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--port" and i + 1 < len(args):
            port = int(args[i + 1])
            i += 2
        elif args[i] == "--host" and i + 1 < len(args):
            host = args[i + 1]
            i += 2
        else:
            i += 1
    return host, port


def parse_content_length_stream(data: bytes, pos: int) -> Optional[tuple[bytes, int]]:
    """Parse one legacy Content-Length frame, or return None if incomplete."""
    end = data.find(b"\r\n\r\n", pos)
    if end == -1:
        return None
    header_block = data[pos:end].decode("ascii", errors="replace")
    content_length = None
    for line in header_block.split("\r\n"):
        if line.lower().startswith("content-length:"):
            try:
                content_length = int(line.split(":", 1)[1].strip())
            except ValueError as exc:
                raise ValueError("invalid Content-Length header") from exc
    if content_length is None:
        raise ValueError("missing Content-Length header")
    if content_length < 0:
        raise ValueError("Content-Length must not be negative")
    if content_length > MAX_CONTENT_LENGTH:
        raise ValueError(
            f"Content-Length exceeds maximum frame size ({MAX_CONTENT_LENGTH} bytes)"
        )
    body_start = end + 4
    body_end = body_start + content_length
    if body_end > len(data):
        return None
    return data[body_start:body_end], body_end


def stdin_to_tcp(sock: socket.socket, stop_event: threading.Event, framing: Framing) -> None:
    """Read MCP stdio requests and send newline-delimited JSON to TCP.

    The current Python MCP SDK uses JSON Lines for stdio. Accept legacy
    Content-Length framing as well because the relay is also used by older
    clients and standalone test harnesses. Responses mirror the client style.
    """
    buffer = b""
    header = b"content-length:"
    try:
        while not stop_event.is_set():
            chunk = os.read(sys.stdin.buffer.fileno(), 65536)
            if not chunk:
                break
            buffer += chunk
            pos = 0
            while True:
                remaining = buffer[pos:]
                if not remaining:
                    break
                lower = remaining.lower()
                if header.startswith(lower):
                    # A Content-Length header arrived in fragments.
                    break
                if lower.startswith(header):
                    result = parse_content_length_stream(buffer, pos)
                    if result is None:
                        break
                    msg, new_pos = result
                    framing.select("content-length")
                else:
                    newline = buffer.find(b"\n", pos)
                    if newline == -1:
                        break
                    msg = buffer[pos:newline].strip()
                    new_pos = newline + 1
                    if not msg:
                        pos = new_pos
                        continue
                    framing.select("jsonl")
                # The warm daemon accepts newline-delimited JSON-RPC.
                sock.sendall(msg.rstrip() + b"\n")
                pos = new_pos
            buffer = buffer[pos:]
    except (OSError, BrokenPipeError, ValueError) as exc:
        print(f"semantic-memory-mcp-relay: stdin forwarding failed: {exc}", file=sys.stderr)
    finally:
        try:
            sock.shutdown(socket.SHUT_WR)
        except OSError:
            pass


def tcp_to_stdout(sock: socket.socket, stop_event: threading.Event, framing: Framing) -> None:
    """Read TCP JSON Lines and write the matching stdio framing style."""
    buffer = b""
    try:
        if not framing.ready.wait(timeout=10):
            return
        while not stop_event.is_set():
            chunk = sock.recv(65536)
            if not chunk:
                break
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                if framing.mode == "content-length":
                    framed = f"Content-Length: {len(line)}\r\n\r\n".encode("ascii") + line
                else:
                    framed = line + b"\n"
                sys.stdout.buffer.write(framed)
                sys.stdout.buffer.flush()
    except (OSError, BrokenPipeError):
        pass
    finally:
        try:
            sock.shutdown(socket.SHUT_RD)
        except OSError:
            pass


def main() -> int:
    host, port = parse_args()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(10)
    try:
        sock.connect((host, port))
    except (OSError, ConnectionRefusedError) as exc:
        print(f"semantic-memory-mcp-relay: cannot connect to {host}:{port}: {exc}", file=sys.stderr)
        print("Is the warm semantic-memory process running with --mcp-agent-port?", file=sys.stderr)
        return 1

    sock.settimeout(None)
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except OSError:
        pass

    stop_event = threading.Event()
    framing = Framing()
    t_in = threading.Thread(target=stdin_to_tcp, args=(sock, stop_event, framing), daemon=True)
    t_out = threading.Thread(target=tcp_to_stdout, args=(sock, stop_event, framing), daemon=True)
    t_in.start()
    t_out.start()

    t_in.join()
    stop_event.set()
    t_out.join(timeout=3)
    try:
        sock.close()
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
