#!/usr/bin/env python3
"""Focused regression checks for the semantic-memory MCP stdio/TCP relay."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


RELAY_PATH = Path(__file__).with_name("semantic-memory-mcp-relay.py")
spec = importlib.util.spec_from_file_location("semantic_memory_mcp_relay", RELAY_PATH)
assert spec is not None and spec.loader is not None
relay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(relay)


class ContentLengthParserTests(unittest.TestCase):
    def test_accepts_complete_nonnegative_frame(self) -> None:
        body = b'{"jsonrpc":"2.0","id":1}'
        frame = b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
        self.assertEqual(relay.parse_content_length_stream(frame, 0), (body, len(frame)))

    def test_rejects_negative_content_length(self) -> None:
        with self.assertRaises(ValueError):
            relay.parse_content_length_stream(b"Content-Length: -1\r\n\r\n", 0)

    def test_rejects_oversized_content_length(self) -> None:
        oversized = relay.MAX_CONTENT_LENGTH + 1
        with self.assertRaises(ValueError):
            relay.parse_content_length_stream(
                f"Content-Length: {oversized}\r\n\r\n".encode(), 0
            )


if __name__ == "__main__":
    unittest.main()
