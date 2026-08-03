# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from data_juicer.ops.mapper.pii_redaction_mapper import (
    PLACEHOLDER_IP,
    PLACEHOLDER_JWT,
    PLACEHOLDER_MAC,
    PLACEHOLDER_PEM,
    PLACEHOLDER_URL,
    PiiRedactionMapper,
)
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class TestPiiRedactionExtended(DataJuicerTestCaseBase):
    def _m(self, **kwargs):
        return PiiRedactionMapper(text_key="text", **kwargs)

    def test_extended_url_jwt_ip(self):
        m = self._m(mask_urls=True)  # URL off by default, explicitly enable
        raw = (
            "see https://api.example.com/v1?token=secret "
            "and eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.sig "
            "host 203.0.113.5 ok"
        )
        out = m._redact_text(raw)
        self.assertIn(PLACEHOLDER_URL, out)
        self.assertIn(PLACEHOLDER_JWT, out)
        self.assertIn(PLACEHOLDER_IP, out)

    def test_extended_ipv6_mac_pem(self):
        m = self._m()  # PEM/IP/MAC on by default
        raw = (
            "addr [2001:db8::1]:443 mac aa:bb:cc:dd:ee:ff key:\n"
            "-----BEGIN PRIVATE KEY-----\nMIIE\n-----END PRIVATE KEY-----"
        )
        out = m._redact_text(raw)
        self.assertIn(PLACEHOLDER_IP, out)
        self.assertIn(PLACEHOLDER_MAC, out)
        self.assertIn(PLACEHOLDER_PEM, out)

    def test_extended_url_off_by_default(self):
        """URL masking is off by default, but JWT and IP are on."""
        m = self._m()
        raw = "https://x.com eyJh.eyJh.sig 1.2.3.4"
        out = m._redact_text(raw)
        # URL not replaced (mask_urls=False by default)
        self.assertIn("https://x.com", out)
        # JWT and IP are replaced by default
        self.assertIn(PLACEHOLDER_JWT, out)
        self.assertIn(PLACEHOLDER_IP, out)

    def test_extra_patterns_ignore_invalid_regex(self):
        m = self._m(extra_patterns=[(r"ticket-\d+", "[TICKET]"), ("(", "bad")])
        self.assertEqual(m._redact_text("see ticket-123"), "see [TICKET]")

    def test_redact_text_empty_and_disabled_masks(self):
        m = self._m(mask_emails=False, mask_paths=False, mask_secrets=False)
        self.assertEqual(m._redact_text(""), "")
        self.assertIsNone(m._redact_text(None))
        raw = "mail dev@example.com path /home/alice/a.txt token=abc"
        self.assertEqual(m._redact_text(raw), raw)

    def test_redact_value_recurses_nested_dicts_and_lists(self):
        m = self._m()
        payload = {
            "safe": "dev@example.com",
            "tool": {
                "arguments": "open /tmp/secret.txt",
                "nested": [{"path": "C:\\Users\\alice\\secret.txt"}],
            },
            "items": ["call 13800138000", {"file_path": "/home/bob/data.json"}],
        }
        m._redact_value(payload)
        self.assertIn("dev@example.com", payload["safe"])
        self.assertIn("[PATH_REDACTED]", payload["tool"]["arguments"])
        self.assertIn("[PATH_REDACTED]", payload["tool"]["nested"][0]["path"])
        self.assertIn("[PHONE_REDACTED]", payload["items"][0])
        self.assertIn("[PATH_REDACTED]", payload["items"][1]["file_path"])

    def test_redact_messages_handles_content_blocks_and_tool_calls(self):
        m = self._m(mask_urls=True)
        messages = [
            "ignore",
            {
                "content": [
                    {"text": "email dev@example.com"},
                    {"content": "url https://example.com/a?token=x"},
                    "phone +1 5551234567",
                ],
                "tool_calls": [
                    {"function": {"arguments": "path=/home/alice/file.txt"}},
                    {"arguments": {"path": "\\\\server\\share\\file.txt"}},
                    "ignore",
                ],
                "tool_use": [
                    {"function_call": {"arguments": {"file": "/tmp/x.txt"}}},
                    {"arguments": "token=abc"},
                ],
            },
            {"content": "id session_id=abc123"},
        ]
        m._redact_messages(messages)
        msg = messages[1]
        self.assertIn("[EMAIL_REDACTED]", msg["content"][0]["text"])
        self.assertIn("[URL_REDACTED]", msg["content"][1]["content"])
        self.assertIn("[PHONE_REDACTED]", msg["content"][2])
        self.assertIn("[PATH_REDACTED]", msg["tool_calls"][0]["function"]["arguments"])
        self.assertIn("[PATH_REDACTED]", msg["tool_calls"][1]["arguments"]["path"])
        self.assertIn("[PATH_REDACTED]", msg["tool_use"][0]["function_call"]["arguments"]["file"])
        self.assertIn("[REDACTED]", msg["tool_use"][1]["arguments"])
        self.assertIn("[ID_REDACTED]", messages[2]["content"])

    def test_redact_messages_ignores_non_list(self):
        m = self._m()
        self.assertIsNone(m._redact_messages({"content": "dev@example.com"}))

    def test_process_single_redacts_default_keys_and_dialog_history(self):
        m = self._m(mask_urls=True)
        sample = {
            "text": "path /home/alice/a.txt",
            "query": "mail dev@example.com",
            "response": "url https://example.com/x",
            "dialog_history": [
                ("phone 13800138000", "token=secret"),
                ["keep", "/tmp/path.txt"],
                "bad-shape",
            ],
            "messages": [{"content": "trace_id=abc"}],
            "nested": {"arguments": "file /var/log/a.log"},
        }
        out = m.process_single(sample)
        self.assertIn("[PATH_REDACTED]", out["text"])
        self.assertIn("[EMAIL_REDACTED]", out["query"])
        self.assertIn("[URL_REDACTED]", out["response"])
        self.assertIn("[PHONE_REDACTED]", out["dialog_history"][0][0])
        self.assertIn("[REDACTED]", out["dialog_history"][0][1])
        self.assertIsInstance(out["dialog_history"][1], tuple)
        self.assertIn("[PATH_REDACTED]", out["dialog_history"][1][1])
        self.assertEqual(out["dialog_history"][2], "bad-shape")
        self.assertIn("[ID_REDACTED]", out["messages"][0]["content"])
        self.assertIn("[PATH_REDACTED]", out["nested"]["arguments"])

    def test_process_single_preserves_pair_container_type(self):
        m = self._m(redact_keys=["pair_list", "pair_tuple"], messages_key=None)
        out = m.process_single(
            {
                "pair_list": ["dev@example.com", "/home/alice/a.txt"],
                "pair_tuple": ("token=abc", 3),
            }
        )
        self.assertIsInstance(out["pair_list"], list)
        self.assertIn("[EMAIL_REDACTED]", out["pair_list"][0])
        self.assertIsInstance(out["pair_tuple"], tuple)
        self.assertIn("[REDACTED]", out["pair_tuple"][0])


if __name__ == "__main__":
    unittest.main()
