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


if __name__ == "__main__":
    unittest.main()
