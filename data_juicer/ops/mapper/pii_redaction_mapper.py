# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# PII redaction for agent/dialog text: paths (Win/Unix), emails, secrets, IDs,
# agent channel identifiers (飞书/钉钉/企业微信/邮箱等). Configurable and
# extensible for multi-platform and custom patterns.

import re
from typing import Any, List, Optional, Tuple

from data_juicer.ops.base_op import OPERATORS, Mapper

OP_NAME = "pii_redaction_mapper"

PLACEHOLDER_PATH = "[PATH_REDACTED]"
PLACEHOLDER_EMAIL = "[EMAIL_REDACTED]"
PLACEHOLDER_SECRET = "[REDACTED]"
PLACEHOLDER_ID = "[ID_REDACTED]"
PLACEHOLDER_PHONE = "[PHONE_REDACTED]"
PLACEHOLDER_ID_CARD = "[ID_CARD_REDACTED]"
PLACEHOLDER_CHANNEL_ID = "[CHANNEL_ID_REDACTED]"
PLACEHOLDER_URL = "[URL_REDACTED]"
PLACEHOLDER_IP = "[IP_REDACTED]"
PLACEHOLDER_JWT = "[JWT_REDACTED]"
PLACEHOLDER_PEM = "[PEM_REDACTED]"
PLACEHOLDER_MAC = "[MAC_REDACTED]"

# Keys that often hold paths/emails/PII; redact when found in any nested dict
PII_VALUE_KEYS = ("file_path", "path", "file", "arguments", "file_paths")


def _compile_patterns() -> dict:
    """Build regex patterns for multi-platform and common PII."""
    ipv4 = r"(?<![0-9.])" r"(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}" r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)" r"(?![0-9])"
    return {
        # Unix: /Users/..., /home/..., /tmp/..., /opt/...
        "path_unix": re.compile(
            r"(^|[\s\"'(\[=])(/(?:Users?|home|tmp|etc|var|opt|Applications)" r"[^\s\"')\]]*(?:/[^\s\"')\]]*)*)"
        ),
        # Windows: C:\, D:\, \\server\share
        "path_win": re.compile(r"(^|[\s\"'(\[=])([A-Za-z]:\\[^\s\"')\]]*(?:\\[^\s\"')\]]*)*)"),
        "path_win_unc": re.compile(r"(^|[\s\"'(\[=])(\\\\[^\s\"')\]]+)"),
        # Email: local@domain.tld (gwm.cn, qq.com, etc.)
        "email": re.compile(r"[a-zA-Z0-9][\w.+-]*@[\w.-]+\.(?:[a-zA-Z]{2,}|\w{2,})"),
        # Secret keys: key:= value or "key": "value"
        "secret_kv": re.compile(
            r"(\b(?:api[_-]?key|apikey|secret|password|passwd|token|auth|authorization"
            r"|credential|license[_-]?key)\s*[:=]\s*[\"']?)([^\s\"',}\]]+)",
            re.IGNORECASE,
        ),
        # Session/user/request IDs (session_id: 123, user_id: ou_xxx)
        "id_kv": re.compile(
            r"\b(session[_-]?id|user[_-]?id|request[_-]?id|trace[_-]?id)\s*[:=]\s*[\"']?" r"[\w.-]+",
            re.IGNORECASE,
        ),
        # Agent channel: 当前的channel: feishu / dingtalk / email / 飞书 / 钉钉
        "channel_kv": re.compile(
            r"(\bchannel\s*[:=]\s*[\"']?|当前的\s*channel\s*[:：]\s*)"
            r"(feishu|dingtalk|wecom|wechat_work|email|mail|飞书|钉钉|企业微信|邮箱)\b",
            re.IGNORECASE,
        ),
        # 飞书 open_id (ou_ + 32 hex)
        "feishu_open_id": re.compile(r"\bou_[0-9a-f]{32}\b", re.IGNORECASE),
        # 钉钉/企业微信 等常见 open_id 格式 (字母+数字+下划线，较长)
        "platform_open_id": re.compile(r"\b(ou_|u_|uid_)[0-9a-zA-Z_-]{16,64}\b"),
        # Chinese mobile (simple)
        "phone_cn": re.compile(r"\b1[3-9]\d{9}\b"),
        # Generic phone (international, simple)
        "phone_intl": re.compile(r"\+\d{1,4}[-.\s]?\d{6,14}\b"),
        # Chinese ID card (18 digits)
        "id_card_cn": re.compile(
            r"\b[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])" r"(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]\b"
        ),
        # PEM / SSH keys & certs (strip whole block first)
        "pem_block": re.compile(
            r"-----BEGIN [A-Z0-9 -]+-----[\s\S]*?-----END [A-Z0-9 -]+-----",
            re.MULTILINE,
        ),
        # Typical JWS (header often starts with eyJ = {"alg"...)
        "jwt": re.compile(
            r"\beyJ[A-Za-z0-9_=-]+\.[A-Za-z0-9_=-]+\.[A-Za-z0-9_=-]+\b",
        ),
        # http(s) URLs (query strings may carry tokens)
        "url": re.compile(r"https?://[^\s\"'<>]+"),
        "ipv4": re.compile(ipv4),
        # Bracketed IPv6 + optional :port
        "ipv6": re.compile(
            r"\[[0-9a-fA-F:.]{3,}\](?::\d{1,5})?\b",
        ),
        "mac": re.compile(
            r"(?<![0-9A-Fa-f:])(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b",
        ),
    }


@OPERATORS.register_module(OP_NAME)
class PiiRedactionMapper(Mapper):
    """Redact PII in text and optionally in messages/query/response.

    Covers paths (Unix/Windows), emails, secrets, IDs, phones, agent channel
    identifiers (飞书/钉钉/企业微信 open_id, channel: feishu|dingtalk|email).
    Optional: PEM blocks, JWT-shaped tokens, http(s) URLs, IPv4, bracketed
    IPv6, MAC addresses (see ``mask_extended_pii`` or individual flags).
    Use redact_keys to apply to text, query, response, and/or messages (recursive).
    """

    def __init__(
        self,
        mask_paths: bool = True,
        mask_emails: bool = True,
        mask_secrets: bool = True,
        mask_ids: bool = True,
        mask_phones: bool = True,
        mask_id_cards: bool = True,
        mask_channel_ids: bool = True,
        mask_platform_open_ids: bool = True,
        mask_pem: bool = True,
        mask_jwt: bool = True,
        mask_urls: bool = False,
        mask_ips: bool = True,
        mask_macs: bool = True,
        path_replacement: str = PLACEHOLDER_PATH,
        email_replacement: str = PLACEHOLDER_EMAIL,
        secret_replacement: str = PLACEHOLDER_SECRET,
        id_replacement: str = PLACEHOLDER_ID,
        phone_replacement: str = PLACEHOLDER_PHONE,
        id_card_replacement: str = PLACEHOLDER_ID_CARD,
        channel_id_replacement: str = PLACEHOLDER_CHANNEL_ID,
        pem_replacement: str = PLACEHOLDER_PEM,
        jwt_replacement: str = PLACEHOLDER_JWT,
        url_replacement: str = PLACEHOLDER_URL,
        ip_replacement: str = PLACEHOLDER_IP,
        mac_replacement: str = PLACEHOLDER_MAC,
        extra_patterns: Optional[List[Tuple[str, str]]] = None,
        text_key: str = "text",
        redact_keys: Optional[List[str]] = None,
        messages_key: Optional[str] = "messages",
        **kwargs,
    ):
        super().__init__(text_key=text_key, **kwargs)
        self.mask_paths = mask_paths
        self.mask_emails = mask_emails
        self.mask_secrets = mask_secrets
        self.mask_ids = mask_ids
        self.mask_phones = mask_phones
        self.mask_id_cards = mask_id_cards
        self.mask_channel_ids = mask_channel_ids
        self.mask_platform_open_ids = mask_platform_open_ids
        self.mask_pem = mask_pem
        self.mask_jwt = mask_jwt
        self.mask_urls = mask_urls
        self.mask_ips = mask_ips
        self.mask_macs = mask_macs
        self.path_replacement = path_replacement
        self.email_replacement = email_replacement
        self.secret_replacement = secret_replacement
        self.id_replacement = id_replacement
        self.phone_replacement = phone_replacement
        self.id_card_replacement = id_card_replacement
        self.channel_id_replacement = channel_id_replacement
        self.pem_replacement = pem_replacement
        self.jwt_replacement = jwt_replacement
        self.url_replacement = url_replacement
        self.ip_replacement = ip_replacement
        self.mac_replacement = mac_replacement
        self.extra_patterns = extra_patterns or []
        if redact_keys is not None:
            self.redact_keys = redact_keys
        else:
            self.redact_keys = [
                text_key,
                "query",
                "response",
                "dialog_history",
                "messages",
            ]
        self.messages_key = messages_key

        patterns = _compile_patterns()
        self._path_unix = patterns["path_unix"]
        self._path_win = patterns["path_win"]
        self._path_win_unc = patterns["path_win_unc"]
        self._email_re = patterns["email"]
        self._secret_kv = patterns["secret_kv"]
        self._id_kv = patterns["id_kv"]
        self._channel_kv = patterns["channel_kv"]
        self._feishu_open_id = patterns["feishu_open_id"]
        self._platform_open_id = patterns["platform_open_id"]
        self._phone_cn = patterns["phone_cn"]
        self._phone_intl = patterns["phone_intl"]
        self._id_card_cn = patterns["id_card_cn"]
        self._pem_block = patterns["pem_block"]
        self._jwt_re = patterns["jwt"]
        self._url_re = patterns["url"]
        self._ipv4_re = patterns["ipv4"]
        self._ipv6_re = patterns["ipv6"]
        self._mac_re = patterns["mac"]

        self._extra_compiled = []
        for pat_str, repl in self.extra_patterns:
            try:
                self._extra_compiled.append((re.compile(pat_str), repl))
            except re.error:
                pass

    def _redact_text(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return text
        # Order: broad blocks / URL-like first, then finer tokens.
        if self.mask_pem:
            text = self._pem_block.sub(self.pem_replacement, text)
        if self.mask_jwt:
            text = self._jwt_re.sub(self.jwt_replacement, text)
        if self.mask_urls:
            text = self._url_re.sub(self.url_replacement, text)
        if self.mask_ips:
            text = self._ipv4_re.sub(self.ip_replacement, text)
            text = self._ipv6_re.sub(self.ip_replacement, text)
        if self.mask_macs:
            text = self._mac_re.sub(self.mac_replacement, text)
        if self.mask_paths:
            text = self._path_unix.sub(r"\1" + self.path_replacement, text)
            text = self._path_win.sub(r"\1" + self.path_replacement, text)
            text = self._path_win_unc.sub(r"\1" + self.path_replacement, text)
        if self.mask_emails:
            text = self._email_re.sub(self.email_replacement, text)
        if self.mask_secrets:
            text = self._secret_kv.sub(r"\1" + self.secret_replacement, text)
        if self.mask_ids:
            text = self._id_kv.sub(r"\1" + self.id_replacement, text)
        if self.mask_channel_ids:
            text = self._channel_kv.sub(r"\1" + self.channel_id_replacement, text)
        if self.mask_platform_open_ids:
            text = self._feishu_open_id.sub(self.channel_id_replacement, text)
            text = self._platform_open_id.sub(self.channel_id_replacement, text)
        if self.mask_phones:
            text = self._phone_cn.sub(self.phone_replacement, text)
            text = self._phone_intl.sub(self.phone_replacement, text)
        if self.mask_id_cards:
            text = self._id_card_cn.sub(self.id_card_replacement, text)
        for pat, repl in self._extra_compiled:
            text = pat.sub(repl, text)
        return text

    def _redact_value(self, val: Any) -> None:
        """Recursively redact PII in dict/list; in-place. For str, use _redact_text."""
        if isinstance(val, dict):
            for k, v in list(val.items()):
                if k in PII_VALUE_KEYS and isinstance(v, str):
                    val[k] = self._redact_text(v)
                else:
                    self._redact_value(v)
        elif isinstance(val, list):
            for i, item in enumerate(val):
                if isinstance(item, str):
                    val[i] = self._redact_text(item)
                else:
                    self._redact_value(item)

    def _redact_messages(self, messages: Any) -> None:
        """In-place redact all string content inside messages (list of msg)."""
        if not isinstance(messages, list):
            return
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if content is not None:
                if isinstance(content, str):
                    msg["content"] = self._redact_text(content)
                elif isinstance(content, list):
                    for i, block in enumerate(content):
                        if isinstance(block, dict):
                            for k in ("text", "content"):
                                if k in block and isinstance(block[k], str):
                                    block[k] = self._redact_text(block[k])
                        elif isinstance(block, str):
                            content[i] = self._redact_text(block)
            # Redact tool_calls / tool_use (e.g. function.arguments with path/email)
            for key in ("tool_calls", "tool_use"):
                calls = msg.get(key)
                if not isinstance(calls, list):
                    continue
                for call in calls:
                    if not isinstance(call, dict):
                        continue
                    fn = call.get("function") or call.get("function_call")
                    if isinstance(fn, dict):
                        args = fn.get("arguments")
                        if isinstance(args, str):
                            fn["arguments"] = self._redact_text(args)
                        elif isinstance(args, dict):
                            self._redact_value(args)
                    # Some formats put arguments at top level of call
                    args = call.get("arguments")
                    if isinstance(args, str):
                        call["arguments"] = self._redact_text(args)
                    elif isinstance(args, dict):
                        self._redact_value(args)

    def process_single(self, sample: dict) -> dict:
        for key in self.redact_keys:
            if key not in sample:
                continue
            val = sample[key]
            if key == self.messages_key and isinstance(val, list):
                self._redact_messages(val)
                continue
            if isinstance(val, str):
                sample[key] = self._redact_text(val)
            elif isinstance(val, list) and key == "dialog_history":
                new_history = []
                for pair in val:
                    if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                        q, r = pair[0], pair[1]
                        if isinstance(q, str):
                            q = self._redact_text(q)
                        if isinstance(r, str):
                            r = self._redact_text(r)
                        new_history.append((q, r))
                    else:
                        new_history.append(pair)
                sample[key] = new_history
            elif isinstance(val, (list, tuple)) and len(val) == 2:
                q, r = val[0], val[1]
                redacted_q = self._redact_text(q) if isinstance(q, str) else q
                redacted_r = self._redact_text(r) if isinstance(r, str) else r
                if redacted_q is not q or redacted_r is not r:
                    # Preserve list vs. tuple (old code always wrote a tuple).
                    sample[key] = type(val)((redacted_q, redacted_r))
        # Redact PII in nested file_path/path/arguments anywhere in sample
        self._redact_value(sample)
        return sample
