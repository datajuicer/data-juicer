# pii_redaction_mapper

Redact PII in text and optionally in messages/query/response.

Covers paths (Unix/Windows), emails, secrets, IDs, phones, agent channel
identifiers (飞书/钉钉/企业微信 open_id, channel: feishu|dingtalk|email).
Optional: PEM blocks, JWT-shaped tokens, http(s) URLs, IPv4, bracketed
IPv6, MAC addresses (see ``mask_extended_pii`` or individual flags).
Use redact_keys to apply to text, query, response, and/or messages (recursive).

对文本中的 PII 进行脱敏，并可选地对 messages/query/response 进行脱敏。

涵盖路径（Unix/Windows）、电子邮件、密钥、ID、电话号码、Agent 渠道标识符（飞书/钉钉/企业微信 open_id，channel: feishu|dingtalk|email）。可选项：PEM 块、JWT 格式的 token、http(s) URL、IPv4、带括号的 IPv6、MAC 地址（参见 ``mask_extended_pii`` 或各个独立标志）。使用 redact_keys 应用于 text、query、response 和/或 messages（递归）。

Type 算子类型: **mapper**

Tags 标签: cpu

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `mask_paths` | <class 'bool'> | `True` |  |
| `mask_emails` | <class 'bool'> | `True` |  |
| `mask_secrets` | <class 'bool'> | `True` |  |
| `mask_ids` | <class 'bool'> | `True` |  |
| `mask_phones` | <class 'bool'> | `True` |  |
| `mask_id_cards` | <class 'bool'> | `True` |  |
| `mask_channel_ids` | <class 'bool'> | `True` |  |
| `mask_platform_open_ids` | <class 'bool'> | `True` |  |
| `mask_pem` | <class 'bool'> | `True` |  |
| `mask_jwt` | <class 'bool'> | `True` |  |
| `mask_urls` | <class 'bool'> | `False` |  |
| `mask_ips` | <class 'bool'> | `True` |  |
| `mask_macs` | <class 'bool'> | `True` |  |
| `path_replacement` | <class 'str'> | `'[PATH_REDACTED]'` |  |
| `email_replacement` | <class 'str'> | `'[EMAIL_REDACTED]'` |  |
| `secret_replacement` | <class 'str'> | `'[REDACTED]'` |  |
| `id_replacement` | <class 'str'> | `'[ID_REDACTED]'` |  |
| `phone_replacement` | <class 'str'> | `'[PHONE_REDACTED]'` |  |
| `id_card_replacement` | <class 'str'> | `'[ID_CARD_REDACTED]'` |  |
| `channel_id_replacement` | <class 'str'> | `'[CHANNEL_ID_REDACTED]'` |  |
| `pem_replacement` | <class 'str'> | `'[PEM_REDACTED]'` |  |
| `jwt_replacement` | <class 'str'> | `'[JWT_REDACTED]'` |  |
| `url_replacement` | <class 'str'> | `'[URL_REDACTED]'` |  |
| `ip_replacement` | <class 'str'> | `'[IP_REDACTED]'` |  |
| `mac_replacement` | <class 'str'> | `'[MAC_REDACTED]'` |  |
| `extra_patterns` | typing.Optional[typing.List[typing.Tuple[str, str]]] | `None` |  |
| `text_key` | <class 'str'> | `'text'` |  |
| `redact_keys` | typing.Optional[typing.List[str]] | `None` |  |
| `messages_key` | typing.Optional[str] | `'messages'` |  |
| `kwargs` |  | `` |  |




## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/pii_redaction_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_pii_redaction_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)