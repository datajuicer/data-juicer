import json
import os
import time
import unittest
from unittest.mock import MagicMock, patch

from data_juicer.ops.mixins import EventDrivenMixin, NotificationMixin
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


# Concrete classes for testing (mixins need a base)
class EventDrivenTestClass(EventDrivenMixin):
    pass


class _KwargsAbsorber:
    """Base class that absorbs any kwargs so object.__init__ won't complain."""
    def __init__(self, *args, **kwargs):
        pass


class NotificationTestClass(NotificationMixin, _KwargsAbsorber):
    pass


class EventDrivenMixinTest(DataJuicerTestCaseBase):

    def test_register_and_trigger_event(self):
        obj = EventDrivenTestClass()
        results = []
        obj.register_event_handler("test_event", lambda d: results.append(d))
        obj.trigger_event("test_event", {"key": "value"})
        self.assertEqual(results, [{"key": "value"}])

    def test_register_multiple_handlers(self):
        obj = EventDrivenTestClass()
        results = []
        obj.register_event_handler("evt", lambda d: results.append("a"))
        obj.register_event_handler("evt", lambda d: results.append("b"))
        obj.trigger_event("evt", {})
        self.assertEqual(results, ["a", "b"])

    def test_trigger_unregistered_event_does_nothing(self):
        obj = EventDrivenTestClass()
        # Should not raise
        obj.trigger_event("nonexistent", {"data": 1})

    def test_start_and_stop_polling(self):
        obj = EventDrivenTestClass()
        call_count = {"n": 0}

        def poll_func():
            call_count["n"] += 1
            return {"polled": True}

        triggered = []
        obj.register_event_handler("poll_evt", lambda d: triggered.append(d))

        obj.start_polling("poll_evt", poll_func, interval=0.05)
        time.sleep(0.2)
        obj.stop_polling("poll_evt")

        self.assertGreater(call_count["n"], 0)
        self.assertGreater(len(triggered), 0)
        self.assertEqual(triggered[0], {"polled": True})
        self.assertNotIn("poll_evt", obj.polling_threads)

    def test_start_polling_already_running(self):
        obj = EventDrivenTestClass()
        obj.start_polling("evt", lambda: None, interval=0.5)
        thread1 = obj.polling_threads["evt"]

        # Starting again should not create a new thread
        obj.start_polling("evt", lambda: None, interval=0.5)
        thread2 = obj.polling_threads["evt"]

        self.assertIs(thread1, thread2)
        obj.stop_polling("evt")

    def test_polling_handles_exception(self):
        obj = EventDrivenTestClass()
        call_count = {"n": 0}

        def failing_poll():
            call_count["n"] += 1
            raise ValueError("poll error")

        obj.start_polling("err_evt", failing_poll, interval=0.05)
        time.sleep(0.2)
        obj.stop_polling("err_evt")

        # Should have been called multiple times despite exceptions
        self.assertGreater(call_count["n"], 1)

    def test_polling_returns_none_no_trigger(self):
        obj = EventDrivenTestClass()
        triggered = []
        obj.register_event_handler("evt", lambda d: triggered.append(d))

        obj.start_polling("evt", lambda: None, interval=0.05)
        time.sleep(0.15)
        obj.stop_polling("evt")

        # None return should not trigger events
        self.assertEqual(len(triggered), 0)

    def test_stop_all_polling(self):
        obj = EventDrivenTestClass()
        obj.start_polling("evt1", lambda: None, interval=1)
        obj.start_polling("evt2", lambda: None, interval=1)

        self.assertEqual(len(obj.polling_threads), 2)
        obj.stop_all_polling()
        self.assertEqual(len(obj.polling_threads), 0)

    def test_wait_for_completion_success(self):
        counter = {"n": 0}

        def condition():
            counter["n"] += 1
            return counter["n"] >= 3

        obj = EventDrivenTestClass()
        result = obj.wait_for_completion(
            condition, timeout=5, poll_interval=0.05
        )
        self.assertTrue(result)
        self.assertGreaterEqual(counter["n"], 3)

    def test_wait_for_completion_timeout(self):
        obj = EventDrivenTestClass()
        with self.assertRaises(TimeoutError) as ctx:
            obj.wait_for_completion(
                lambda: False,
                timeout=0.1,
                poll_interval=0.02,
                error_message="custom timeout msg",
            )
        self.assertIn("custom timeout msg", str(ctx.exception))

    def test_wait_for_completion_immediate(self):
        obj = EventDrivenTestClass()
        result = obj.wait_for_completion(lambda: True, timeout=1)
        self.assertTrue(result)


class NotificationMixinTest(DataJuicerTestCaseBase):

    def _make_obj(self, **config):
        return NotificationTestClass(notification_config=config)

    def test_init_disabled_by_default(self):
        obj = self._make_obj()
        self.assertEqual(obj.notification_config, {})

    def test_init_enabled(self):
        obj = self._make_obj(enabled=True)
        self.assertTrue(obj.notification_config["enabled"])

    def test_send_notification_disabled(self):
        obj = self._make_obj(enabled=False)
        result = obj.send_notification("hello", notification_type="email")
        self.assertTrue(result)  # Returns True when disabled

    def test_send_notification_no_config(self):
        obj = NotificationTestClass()
        result = obj.send_notification("hello", notification_type="email")
        self.assertTrue(result)

    def test_send_notification_none_type(self):
        obj = self._make_obj(enabled=True)
        mock_handler = MagicMock(return_value=True)
        obj.notification_handlers = {"email": mock_handler}
        # notification_type=None must dispatch to no handler at all
        obj.send_notification("hello", notification_type=None)
        mock_handler.assert_not_called()

    def test_send_notification_unsupported_type(self):
        obj = self._make_obj(enabled=True)
        result = obj.send_notification("hello", notification_type="telegram")
        self.assertFalse(result)

    def test_send_notification_channel_disabled(self):
        obj = self._make_obj(
            enabled=True,
            email={"enabled": False},
        )
        mock_handler = MagicMock(return_value=True)
        obj.notification_handlers["email"] = mock_handler
        result = obj.send_notification("hello", notification_type="email")
        # disabled channel returns True but must NOT actually send
        self.assertTrue(result)
        mock_handler.assert_not_called()

    def test_send_notification_kwargs_override(self):
        obj = self._make_obj(
            enabled=True,
            email={"smtp_server": "original.com"},
        )

        mock_handler = MagicMock(return_value=True)
        obj.notification_handlers["email"] = mock_handler

        obj.send_notification(
            "hello",
            notification_type="email",
            email={"smtp_server": "override.com"},
        )
        mock_handler.assert_called_once()

        # After the call, the top-level notification_config reference
        # should be restored (the finally block restores it).
        self.assertTrue(obj.notification_config["enabled"])

    @patch("smtplib.SMTP_SSL")
    def test_send_email_ssl_with_password(self, mock_smtp_ssl):
        mock_server = MagicMock()
        mock_smtp_ssl.return_value.__enter__ = MagicMock(
            return_value=mock_server
        )
        mock_smtp_ssl.return_value.__exit__ = MagicMock(return_value=False)

        obj = self._make_obj(
            enabled=True,
            email={
                "smtp_server": "smtp.test.com",
                "smtp_port": 465,
                "use_ssl": True,
                "username": "user@test.com",
                "sender_email": "user@test.com",
                "recipients": ["dest@test.com"],
                "password": "secret123",
            },
        )
        result = obj.send_notification("test msg", notification_type="email")
        self.assertTrue(result)
        mock_smtp_ssl.assert_called_once()

    @patch("smtplib.SMTP")
    def test_send_email_starttls_with_password(self, mock_smtp):
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp.return_value.__exit__ = MagicMock(return_value=False)

        obj = self._make_obj(
            enabled=True,
            email={
                "smtp_server": "smtp.test.com",
                "smtp_port": 587,
                "use_ssl": False,
                "username": "user@test.com",
                "sender_email": "user@test.com",
                "recipients": ["dest@test.com"],
                "password": "secret123",
            },
        )
        result = obj.send_notification("test msg", notification_type="email")
        self.assertTrue(result)
        mock_smtp.assert_called_once()

    def test_send_email_missing_server(self):
        obj = self._make_obj(
            enabled=True,
            email={
                "smtp_server": "",
                "recipients": ["dest@test.com"],
                "password": "pw",
            },
        )
        result = obj.send_notification("test", notification_type="email")
        self.assertFalse(result)

    def test_send_email_missing_credentials(self):
        obj = self._make_obj(
            enabled=True,
            email={
                "smtp_server": "smtp.test.com",
                "recipients": ["dest@test.com"],
                # no password, no cert
            },
        )
        result = obj.send_notification("test", notification_type="email")
        self.assertFalse(result)

    def test_send_email_missing_cert_files(self):
        obj = self._make_obj(
            enabled=True,
            email={
                "smtp_server": "smtp.test.com",
                "recipients": ["dest@test.com"],
                "use_cert_auth": True,
                # no cert/key files
            },
        )
        result = obj.send_notification("test", notification_type="email")
        self.assertFalse(result)

    @patch("smtplib.SMTP")
    @patch("ssl.create_default_context")
    def test_send_email_starttls_with_cert(self, mock_ssl_ctx, mock_smtp):
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp.return_value.__exit__ = MagicMock(return_value=False)
        mock_ctx = MagicMock()
        mock_ssl_ctx.return_value = mock_ctx

        obj = self._make_obj(
            enabled=True,
            email={
                "smtp_server": "smtp.test.com",
                "smtp_port": 587,
                "use_ssl": False,
                "use_cert_auth": True,
                "client_cert_file": "/fake/cert.pem",
                "client_key_file": "/fake/key.pem",
                "sender_email": "user@test.com",
                "recipients": ["dest@test.com"],
            },
        )
        result = obj.send_notification("test msg", notification_type="email")
        self.assertTrue(result)
        mock_ctx.load_cert_chain.assert_called_once_with(
            certfile="/fake/cert.pem", keyfile="/fake/key.pem"
        )

    @patch("smtplib.SMTP_SSL")
    @patch("ssl.create_default_context")
    def test_send_email_ssl_with_cert(self, mock_ssl_ctx, mock_smtp_ssl):
        mock_server = MagicMock()
        mock_smtp_ssl.return_value.__enter__ = MagicMock(
            return_value=mock_server
        )
        mock_smtp_ssl.return_value.__exit__ = MagicMock(return_value=False)
        mock_ctx = MagicMock()
        mock_ssl_ctx.return_value = mock_ctx

        obj = self._make_obj(
            enabled=True,
            email={
                "smtp_server": "smtp.test.com",
                "smtp_port": 465,
                "use_ssl": True,
                "use_cert_auth": True,
                "client_cert_file": "/fake/cert.pem",
                "client_key_file": "/fake/key.pem",
                "sender_email": "user@test.com",
                "recipients": ["dest@test.com"],
            },
        )
        result = obj.send_notification("test msg", notification_type="email")
        self.assertTrue(result)

    def test_send_email_password_from_env(self):
        env_vars = {
            "DATA_JUICER_EMAIL_PASSWORD": "env_password",
            "DATA_JUICER_SMTP_TEST_COM_PASSWORD": "",
            "DATA_JUICER_EMAIL_CERT": "",
            "DATA_JUICER_EMAIL_KEY": "",
        }
        with patch.dict(os.environ, env_vars, clear=False), \
             patch("smtplib.SMTP_SSL") as mock_smtp_ssl:
            mock_server = MagicMock()
            mock_smtp_ssl.return_value.__enter__ = MagicMock(
                return_value=mock_server
            )
            mock_smtp_ssl.return_value.__exit__ = MagicMock(return_value=False)

            obj = self._make_obj(
                enabled=True,
                email={
                    "smtp_server": "smtp.test.com",
                    "smtp_port": 465,
                    "use_ssl": True,
                    "username": "user@test.com",
                    "sender_email": "user@test.com",
                    "recipients": ["dest@test.com"],
                    # no password in config; should use env var
                },
            )
            result = obj.send_notification("test msg", notification_type="email")
            self.assertTrue(result)

    def test_send_email_sender_name_formatting(self):
        obj = self._make_obj(
            enabled=True,
            email={
                "smtp_server": "smtp.test.com",
                "smtp_port": 465,
                "sender_email": "user@test.com",
                "sender_name": "Test User",
                "username": "user@test.com",
                "password": "pw",
                "recipients": ["dest@test.com"],
            },
        )
        # Patch SMTP to avoid real connection but verify the message
        with patch("smtplib.SMTP_SSL") as mock_ssl:
            mock_server = MagicMock()
            mock_ssl.return_value.__enter__ = MagicMock(
                return_value=mock_server
            )
            mock_ssl.return_value.__exit__ = MagicMock(return_value=False)
            obj.send_notification("test", notification_type="email")

            # Check the sendmail call has formatted sender
            call_args = mock_server.sendmail.call_args
            self.assertIn("Test User", call_args[0][0])

    def test_send_email_exception_returns_false(self):
        obj = self._make_obj(
            enabled=True,
            email={
                "smtp_server": "smtp.test.com",
                "smtp_port": 465,
                "username": "u",
                "password": "p",
                "sender_email": "u@t.com",
                "recipients": ["d@t.com"],
            },
        )
        with patch("smtplib.SMTP_SSL", side_effect=Exception("conn fail")):
            result = obj.send_notification("test", notification_type="email")
            self.assertFalse(result)

    @patch("requests.post")
    def test_send_slack_notification_success(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200)

        obj = self._make_obj(
            enabled=True,
            slack={
                "webhook_url": "https://hooks.slack.com/test",
                "channel": "#test",
            },
        )
        result = obj.send_notification("hello slack", notification_type="slack")
        self.assertTrue(result)
        mock_post.assert_called_once()
        # url is the first positional arg; payload is JSON in the `data` kwarg
        self.assertEqual(
            mock_post.call_args[0][0], "https://hooks.slack.com/test")
        payload = json.loads(mock_post.call_args[1]["data"])
        self.assertEqual(payload["text"], "hello slack")
        self.assertEqual(payload["channel"], "#test")

    @patch("requests.post")
    def test_send_slack_notification_failure(self, mock_post):
        mock_post.return_value = MagicMock(status_code=500)

        obj = self._make_obj(
            enabled=True,
            slack={"webhook_url": "https://hooks.slack.com/test"},
        )
        result = obj.send_notification("hello", notification_type="slack")
        self.assertFalse(result)

    def test_send_slack_missing_webhook(self):
        obj = self._make_obj(enabled=True, slack={})
        result = obj.send_notification("hello", notification_type="slack")
        self.assertFalse(result)

    def test_send_slack_exception_returns_false(self):
        obj = self._make_obj(
            enabled=True,
            slack={"webhook_url": "https://hooks.slack.com/test"},
        )
        with patch("requests.post", side_effect=Exception("network error")):
            result = obj.send_notification("hello", notification_type="slack")
            self.assertFalse(result)

    @patch("requests.post")
    def test_send_dingtalk_notification_success(self, mock_post):
        mock_post.return_value = MagicMock(
            json=MagicMock(return_value={"errcode": 0})
        )

        obj = self._make_obj(
            enabled=True,
            dingtalk={"access_token": "fake_token"},
        )
        result = obj.send_notification(
            "hello dingtalk", notification_type="dingtalk"
        )
        self.assertTrue(result)
        mock_post.assert_called_once()

    @patch("requests.post")
    def test_send_dingtalk_with_secret(self, mock_post):
        mock_post.return_value = MagicMock(
            json=MagicMock(return_value={"errcode": 0})
        )

        obj = self._make_obj(
            enabled=True,
            dingtalk={
                "access_token": "fake_token",
                "secret": "fake_secret",
            },
        )
        result = obj.send_notification("hello", notification_type="dingtalk")
        self.assertTrue(result)

        # URL should contain timestamp and sign params
        call_url = mock_post.call_args[0][0]
        self.assertIn("timestamp=", call_url)
        self.assertIn("sign=", call_url)

    def test_send_dingtalk_missing_token(self):
        obj = self._make_obj(enabled=True, dingtalk={})
        result = obj.send_notification("hello", notification_type="dingtalk")
        self.assertFalse(result)

    def test_send_dingtalk_exception_returns_false(self):
        obj = self._make_obj(
            enabled=True,
            dingtalk={"access_token": "fake_token"},
        )
        with patch("requests.post", side_effect=Exception("fail")):
            result = obj.send_notification(
                "hello", notification_type="dingtalk"
            )
            self.assertFalse(result)

    @patch("requests.post")
    def test_send_dingtalk_api_error(self, mock_post):
        mock_post.return_value = MagicMock(
            json=MagicMock(return_value={"errcode": 310000, "errmsg": "fail"})
        )
        obj = self._make_obj(
            enabled=True,
            dingtalk={"access_token": "fake_token"},
        )
        result = obj.send_notification("hello", notification_type="dingtalk")
        self.assertFalse(result)

    def test_notification_handlers_not_initialized(self):
        """Test when notification_handlers is missing."""
        obj = self._make_obj(enabled=True)
        del obj.notification_handlers
        result = obj.send_notification("hello", notification_type="email")
        self.assertFalse(result)

    @patch("smtplib.SMTP_SSL")
    def test_send_email_no_include_port(self, mock_smtp_ssl):
        mock_server = MagicMock()
        mock_smtp_ssl.return_value.__enter__ = MagicMock(
            return_value=mock_server
        )
        mock_smtp_ssl.return_value.__exit__ = MagicMock(return_value=False)

        obj = self._make_obj(
            enabled=True,
            email={
                "smtp_server": "smtp.test.com",
                "smtp_port": 465,
                "use_ssl": True,
                "include_port_in_address": False,
                "username": "user@test.com",
                "sender_email": "user@test.com",
                "password": "pw",
                "recipients": ["dest@test.com"],
            },
        )
        result = obj.send_notification("test", notification_type="email")
        self.assertTrue(result)
        # Server address should not include port
        call_args = mock_smtp_ssl.call_args
        self.assertEqual(call_args[0][0], "smtp.test.com")

    def test_send_email_cert_from_env(self):
        env_vars = {
            "DATA_JUICER_EMAIL_CERT": "/env/cert.pem",
            "DATA_JUICER_EMAIL_KEY": "/env/key.pem",
            "DATA_JUICER_EMAIL_PASSWORD": "",
            "DATA_JUICER_SMTP_TEST_COM_PASSWORD": "",
        }
        with patch.dict(os.environ, env_vars, clear=False), \
             patch("smtplib.SMTP_SSL") as mock_smtp_ssl, \
             patch("ssl.create_default_context") as mock_ssl_ctx:
            mock_server = MagicMock()
            mock_smtp_ssl.return_value.__enter__ = MagicMock(
                return_value=mock_server
            )
            mock_smtp_ssl.return_value.__exit__ = MagicMock(return_value=False)
            mock_ctx = MagicMock()
            mock_ssl_ctx.return_value = mock_ctx

            obj = self._make_obj(
                enabled=True,
                email={
                    "smtp_server": "smtp.test.com",
                    "smtp_port": 465,
                    "use_ssl": True,
                    "use_cert_auth": True,
                    "sender_email": "user@test.com",
                    "recipients": ["dest@test.com"],
                    # cert/key from env, not config
                },
            )
            result = obj.send_notification("test", notification_type="email")
            self.assertTrue(result)
            mock_ctx.load_cert_chain.assert_called_once_with(
                certfile="/env/cert.pem", keyfile="/env/key.pem"
            )


if __name__ == "__main__":
    unittest.main()
