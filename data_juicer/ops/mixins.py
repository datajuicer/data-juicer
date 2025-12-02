import threading
import time
from typing import Callable, Dict

from loguru import logger


class EventDrivenMixin:
    """Mixin for event-driven capabilities in operations.

    This mixin provides functionality for registering event handlers,
    triggering events, and managing event polling.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.event_handlers = {}
        self.polling_threads = {}
        self.stop_polling_flags = {}

    def register_event_handler(self, event_type: str, handler: Callable):
        """Register a handler for a specific event type.

        Args:
            event_type: Type of event to handle
            handler: Callback function to handle the event
        """
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)

    def trigger_event(self, event_type: str, data: Dict):
        """Trigger an event and call all registered handlers.

        Args:
            event_type: Type of event to trigger
            data: Event data to pass to handlers
        """
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                handler(data)

    def start_polling(self, event_type: str, poll_func: Callable, interval: int = 60):
        """Start polling for a specific event type.

        Args:
            event_type: Type of event to poll for
            poll_func: Function to call for polling
            interval: Polling interval in seconds
        """
        if event_type in self.polling_threads and self.polling_threads[event_type].is_alive():
            return

        self.stop_polling_flags[event_type] = False

        def polling_worker():
            while not self.stop_polling_flags.get(event_type, False):
                try:
                    result = poll_func()
                    if result:
                        self.trigger_event(event_type, result)
                except Exception as e:
                    import logging

                    logging.error(f"Error in polling for {event_type}: {e}")
                time.sleep(interval)

        thread = threading.Thread(target=polling_worker, daemon=True)
        self.polling_threads[event_type] = thread
        thread.start()

    def stop_polling(self, event_type: str):
        """Stop polling for a specific event type.

        Args:
            event_type: Type of event to stop polling for
        """
        if event_type in self.polling_threads:
            self.stop_polling_flags[event_type] = True
            self.polling_threads[event_type].join(timeout=1)
            del self.polling_threads[event_type]

    def stop_all_polling(self):
        """Stop all polling threads."""
        event_types = list(self.polling_threads.keys())
        for event_type in event_types:
            self.stop_polling(event_type)

    def wait_for_completion(
        self,
        condition_func: Callable[[], bool],
        timeout: int = 3600,
        poll_interval: int = 10,
        error_message: str = "Operation timed out",
    ):
        """Wait for a condition to be met.

        Args:
            condition_func: Function that returns True when condition is met
            timeout: Maximum time to wait in seconds
            poll_interval: Polling interval in seconds
            error_message: Error message to raise on timeout

        Raises:
            TimeoutError: If the condition is not met within the timeout
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if condition_func():
                return True
            time.sleep(poll_interval)

        raise TimeoutError(error_message)


class NotificationMixin:
    """Mixin for sending notifications through various channels.

    This mixin provides functionality for sending notifications via email,
    Slack, DingTalk, and other platforms.

    Notification configuration can be specified as a "notification_config"
    parameter within an operator (for backward compatibility):
    ```yaml
    process:
      - some_mapper:
          notification_config:
            enabled: true
            email:
              # ... email settings ...
    ```

    For security best practices, sensitive information like passwords and
    tokens should be provided via environment variables:

    - Email: set 'DATA_JUICER_EMAIL_PASSWORD' environment variable
      or service-specific 'DATA_JUICER_SMTP_SERVER_NAME_PASSWORD'
    - Slack: set 'DATA_JUICER_SLACK_WEBHOOK' environment variable
    - DingTalk: set 'DATA_JUICER_DINGTALK_TOKEN' and
      'DATA_JUICER_DINGTALK_SECRET' environment variables

    For even more secure email authentication, you can use TLS client
    certificates instead of passwords:

    1. Generate a client certificate and key (example using OpenSSL):
       ```bash
       # Generate a private key
       openssl genrsa -out client.key 2048

       # Generate a certificate signing request (CSR)
       openssl req -new -key client.key -out client.csr

       # Generate a self-signed certificate
       openssl x509 -req -days 365 -in client.csr -signkey client.key
            -out client.crt
       ```

    2. Configure your SMTP server to accept this client certificate for
        authentication

    3. Configure Data Juicer to use certificate authentication:
       ```yaml
       notification:
         enabled: true
         email:
           use_cert_auth: true
           client_cert_file: "/path/to/client.crt"
           client_key_file: "/path/to/client.key"
           smtp_server: "smtp.example.com"
           smtp_port: 587
           sender_email: "notifications@example.com"
           recipients: ["recipient@example.com"]
       ```

    4. Or use environment variables:
       ```bash
       export DATA_JUICER_EMAIL_CERT="/path/to/client.crt"
       export DATA_JUICER_EMAIL_KEY="/path/to/client.key"
       ```

    For maximum connection security, you can use a direct SSL connection
    instead of STARTTLS by enabling the 'use_ssl' option:

    ```yaml
    notification:
      enabled: true
      email:
        use_ssl: true
        smtp_port: 465  # Common port for SMTP over SSL
        # ... other email configuration ...
    ```

    This establishes an encrypted connection from the beginning, rather than
     starting with an unencrypted connection and upgrading to TLS as with
     STARTTLS. Note that this option can be combined with certificate
     authentication for maximum security.

    The email notification system supports various email server configurations
      through a flexible configuration system. Here are some examples for
      different servers:

    Standard SMTP with STARTTLS:
    ```yaml
    notification:
      enabled: true
      email:
        smtp_server: "smtp.example.com"
        smtp_port: 587
        username: "your.username@example.com"
        sender_email: "your.username@example.com"
        sender_name: "Your Name"  # Optional
        recipients: ["recipient1@example.com", "recipient2@example.com"]
    ```

    Direct SSL Connection (e.g., Gmail):
    ```yaml
    notification:
      enabled: true
      email:
        smtp_server: "smtp.gmail.com"
        smtp_port: 465
        use_ssl: true
        username: "your.username@gmail.com"
        sender_email: "your.username@gmail.com"
        sender_name: "Your Name"
        recipients: ["recipient1@example.com", "recipient2@example.com"]
    ```

    Alibaba Email Server:
    ```yaml
    notification:
      enabled: true
      email:
        smtp_server: "smtp.alibaba-inc.com"
        smtp_port: 465
        username: "your.username@alibaba-inc.com"
        sender_email: "your.username@alibaba-inc.com"
        sender_name: "Your Name"
        recipient_separator: ";"       # Use semicolons to separate recipients
        recipients: ["recipient1@example.com", "recipient2@example.com"]
    ```

    Environment variable usage examples:
    ```bash
    # General email password
    export DATA_JUICER_EMAIL_PASSWORD="your_email_password"

    # Server-specific passwords (preferred for clarity)
    export DATA_JUICER_SMTP_GMAIL_COM_PASSWORD="your_gmail_password"
    export DATA_JUICER_SMTP_ALIBABA_INC_COM_PASSWORD="your_alibaba_password"

    # Slack webhook
    export DATA_JUICER_SLACK_WEBHOOK="your_slack_webhook_url"

    # DingTalk credentials
    export DATA_JUICER_DINGTALK_TOKEN="your_dingtalk_token"
    export DATA_JUICER_DINGTALK_SECRET="your_dingtalk_secret"
    ```

    If environment variables are not set, the system will fall back to using
    values from the configuration file, but this is less secure and not
    recommended for production environments.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Get notification configuration directly from kwargs
        self.notification_config = kwargs.get("notification_config", {})

        # Initialize notification handlers if configured
        self.notification_handlers = {
            "email": self._send_email_notification,
            "slack": self._send_slack_notification,
            "dingtalk": self._send_dingtalk_notification,
        }

        # Check if notifications are enabled for this operator
        if self.notification_config.get("enabled", False):
            logger.info(f"Notification is enabled for {self.__class__.__name__}")
        else:
            logger.debug(f"Notification is disabled for {self.__class__.__name__}")

    def send_notification(self, message: str, notification_type: str = None, **kwargs):
        """Send a notification message.

        Args:
            message: The message to send
            notification_type: The type of notification to send.
                                Email, Slack, DingTalk.
                                If None, send nothing
            **kwargs: Additional arguments to pass to the notification handler
                      These can override any configuration settings for this
                      specific notification

        Returns:
            bool: True if the notification was sent successfully, else False
        """
        # Check if notifications are enabled
        if not hasattr(self, "notification_config") or not self.notification_config.get("enabled", False):
            # Notifications are disabled, log and return
            logger.debug(f"Not sending notification: disabled for " f"{self.__class__.__name__}")
            return True

        # Check if notification handlers are initialized
        if not hasattr(self, "notification_handlers"):
            logger.error("Notification handlers not initialized")
            return False

        # Apply any temporary overrides for this specific notification
        temp_config = self.notification_config.copy() if hasattr(self, "notification_config") else {}

        # Process the specific notification overrides from kwargs
        for key, value in kwargs.items():
            if key in temp_config and isinstance(temp_config[key], dict) and isinstance(value, dict):
                # For dict values (like email settings), merge them
                temp_config[key].update(value)
            else:
                # For other values, just replace them
                temp_config[key] = value

        # Store the original config and set the temporary one
        original_config = self.notification_config
        self.notification_config = temp_config

        try:
            if notification_type is None:
                logger.info("No notification type specified, ignoring... ")
            elif notification_type in self.notification_handlers:
                # Check if this specific channel is enabled
                channel_config = self.notification_config.get(notification_type, {})
                if isinstance(channel_config, dict) and not channel_config.get("enabled", True):
                    logger.debug(f"Not sending {notification_type} notification: " f"channel disabled")
                    return True
                return self.notification_handlers[notification_type](message, **kwargs)
            else:
                logger.error(f"Unsupported notification type: {notification_type}")
                return False
        finally:
            # Restore the original configuration
            self.notification_config = original_config

    def _send_email_notification(self, message: str, **kwargs):
        """Send an email notification.

        Args:
            message: The message to send
            **kwargs: Additional parameters for email configuration
                      (recipients, subject, etc.)

        Returns:
            bool: Whether the email was sent successfully
        """
        try:
            import os
            import smtplib
            import ssl
            from email.mime.text import MIMEText

            config = self.notification_config.get("email", {})

            # Get email configuration settings with priority order:
            # 1. kwargs (passed directly to the method)
            # 2. config (from notification_config)
            # 3. default values (Alibaba-specific defaults)

            # SMTP server configuration
            smtp_server = kwargs.get("smtp_server", config.get("smtp_server"))
            smtp_port = kwargs.get("smtp_port", config.get("smtp_port", 465))  # Default to 465 (SSL)
            use_ssl = kwargs.get("use_ssl", config.get("use_ssl", True))  # Default to SSL

            # Some more defaults
            include_port_in_address = kwargs.get("include_port_in_address", config.get("include_port_in_address", True))
            recipient_separator = kwargs.get(
                "recipient_separator", config.get("recipient_separator", ";")
            )  # Default to semicolon
            message_encoding = kwargs.get("message_encoding", config.get("message_encoding", "utf-8"))

            # Authentication method
            use_cert_auth = kwargs.get("use_cert_auth", config.get("use_cert_auth", False))

            # Sender information
            sender_email = kwargs.get("sender_email", config.get("sender_email"))
            sender_name = kwargs.get("sender_name", config.get("sender_name", ""))

            # Format sender with name if provided
            formatted_sender = sender_email
            if sender_name and sender_name != "" and "<" not in formatted_sender:
                formatted_sender = f"{sender_name}<{sender_email}>"

            # Authentication credentials
            username = kwargs.get("username", config.get("username", sender_email))

            # Recipients and subject
            recipients = kwargs.get("recipients", config.get("recipients", []))
            subject = kwargs.get("subject", config.get("subject", "Notification from Data Juicer"))

            # Try to get password from environment var first (most secure)
            # Check for service-specific env vars first, then fall back to
            # generic one
            password = None
            env_password_keys = [
                f"DATA_JUICER_{smtp_server.upper().replace('.', '_')}_PASSWORD",  # noqa: E501
                "DATA_JUICER_EMAIL_PASSWORD",
            ]

            for env_key in env_password_keys:
                if os.environ.get(env_key):
                    password = os.environ.get(env_key)
                    break

            # Fall back to config if no environment variable is set
            if not password:
                logger.warning(
                    "Email password environment variables not set. "
                    f'Tried: {", ".join(env_password_keys)}. '
                    "Falling back to configuration. Consider using "
                    "environmentvariables for better security."
                )
                password = kwargs.get("password", config.get("password"))

            # Certificate authentication settings
            client_cert_file = os.environ.get("DATA_JUICER_EMAIL_CERT")
            client_key_file = os.environ.get("DATA_JUICER_EMAIL_KEY")

            # Fall back to config if environment variables not set
            if not client_cert_file:
                client_cert_file = kwargs.get("client_cert_file", config.get("client_cert_file"))
            if not client_key_file:
                client_key_file = kwargs.get("client_key_file", config.get("client_key_file"))

            # Validate required parameters
            if not smtp_server or not recipients:
                logger.error("Missing required email configuration (server/recipients)")
                return False

            if not use_cert_auth and not (username and password):
                logger.error("Missing credentials for password-based authentication")
                return False

            if use_cert_auth and (not client_cert_file or not client_key_file):
                logger.error("Missing client certificate or key for authentication")
                return False

            # Create SSL context for certificate authentication if needed
            context = None
            if use_cert_auth and client_cert_file and client_key_file:
                logger.info("Using TLS client certificate authentication for email")
                context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
                context.load_cert_chain(certfile=client_cert_file, keyfile=client_key_file)

            # Create email message
            msg = MIMEText(message, "plain", message_encoding)
            msg["From"] = formatted_sender
            msg["To"] = recipient_separator.join(recipients)
            msg["Subject"] = subject

            # Determine connection type and send message
            # Port 465 is typically for SMTP over SSL
            if use_ssl or smtp_port == 465:
                logger.info(f"Using direct SSL connection to {smtp_server}:{smtp_port}")

                # Handle different connection string formats
                server_address = smtp_server
                if include_port_in_address:
                    server_address = f"{smtp_server}:{smtp_port}"

                # Use the SSL context if certificate authentication is enabled
                if context:
                    with smtplib.SMTP_SSL(
                        server_address, smtp_port if not include_port_in_address else None, context=context
                    ) as server:
                        # No login needed with client certificates
                        server.send_message(msg)
                else:
                    # Standard SSL connection with password
                    with smtplib.SMTP_SSL(server_address, smtp_port if not include_port_in_address else None) as server:
                        server.login(username, password)
                        server.sendmail(formatted_sender, recipients, msg.as_string())
            else:
                # Standard connection with STARTTLS upgrade
                logger.info(f"Using STARTTLS connection to {smtp_server}:{smtp_port}")

                if use_cert_auth and client_cert_file and client_key_file:
                    # Connect with certificate authentication
                    with smtplib.SMTP(smtp_server, smtp_port) as server:
                        server.starttls(context=context)
                        # No login needed with client certificates
                        server.send_message(msg)
                else:
                    # Connect with password authentication
                    with smtplib.SMTP(smtp_server, smtp_port) as server:
                        server.starttls()
                        server.login(username, password)
                        server.send_message(msg)

            return True
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False

    def _send_slack_notification(self, message: str, **kwargs):
        """Send a Slack notification.

        Args:
            message: The message to send
            **kwargs: Additional parameters for Slack configuration
                      (webhook_url, channel, etc.)

        Returns:
            bool: Whether the notification was sent successfully
        """
        try:
            import json

            import requests

            config = self.notification_config.get("slack", {})

            # Override config with kwargs if provided
            webhook_url = kwargs.get("webhook_url", config.get("webhook_url"))
            channel = kwargs.get("channel", config.get("channel"))
            username = kwargs.get("username", config.get("username", "Data Juicer"))

            if not webhook_url:
                logger.error("Missing required Slack webhook URL")
                return False

            # Prepare payload
            payload = {
                "text": message,
                "username": username,
            }

            if channel:
                payload["channel"] = channel

            # Send notification
            response = requests.post(
                webhook_url, data=json.dumps(payload), headers={"Content-Type": "application/json"}
            )

            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False

    def _send_dingtalk_notification(self, message: str, **kwargs):
        """Send a DingTalk notification.

        Args:
            message: The message to send
            **kwargs: Additional parameters for DingTalk configuration
                      (access_token, secret, etc.)

        Returns:
            bool: Whether the notification was sent successfully
        """
        try:
            import base64
            import hashlib
            import hmac
            import json
            import time
            import urllib.parse

            import requests

            config = self.notification_config.get("dingtalk", {})

            # Override config with kwargs if provided
            access_token = kwargs.get("access_token", config.get("access_token"))
            secret = kwargs.get("secret", config.get("secret"))

            if not access_token:
                import logging

                logging.error("Missing required DingTalk access token")
                return False

            # Prepare URL with signature if secret is provided
            url = f"https://oapi.dingtalk.com/robot/send?access_token={access_token}"  # noqa: E501

            if secret:
                timestamp = str(round(time.time() * 1000))
                string_to_sign = f"{timestamp}\n{secret}"
                hmac_code = hmac.new(secret.encode(), string_to_sign.encode(), digestmod=hashlib.sha256).digest()
                sign = urllib.parse.quote_plus(base64.b64encode(hmac_code).decode())
                url = f"{url}&timestamp={timestamp}&sign={sign}"

            # Prepare payload
            payload = {"msgtype": "text", "text": {"content": message}}

            # Send notification
            response = requests.post(url, data=json.dumps(payload), headers={"Content-Type": "application/json"})

            result = response.json()
            return result.get("errcode") == 0
        except Exception as e:
            logger.error(f"Failed to send DingTalk notification: {e}")
            return False
