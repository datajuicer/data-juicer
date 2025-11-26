## Notification System

Data Juicer supports notifications for operators that inherit from the `NotificationMixin` class. Each operator defines its own notification configuration, which can include email, Slack, and DingTalk notifications.

### Basic Configuration

To enable notifications for an operator:

```yaml
- human_preference_annotation_mapper:
    # Enable notifications for this operator
    notification_config:
      enabled: true  # Master switch for notifications
      
      # Email channel configuration
      email:
        enabled: true  # Enable email notifications
        smtp_server: "smtp.example.com"
        smtp_port: 587
        username: "your-email@example.com"
        sender_email: "your-email@example.com"
        recipients: ["recipient1@example.com", "recipient2@example.com"]
        subject: "Notification from Data Juicer"
      
      # Slack channel configuration (optional)
      slack:
        enabled: false  # Disable Slack notifications
```

### Channel-Specific Settings

Each notification channel (email, Slack, DingTalk) can be independently enabled or disabled:

```yaml
notification_config:
  enabled: true  # Master switch for notifications
  
  email:
    enabled: true  # Enable email notifications
    # Email settings...
  
  slack:
    enabled: false  # Disable Slack notifications
    # Slack settings (won't be used since disabled)
  
  dingtalk:
    enabled: true  # Enable DingTalk notifications
    # DingTalk settings...
```

### Email Configuration

Email notifications support various server configurations:

#### Standard SMTP with STARTTLS:
```yaml
email:
  enabled: true
  smtp_server: "smtp.example.com"
  smtp_port: 587
  username: "your.username@example.com"
  sender_email: "your.username@example.com"
  sender_name: "Your Name"  # Optional
  recipients: ["recipient1@example.com", "recipient2@example.com"]
```

#### Direct SSL Connection (e.g., Gmail):
```yaml
email:
  enabled: true
  smtp_server: "smtp.gmail.com"
  smtp_port: 465
  use_ssl: true
  username: "your.username@gmail.com"
  sender_email: "your.username@gmail.com"
  recipients: ["recipient1@example.com"]
```

#### Alibaba Email Server:
```yaml
email:
  enabled: true
  smtp_server: "smtp.alibaba-inc.com"
  smtp_port: 465
  use_ssl: true
  include_port_in_address: true  # Include port in server address
  use_mime_multipart: false      # Use simple MIMEText
  username: "your.username@alibaba-inc.com"
  sender_email: "your.username@alibaba-inc.com"
  recipient_separator: ";"       # Use semicolons for recipients
  add_message_id: true           # Add Message-ID header
  recipients: ["recipient1@example.com", "recipient2@example.com"]
```

### Secure Password Handling

For security, passwords should be provided via environment variables:

```bash
# General email password
export DATA_JUICER_EMAIL_PASSWORD="your_password"

# Server-specific passwords (preferred)
export DATA_JUICER_SMTP_GMAIL_COM_PASSWORD="your_gmail_password"
export DATA_JUICER_SMTP_ALIBABA_INC_COM_PASSWORD="your_alibaba_password"

# Slack webhook
export DATA_JUICER_SLACK_WEBHOOK="your_slack_webhook_url"
```

### Certificate Authentication

For enhanced security, you can use TLS client certificates:

```yaml
email:
  enabled: true
  use_cert_auth: true
  client_cert_file: "/path/to/client.crt"
  client_key_file: "/path/to/client.key"
  smtp_server: "smtp.example.com"
  smtp_port: 587
  sender_email: "notifications@example.com"
  recipients: ["recipient@example.com"]
```

### Examples

See these example configuration files:
- [`annotation_with_notifications.yaml`](https://github.com/datajuicer/data-juicer-hub/blob/main/annotation_config/annotation_with_notifications.yaml): Shows how to use notifications with annotation operations.
- [`notification_alibaba_email.yaml`](notification_alibaba_email.yaml): Demonstrates Alibaba email server configuration. 