#!/usr/bin/env python3
"""
Script to fetch GitHub Traffic data (Referring sites) and send email reports.

This script uses GitHub API to fetch traffic data for the repository,
specifically the referring sites information. The data is then formatted
and sent via email to specified recipients.

Environment Variables Required:
    - GITHUB_TOKEN: GitHub personal access token with repo access
    - SMTP_SERVER: SMTP server address (e.g., smtp.gmail.com)
    - SMTP_PORT: SMTP server port (default: 465 for SSL)
    - SMTP_USERNAME: Email username for authentication
    - SMTP_PASSWORD: Email password for authentication
    - SENDER_EMAIL: Email address to send from
    - RECIPIENT_EMAILS: Comma-separated list of recipient emails
    - GITHUB_REPOSITORY: Repository name (format: owner/repo)
"""

import os
import smtplib
import ssl
import sys
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict

try:
    import requests
except ImportError:
    print("requests library not found. Installing...")
    os.system(f"{sys.executable} -m pip install requests")
    import requests


def fetch_traffic_data(repo: str, token: str) -> Dict[str, Any]:
    """
    Fetch traffic data from GitHub API.

    Args:
        repo: Repository name in format 'owner/repo'
        token: GitHub personal access token

    Returns:
        Dictionary containing traffic data
    """
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}

    base_url = f"https://api.github.com/repos/{repo}/traffic"

    traffic_data = {}

    try:
        # Fetch referring sites
        print(f"Fetching referring sites for {repo}...")
        response = requests.get(f"{base_url}/popular/referrers", headers=headers)
        response.raise_for_status()
        traffic_data["referrers"] = response.json()

        # Fetch popular paths
        print(f"Fetching popular paths for {repo}...")
        response = requests.get(f"{base_url}/popular/paths", headers=headers)
        response.raise_for_status()
        traffic_data["paths"] = response.json()

        # Fetch views
        print(f"Fetching views for {repo}...")
        response = requests.get(f"{base_url}/views", headers=headers)
        response.raise_for_status()
        traffic_data["views"] = response.json()

        # Fetch clones
        print(f"Fetching clones for {repo}...")
        response = requests.get(f"{base_url}/clones", headers=headers)
        response.raise_for_status()
        traffic_data["clones"] = response.json()

        print("Successfully fetched all traffic data")
        return traffic_data

    except requests.exceptions.RequestException as e:
        print(f"Error fetching traffic data: {e}")
        raise


def format_traffic_report(repo: str, traffic_data: Dict[str, Any]) -> str:
    """
    Format traffic data into a readable HTML email report.

    Args:
        repo: Repository name
        traffic_data: Dictionary containing traffic data

    Returns:
        HTML formatted report string
    """
    report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
            }}
            h1 {{ color: #0366d6; }}
            h2 {{
                color: #0366d6;
                border-bottom: 2px solid #0366d6;
                padding-bottom: 5px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 20px;
            }}
            th, td {{
                padding: 10px;
                text-align: left;
                border: 1px solid #ddd;
            }}
            th {{ background-color: #0366d6; color: white; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .summary {{
                background-color: #f0f8ff;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
            .no-data {{ color: #999; font-style: italic; }}
        </style>
    </head>
    <body>
        <h1>GitHub Traffic Report for {repo}</h1>
        <p><strong>Report Generated:</strong> {report_date}</p>

        <div class="summary">
            <h2>📊 Summary</h2>
            <p><strong>Total Views (last 14 days):</strong>
               {traffic_data.get('views', {}).get('count', 0)}
               (Unique: {traffic_data.get('views', {}).get('uniques', 0)})
            </p>
            <p><strong>Total Clones (last 14 days):</strong>
               {traffic_data.get('clones', {}).get('count', 0)}
               (Unique: {traffic_data.get('clones', {}).get('uniques', 0)})
            </p>
        </div>

        <h2>🔗 Referring Sites</h2>
    """

    # Referring sites table
    referrers = traffic_data.get("referrers", [])
    if referrers:
        html += """
        <table>
            <tr>
                <th>Referrer</th>
                <th>Total Views</th>
                <th>Unique Visitors</th>
            </tr>
        """
        for ref in referrers:
            html += f"""
            <tr>
                <td>{ref.get('referrer', 'N/A')}</td>
                <td>{ref.get('count', 0)}</td>
                <td>{ref.get('uniques', 0)}</td>
            </tr>
            """
        html += "</table>"
    else:
        html += '<p class="no-data">No referring sites data available</p>'

    # Popular paths
    html += "<h2>📄 Popular Paths</h2>"
    paths = traffic_data.get("paths", [])
    if paths:
        html += """
        <table>
            <tr>
                <th>Path</th>
                <th>Title</th>
                <th>Total Views</th>
                <th>Unique Visitors</th>
            </tr>
        """
        for path in paths[:10]:  # Top 10 paths
            html += f"""
            <tr>
                <td>
                    <a href="https://github.com/{repo}{path.get('path', '')}">
                        {path.get('path', 'N/A')}
                    </a>
                </td>
                <td>{path.get('title', 'N/A')}</td>
                <td>{path.get('count', 0)}</td>
                <td>{path.get('uniques', 0)}</td>
            </tr>
            """
        html += "</table>"
    else:
        html += '<p class="no-data">No popular paths data available</p>'

    # Views timeline
    html += "<h2>👁️ Views Timeline (Last 14 Days)</h2>"
    views_timeline = traffic_data.get("views", {}).get("views", [])
    if views_timeline:
        html += """
        <table>
            <tr>
                <th>Date</th>
                <th>Views</th>
                <th>Unique Visitors</th>
            </tr>
        """
        for view in views_timeline:
            date = view.get("timestamp", "").split("T")[0]
            html += f"""
            <tr>
                <td>{date}</td>
                <td>{view.get('count', 0)}</td>
                <td>{view.get('uniques', 0)}</td>
            </tr>
            """
        html += "</table>"
    else:
        html += '<p class="no-data">No views timeline data available</p>'

    # Clones timeline
    html += "<h2>📥 Clones Timeline (Last 14 Days)</h2>"
    clones_timeline = traffic_data.get("clones", {}).get("clones", [])
    if clones_timeline:
        html += """
        <table>
            <tr>
                <th>Date</th>
                <th>Clones</th>
                <th>Unique Cloners</th>
            </tr>
        """
        for clone in clones_timeline:
            date = clone.get("timestamp", "").split("T")[0]
            html += f"""
            <tr>
                <td>{date}</td>
                <td>{clone.get('count', 0)}</td>
                <td>{clone.get('uniques', 0)}</td>
            </tr>
            """
        html += "</table>"
    else:
        html += '<p class="no-data">No clones timeline data available</p>'

    html += """
    </body>
    </html>
    """

    return html


def send_email_report(subject: str, html_content: str, smtp_config: Dict[str, Any]) -> bool:
    """
    Send email report using SMTP.

    Args:
        subject: Email subject
        html_content: HTML content of the email
        smtp_config: Dictionary containing SMTP configuration

    Returns:
        True if email sent successfully, False otherwise
    """
    try:
        # Create message
        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = smtp_config["sender_email"]
        message["To"] = ", ".join(smtp_config["recipients"])

        # Attach HTML content
        html_part = MIMEText(html_content, "html")
        message.attach(html_part)

        # Connect and send
        smtp_port = int(smtp_config.get("smtp_port", 465))
        use_ssl = smtp_config.get("use_ssl", True)

        if use_ssl or smtp_port == 465:
            server_address = smtp_config["smtp_server"]
            print(f"Connecting to {server_address}:{smtp_port} using SSL...")
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(server_address, smtp_port, context=context) as server:
                server.login(smtp_config["username"], smtp_config["password"])
                server.send_message(message)
        else:
            server_address = smtp_config["smtp_server"]
            msg = f"Connecting to {server_address}:{smtp_port} using STARTTLS"
            print(f"{msg}...")
            with smtplib.SMTP(server_address, smtp_port) as server:
                server.starttls()
                server.login(smtp_config["username"], smtp_config["password"])
                server.send_message(message)

        print("Email sent successfully!")
        return True

    except Exception as e:
        print(f"Error sending email: {e}")
        return False


def main():
    """Main function to orchestrate traffic data fetching and email sending."""

    # Get configuration from environment variables
    github_token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")
    smtp_server = os.environ.get("SMTP_SERVER")
    smtp_port = os.environ.get("SMTP_PORT", "465")
    smtp_username = os.environ.get("SMTP_USERNAME")
    smtp_password = os.environ.get("SMTP_PASSWORD")
    sender_email = os.environ.get("SENDER_EMAIL")
    recipient_emails = os.environ.get("RECIPIENT_EMAILS", "")

    # Validate required environment variables
    required_vars = {
        "GITHUB_TOKEN": github_token,
        "GITHUB_REPOSITORY": repo,
        "SMTP_SERVER": smtp_server,
        "SMTP_USERNAME": smtp_username,
        "SMTP_PASSWORD": smtp_password,
        "SENDER_EMAIL": sender_email,
        "RECIPIENT_EMAILS": recipient_emails,
    }

    missing_vars = [var for var, value in required_vars.items() if not value]
    if missing_vars:
        missing_list = ", ".join(missing_vars)
        print(f"Error: Missing required environment variables: {missing_list}")
        sys.exit(1)

    # Parse recipient emails
    recipients = [email.strip() for email in recipient_emails.split(",") if email.strip()]
    if not recipients:
        print("Error: No valid recipient emails provided")
        sys.exit(1)

    try:
        # Fetch traffic data
        print(f"\n{'='*60}")
        print("Fetching GitHub Traffic Data")
        print(f"{'='*60}\n")

        traffic_data = fetch_traffic_data(repo, github_token)

        # Format report
        print(f"\n{'='*60}")
        print("Formatting Report")
        print(f"{'='*60}\n")

        html_report = format_traffic_report(repo, traffic_data)

        # Send email
        print(f"\n{'='*60}")
        print("Sending Email Report")
        print(f"{'='*60}\n")

        smtp_config = {
            "smtp_server": smtp_server,
            "smtp_port": smtp_port,
            "username": smtp_username,
            "password": smtp_password,
            "sender_email": sender_email,
            "recipients": recipients,
            "use_ssl": True,
        }

        report_date = datetime.now().strftime("%Y-%m-%d")
        subject = f"GitHub Traffic Report - {repo} - {report_date}"

        success = send_email_report(subject, html_report, smtp_config)

        if success:
            print(f"\n{'='*60}")
            print("✅ Report sent successfully!")
            print(f"{'='*60}\n")
            sys.exit(0)
        else:
            print(f"\n{'='*60}")
            print("❌ Failed to send report")
            print(f"{'='*60}\n")
            sys.exit(1)

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"❌ Error: {e}")
        print(f"{'='*60}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
