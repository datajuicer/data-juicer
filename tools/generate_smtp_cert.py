#!/usr/bin/env python
"""
Generate SMTP client certificates for secure email authentication.
This script creates a client certificate and key that can be used for
TLS client certificate authentication with SMTP servers.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from loguru import logger


def run_command(command, check=True):
    """Run a shell command and return its output"""
    logger.info(f"Running: {' '.join(command)}")
    result = subprocess.run(command, check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        logger.error(f"Command failed with error: {result.stderr}")
        return None
    return result.stdout


def check_openssl():
    """Check if OpenSSL is installed"""
    try:
        result = run_command(["openssl", "version"])
        logger.info(f"OpenSSL found: {result.strip()}")
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.error("OpenSSL not found. Please install OpenSSL to continue.")
        return False


def generate_certificates(output_dir, common_name, days=365, key_size=2048):
    """Generate client certificate and key for SMTP authentication"""
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    key_file = output_path / "smtp_client.key"
    csr_file = output_path / "smtp_client.csr"
    cert_file = output_path / "smtp_client.crt"

    # Generate private key
    logger.info(f"Generating private key ({key_size} bits)...")
    if not run_command(["openssl", "genrsa", "-out", str(key_file), str(key_size)]):
        return None, None

    # Generate CSR (Certificate Signing Request)
    logger.info("Generating certificate signing request...")
    csr_cmd = ["openssl", "req", "-new", "-key", str(key_file), "-out", str(csr_file), "-subj", f"/CN={common_name}"]
    if not run_command(csr_cmd):
        return None, None

    # Generate self-signed certificate
    logger.info(f"Generating self-signed certificate (valid for {days} days)...")
    cert_cmd = [
        "openssl",
        "x509",
        "-req",
        "-days",
        str(days),
        "-in",
        str(csr_file),
        "-signkey",
        str(key_file),
        "-out",
        str(cert_file),
    ]
    if not run_command(cert_cmd):
        return None, None

    # Clean up CSR file as it's no longer needed
    os.unlink(csr_file)

    logger.info(f"Certificate and key generated successfully in {output_path}")
    return cert_file, key_file


def print_config_example(cert_file, key_file):
    """Print example configuration for using the generated certificates"""
    print("\n" + "=" * 80)
    print("Certificate Generation Complete!")
    print("=" * 80)

    print("\nTo use these certificates with Data Juicer, you can:")

    print("\n1. Set environment variables:")
    print(f'   export DATA_JUICER_EMAIL_CERT="{cert_file}"')
    print(f'   export DATA_JUICER_EMAIL_KEY="{key_file}"')

    print("\n2. Or update your configuration file:")
    print("   ```yaml")
    print("   notification_config:")
    print("     enabled: true")
    print("     email:")
    print("       use_cert_auth: true")
    print(f'       client_cert_file: "{cert_file}"')
    print(f'       client_key_file: "{key_file}"')
    print('       smtp_server: "smtp.example.com"')
    print("       smtp_port: 587")
    print('       sender_email: "notifications@example.com"')
    print('       recipients: ["recipient@example.com"]')
    print("   ```")

    print("\n   For maximum security, you can also use a direct SSL connection:")
    print("   ```yaml")
    print("   notification_config:")
    print("     enabled: true")
    print("     email:")
    print("       use_cert_auth: true")
    print(f'       client_cert_file: "{cert_file}"')
    print(f'       client_key_file: "{key_file}"')
    print("       use_ssl: true")
    print('       smtp_server: "smtp.example.com"')
    print("       smtp_port: 465  # Common port for SMTP over SSL")
    print('       sender_email: "notifications@example.com"')
    print('       recipients: ["recipient@example.com"]')
    print("   ```")

    print("\n3. Configure your SMTP server to accept this client " "certificate for authentication.")
    print("   This step varies depending on your email provider or SMTP " "server software.")
    print("   You may need to contact your email administrator or provider " "for assistance.")

    print(
        "\nIMPORTANT: Keep your private key file secure! Anyone with access "
        "to this file can send emails using your identity."
    )
    print("=" * 80 + "\n")


def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(description="Generate SMTP client certificates for secure email " "authentication")
    parser.add_argument(
        "--output",
        "-o",
        default=os.path.expanduser("~/.data_juicer/certs"),
        help="Output directory for certificates " "(default: ~/.data_juicer/certs)",
    )
    default_cn = f"data-juicer-{os.getenv('USER', 'user')}@{os.uname().nodename}"  # noqa: E501
    parser.add_argument(
        "--cn",
        "--common-name",
        default=default_cn,
        help="Common Name (CN) for the certificate " "(default: data-juicer-<username>@<hostname>)",
    )
    parser.add_argument("--days", "-d", type=int, default=365, help="Validity period in days (default: 365)")
    parser.add_argument("--key-size", "-k", type=int, default=2048, help="Key size in bits (default: 2048)")

    args = parser.parse_args()

    # Check if OpenSSL is installed
    if not check_openssl():
        sys.exit(1)

    # Generate certificates
    cert_file, key_file = generate_certificates(args.output, args.cn, args.days, args.key_size)

    if cert_file and key_file:
        # Print configuration example
        print_config_example(cert_file, key_file)
        return 0
    else:
        logger.error("Failed to generate certificates")
        return 1


if __name__ == "__main__":
    sys.exit(main())
