#!/usr/bin/env python
# flake8: noqa: E501
"""
Utility script to manage Label Studio using Docker/pip for Data Juicer.
It's intended for localhost development/debug purposes.

This script:
1. Checks if Docker is installed and running
2. Pulls the Label Studio Docker image if needed
3. Starts a Label Studio container with proper configuration (using Docker volumes by default)
4. Creates a test project with sample configuration if requested
5. Outputs connection details for use in Data Juicer

Usage:
    python label_studio_service.py [--port PORT] [--create-test-project]
    python label_studio_service.py --status
    python label_studio_service.py --stop [--remove-volumes]
    python label_studio_service.py --kill [--remove-volumes]
    python label_studio_service.py --use-pip  # Use local pip instead of Docker

Options:
    --port PORT                  Port to run Label Studio on (default: 7070)
    --data-dir PATH              Directory to store Label Studio data (only used with --use-host-mount)
    --create-test-project        Create a test project with sample configuration
    --username USERNAME          Admin username for Label Studio (default: admin@example.com)
    --password PASSWORD          Admin password for Label Studio (default: admin)
    --api-token API_TOKEN        Manually specify the API token (skip automatic retrieval)
    --image IMAGE                Label Studio Docker image (default: heartexlabs/label-studio:latest)
    --container-name NAME        Name for Docker container (default: data-juicer-label-studio)
    --network NETWORK            Docker network to connect to (default: None)
    --stop                       Stop and remove the Label Studio container
    --kill                       Force kill and remove the Label Studio container
    --status                     Check if Label Studio container is running
    --use-host-mount             Use host directory mounting instead of Docker volumes (not recommended on macOS)
    --remove-volumes             Remove Docker volumes when stopping the container
    --use-pip                    Use local pip installation instead of Docker (uses a virtual environment)

Notes:
    - By default, the script uses Docker with named volumes which provides better compatibility,
      especially on macOS where file system permission issues are common with bind mounts.
    - If you need to access the data files directly, use --use-host-mount option, but be aware
      that this may cause permission issues on macOS.
    - If you prefer to run Label Studio without Docker, use the --use-pip option.
    - Use --remove-volumes with --stop to remove the Docker volume when stopping the container.
"""

import argparse
import atexit
import fcntl
import json
import logging
import os
import select
import subprocess
import sys
import time
from logging.handlers import RotatingFileHandler

import requests

# Global logger
logger = None


def setup_logging(data_dir):
    """Setup logging to both file and console"""
    # Create logs directory
    logs_dir = os.path.join(os.path.abspath(data_dir), "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Create log file path
    log_file = os.path.join(logs_dir, "label_studio_service.log")

    # Create logger
    logger = logging.getLogger("label_studio_service")
    logger.setLevel(logging.INFO)

    # Clear any existing handlers
    if logger.handlers:
        logger.handlers.clear()

    # Create file handler with rotation (10MB max size, keep 3 backup files)
    file_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=3)
    file_handler.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Start a Label Studio Docker container")
    parser.add_argument("--port", type=int, default=7070, help="Port to run Label Studio on (default: 7070)")
    parser.add_argument(
        "--data-dir",
        default="./label_studio_data",
        help="Directory to store Label Studio data (only used with --use-host-mount)",
    )
    parser.add_argument("--username", help="Username for Label Studio (default: will prompt if not provided)")
    parser.add_argument("--password", help="Password for Label Studio (default: will prompt if not provided)")
    parser.add_argument("--api-token", help="Specify a fixed API token for the admin account")
    parser.add_argument(
        "--image",
        default="heartexlabs/label-studio:latest",
        help="Docker image to use (default: heartexlabs/label-studio:latest)",
    )
    parser.add_argument("--container-name", default="data-juicer-label-studio", help="Name for the Docker container")
    parser.add_argument("--network", help="Docker network to connect to")
    parser.add_argument("--status", action="store_true", help="Check if Label Studio container is running")
    parser.add_argument("--stop", action="store_true", help="Stop the Label Studio container")
    parser.add_argument(
        "--remove-volumes", action="store_true", help="Remove Docker volumes when stopping the container"
    )
    parser.add_argument("--kill", action="store_true", help="Force kill the Label Studio container")
    parser.add_argument("--create-test-project", action="store_true", help="Create a test project in Label Studio")
    parser.add_argument(
        "--use-host-mount",
        action="store_true",
        help="Use host directory mounting instead of Docker volumes (not recommended on macOS)",
    )
    parser.add_argument(
        "--use-pip",
        action="store_true",
        help="Use local pip installation instead of Docker (uses a virtual environment)",
    )
    return parser.parse_args()


def check_docker_installed():
    """Check if Docker is installed and running"""
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            logger.info("Docker is installed and running.")
            return True
        else:
            logger.error("Docker is installed but not running or has permission issues.")
            logger.error(result.stderr)
            return False
    except FileNotFoundError:
        logger.error("Docker is not installed or not in PATH.")
        return False


def pull_docker_image(image):
    """Pull the Label Studio Docker image"""
    logger.info(f"Pulling Docker image: {image}...")
    result = subprocess.run(["docker", "pull", image], check=False)
    return result.returncode == 0


def check_container_exists(container_name):
    """Check if a container with the given name exists"""
    try:
        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        logger.error("Docker is not installed or not in PATH.")
        return False
    return container_name in result.stdout.strip().split("\n")


def check_container_running(container_name):
    """Check if a container is currently running"""
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        logger.error("Docker is not installed or not in PATH.")
        return False
    return container_name in result.stdout.strip().split("\n")


def check_volume_exists(volume_name):
    """Check if a Docker volume exists"""
    try:
        result = subprocess.run(
            ["docker", "volume", "ls", "-q", "-f", f"name={volume_name}"], capture_output=True, text=True, check=False
        )
    except FileNotFoundError:
        logger.error("Docker is not installed or not in PATH.")
        return False
    return volume_name in result.stdout.strip().split("\n")


def remove_volume(volume_name):
    """Remove a Docker volume"""
    if check_volume_exists(volume_name):
        logger.info(f"Removing Docker volume: {volume_name}...")
        subprocess.run(["docker", "volume", "rm", volume_name], check=False)
        logger.info(f"Docker volume {volume_name} removed.")
        return True
    else:
        logger.info(f"Docker volume {volume_name} does not exist.")
        return False


def stop_container(container_name, remove_volumes=False):
    """Stop and remove the Label Studio container"""
    if check_container_exists(container_name):
        logger.info(f"Stopping container: {container_name}...")
        subprocess.run(["docker", "stop", container_name], check=False)
        logger.info(f"Removing container: {container_name}...")
        subprocess.run(["docker", "rm", container_name], check=False)
        logger.info(f"Container {container_name} stopped and removed.")

        # Remove associated volume if requested
        if remove_volumes:
            volume_name = f"{container_name}-data"
            remove_volume(volume_name)

        return True
    else:
        logger.info(f"Container {container_name} does not exist.")
        return False


def kill_container(container_name):
    """Force kill the Label Studio container"""
    if check_container_running(container_name):
        logger.info(f"Force killing container: {container_name}...")
        subprocess.run(["docker", "kill", container_name], check=False)
        logger.info(f"Removing container: {container_name}...")
        subprocess.run(["docker", "rm", "-f", container_name], check=False)
        logger.info(f"Container {container_name} killed and removed.")
        return True
    elif check_container_exists(container_name):
        logger.info(f"Container {container_name} exists but is not running. Removing...")
        subprocess.run(["docker", "rm", "-f", container_name], check=False)
        logger.info(f"Container {container_name} removed.")
        return True
    else:
        logger.info(f"Container {container_name} does not exist.")
        return False


def start_label_studio_container(
    port, data_dir, username, password, image, container_name, network=None, predefined_token=None, use_host_mount=False
):
    """Start a Label Studio Docker container"""
    # If using host mount, ensure data directory exists
    if use_host_mount:
        # Ensure data directory exists with proper permissions
        data_dir = os.path.abspath(data_dir)
        os.makedirs(data_dir, exist_ok=True)

        # Create all required subdirectories for Label Studio
        required_dirs = ["media", "data", "upload", "export", "tmp", "test_data", "upload/avatars"]

        for subdir in required_dirs:
            full_path = os.path.join(data_dir, subdir)
            os.makedirs(full_path, exist_ok=True)
            logger.info(f"Created directory: {full_path}")

        # Create empty files to ensure proper permissions
        touch_files = [os.path.join(data_dir, "label_studio.sqlite3"), os.path.join(data_dir, "logs.txt")]

        for file_path in touch_files:
            try:
                # Create the file if it doesn't exist
                if not os.path.exists(file_path):
                    with open(file_path, "w") as f:
                        pass
                    logger.info(f"Created file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to create file {file_path}: {e}")

        # On Unix-like systems, explicitly set directory permissions
        if os.name != "nt":  # Not Windows
            try:
                # Make sure the data directory and all its subdirectories are writable by everyone
                # This is needed because the container may run as a different user
                logger.info(f"Setting proper permissions for {data_dir}...")
                # Set permissions to 777 (rwxrwxrwx) to ensure the container can write to it
                subprocess.run(["chmod", "-R", "777", data_dir], check=False)
            except Exception as e:
                logger.warning(f"Failed to set permissions for data directory: {e}")
                logger.warning("The container may not be able to write to the data directory")

    # Stop existing container if it exists
    if check_container_exists(container_name):
        stop_container(container_name)

    # Generate a fixed token if not provided
    if not predefined_token:
        import uuid

        predefined_token = str(uuid.uuid4())
        logger.info(f"Generated fixed API token: {predefined_token}")

    # Build docker run command
    cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        container_name,
        # Add restart policy
        "--restart",
        "unless-stopped",
        # Add memory limits
        "--memory",
        "4G",
        # Add more compatible flags for macOS
        "--privileged",
        "-p",
        f"{port}:8080",
    ]

    # Choose volume mounting strategy based on parameters
    if use_host_mount:
        logger.info("Using host directory mount approach (not recommended on macOS)")
        cmd.extend(
            [
                # Use a simple single volume mount
                "-v",
                f"{data_dir}:/label-studio/data:delegated",
            ]
        )
    else:
        logger.info("Using Docker named volume approach (recommended)")

        # Create a named volume for Label Studio data
        volume_name = f"{container_name}-data"
        logger.info(f"Creating/using Docker named volume: {volume_name}")

        # Check if volume exists, if not create it
        check_volume_cmd = ["docker", "volume", "ls", "-q", "-f", f"name={volume_name}"]
        check_result = subprocess.run(check_volume_cmd, capture_output=True, text=True, check=False)

        if volume_name not in check_result.stdout:
            logger.info(f"Creating Docker volume: {volume_name}")
            create_volume_cmd = ["docker", "volume", "create", volume_name]
            create_result = subprocess.run(create_volume_cmd, capture_output=True, text=True, check=False)

            if create_result.returncode != 0:
                logger.error(f"Failed to create Docker volume: {create_result.stderr}")
                logger.info("Falling back to host directory mount")
                cmd.extend(
                    [
                        "-v",
                        f"{data_dir}:/label-studio/data:delegated",
                    ]
                )
            else:
                # Use the named volume for the container
                cmd.extend(
                    [
                        "-v",
                        f"{volume_name}:/label-studio/data",
                    ]
                )
        else:
            # Use the existing named volume for the container
            cmd.extend(
                [
                    "-v",
                    f"{volume_name}:/label-studio/data",
                ]
            )

    # Add common environment variables and options
    cmd.extend(
        [
            # Set user to root for ensuring write permissions inside container
            "--user",
            "root",
            # Add environment variables for database and media settings
            "-e",
            "LABEL_STUDIO_BASE_DATA_DIR=/label-studio/data",
            "-e",
            f"LABEL_STUDIO_USERNAME={username}",
            "-e",
            f"LABEL_STUDIO_PASSWORD={password}",
            "-e",
            "LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK=true",
            "-e",
            f"LABEL_STUDIO_USER_TOKEN={predefined_token}",  # Set fixed token
        ]
    )

    # Add network if specified
    if network:
        cmd.extend(["--network", network])

    # Add image name
    cmd.append(image)

    logger.info(f"Starting Label Studio container on port {port}...")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        logger.error(f"Failed to start container: {result.stderr}")
        # Check if there's a specific error message we can extract
        if "port is already allocated" in result.stderr:
            logger.error(f"Port {port} is already in use. Try a different port.")
        elif "Error response from daemon" in result.stderr:
            # Extract the specific Docker daemon error
            import re

            error_match = re.search(r"Error response from daemon: (.*)", result.stderr)
            if error_match:
                daemon_error = error_match.group(1)
                logger.error(f"Docker daemon error: {daemon_error}")

        # Check if the container was partially created and clean it up
        if check_container_exists(container_name):
            logger.info(f"Cleaning up partially created container: {container_name}")
            stop_container(container_name)

        return None, None

    container_id = result.stdout.strip()
    logger.info(f"Container started with ID: {container_id}")

    # Register cleanup function to stop the container on exit
    atexit.register(lambda: stop_container(container_name))

    # Wait for server to start with better log monitoring
    server_url = f"http://localhost:{port}"
    max_wait_time = 600  # 10 minutes total wait time
    start_time = time.time()
    check_interval = 5  # Check every 5 seconds

    logger.info("Waiting for Label Studio to initialize (this may take several minutes on first run)...")
    logger.info("Monitoring container logs for startup progress...")

    # Initialization stages to look for in logs
    init_stages = {
        "database_init": "Initializing database",
        "migrations": "Applying migrations",
        "collecting_static": "Collecting static files",
        "starting_server": "Starting server",
        "server_running": "Server is running",
    }

    completed_stages = set()
    last_log_position = 0

    while time.time() - start_time < max_wait_time:
        # Check if container is still running
        if not check_container_running(container_name):
            logger.info("Container stopped unexpectedly. Checking logs...")

            # Get exit code and error information
            exit_info = get_container_error_message(container_name)
            if exit_info:
                if "exit_code" in exit_info:
                    logger.info(f"Container exit code: {exit_info['exit_code']}")
                    if "meaning" in exit_info:
                        logger.info(f"Meaning: {exit_info['meaning']}")
                if "error" in exit_info and exit_info["error"]:
                    logger.info(f"Container error: {exit_info['error']}")

            # First check logs from the container
            log_cmd = ["docker", "logs", container_name]
            log_result = subprocess.run(log_cmd, capture_output=True, text=True, check=False)

            # Check for both stdout and stderr output
            if log_result.returncode == 0:
                logger.info("Container logs:")
                # Print stdout if not empty
                if log_result.stdout.strip():
                    logger.info("STDOUT:")
                    logger.info(log_result.stdout)
                else:
                    logger.info("No stdout output from container")

                # Print stderr if not empty
                if log_result.stderr.strip():
                    logger.info("STDERR:")
                    logger.info(log_result.stderr)

                # If both stdout and stderr are empty, try to get more diagnostics
                if not log_result.stdout.strip() and not log_result.stderr.strip():
                    logger.info("No logs found. Trying to get container information...")
                    # Get detailed container info
                    inspect_cmd = ["docker", "inspect", container_name]
                    inspect_result = subprocess.run(inspect_cmd, capture_output=True, text=True, check=False)
                    if inspect_result.returncode == 0:
                        try:
                            import json

                            container_info = json.loads(inspect_result.stdout)
                            if container_info and len(container_info) > 0:
                                state = container_info[0].get("State", {})
                                if "Error" in state and state["Error"]:
                                    logger.info(f"Container error: {state['Error']}")
                                exit_code = state.get("ExitCode")
                                if exit_code is not None:
                                    logger.info(f"Container exit code: {exit_code}")
                        except Exception as e:
                            logger.info(f"Failed to parse container inspect data: {e}")
            else:
                logger.info(f"Failed to get logs: {log_result.stderr}")

            return None, None

        # Try health endpoint
        try:
            response = requests.get(f"{server_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info(f"Label Studio is running at {server_url}")
                return server_url, predefined_token
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            pass  # Expected during startup

        # Check logs for progress
        log_cmd = ["docker", "logs", container_name]
        log_result = subprocess.run(log_cmd, capture_output=True, text=True, check=False)

        if log_result.returncode == 0:
            logs = log_result.stdout
            stderr_logs = log_result.stderr

            # Only process new log content (stdout)
            new_logs = logs[last_log_position:]
            last_log_position = len(logs)

            if new_logs:
                # Print each log line individually for better readability
                for line in new_logs.strip().split("\n"):
                    if line.strip():
                        logger.info(f"CONTAINER: {line.strip()}")

                # Check for initialization stages
                for stage, marker in init_stages.items():
                    if stage not in completed_stages and marker in logs:
                        completed_stages.add(stage)
                        logger.info(f"✓ Detected stage: {marker}")

                # Special case: look for database initialization completion
                if "database_init" in completed_stages and "migrations" not in completed_stages:
                    if "Migrations for 'auth':" in logs:
                        completed_stages.add("migrations")
                        logger.info(f"\n✓ Detected stage: Starting migrations")

                # Special case: look for server startup
                if "Starting development server at" in logs:
                    logger.info(f"\n✓ Detected stage: Server starting up")
                    # Give the server a moment to fully initialize
                    time.sleep(5)
                    try:
                        response = requests.get(f"{server_url}/health", timeout=5)
                        if response.status_code == 200:
                            logger.info(f"Label Studio is running at {server_url}")
                            return server_url, predefined_token
                    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                        logger.info("Server starting but health endpoint not ready yet...")

            # Check stderr logs for errors
            if stderr_logs and stderr_logs.strip():
                logger.info(f"CONTAINER STDERR: {stderr_logs.strip()}")

        # Print periodic status update
        elapsed = int(time.time() - start_time)
        remaining = max_wait_time - elapsed
        if elapsed % 10 == 0:  # Status update every 10 seconds
            logger.info(
                f"\nStill waiting for Label Studio to initialize... ({elapsed}s elapsed, {remaining}s remaining)"
            )
            logger.info(f"Completed stages: {len(completed_stages)}/{len(init_stages)}")

        time.sleep(check_interval)

    logger.info(f"Timeout after waiting {max_wait_time} seconds for Label Studio to start.")
    logger.info("Final container logs:")
    log_cmd = ["docker", "logs", "--tail", "100", container_name]
    log_result = subprocess.run(log_cmd, capture_output=True, text=True, check=False)
    if log_result.returncode == 0:
        # Print stdout if not empty
        if log_result.stdout.strip():
            logger.info("STDOUT:")
            logger.info(log_result.stdout)
        else:
            logger.info("No stdout output from container")

        # Print stderr if not empty
        if log_result.stderr.strip():
            logger.info("STDERR:")
            logger.info(log_result.stderr)

        # If both stdout and stderr are empty, try to get more diagnostics
        if not log_result.stdout.strip() and not log_result.stderr.strip():
            logger.info("No logs found. Getting container inspection data...")
            inspect_cmd = ["docker", "inspect", container_name]
            inspect_result = subprocess.run(inspect_cmd, capture_output=True, text=True, check=False)
            if inspect_result.returncode == 0:
                try:
                    import json

                    container_info = json.loads(inspect_result.stdout)
                    if container_info and len(container_info) > 0:
                        state = container_info[0].get("State", {})
                        logger.info(f"Container state: {json.dumps(state, indent=2)}")
                except Exception as e:
                    logger.info(f"Failed to parse container inspect data: {e}")

    # Check if container is still running
    if check_container_running(container_name):
        logger.info("Container is still running but Label Studio didn't respond in time.")
        logger.info("You may try accessing it manually at: " + server_url)
        logger.info("Or stop the container with: docker stop " + container_name)
    else:
        logger.info("Container is not running. Check the logs above for errors.")

    # Don't automatically stop the container - it might still be initializing
    return None, None


def get_container_error_message(container_name):
    """Get detailed error information from a container that has exited"""
    inspect_cmd = ["docker", "inspect", container_name]
    inspect_result = subprocess.run(inspect_cmd, capture_output=True, text=True, check=False)
    if inspect_result.returncode == 0:
        try:
            import json

            container_info = json.loads(inspect_result.stdout)
            if container_info and len(container_info) > 0:
                state = container_info[0].get("State", {})
                error = state.get("Error", "")

                # Common exit codes and their meanings
                exit_code = state.get("ExitCode")
                exit_code_meanings = {
                    125: "Docker command failed (e.g., unsupported option)",
                    126: "Command cannot be invoked (e.g., permission problem or not executable)",
                    127: "Command not found",
                    137: "Container received SIGKILL (possibly out of memory)",
                    143: "Container received SIGTERM",
                }

                result = {}

                if exit_code is not None:
                    result["exit_code"] = exit_code
                    if exit_code in exit_code_meanings:
                        result["meaning"] = exit_code_meanings[exit_code]

                if error:
                    result["error"] = error

                return result
        except Exception as e:
            logger.error(f"Error parsing container info: {e}")
    return None


def load_connection_info():
    """Load connection information from file"""
    connection_file = "label_studio_localhost_connection.json"
    if os.path.exists(connection_file):
        try:
            with open(connection_file, "r") as f:
                connection_info = json.load(f)
                logger.info(f"Loaded existing connection info from {connection_file}")
                return connection_info
        except Exception as e:
            logger.error(f"Error loading connection info: {e}")
    return None


def cleanup_test_project(server_url, api_token, project_id):
    """Clean up the test project after testing"""
    if not project_id:
        return

    try:
        from label_studio_sdk import Client

        client = Client(url=server_url, api_key=api_token)

        # Delete the project using the client's delete_project method
        client.delete_project(project_id)
        logger.info(f"Successfully deleted test project {project_id}")
    except Exception as e:
        logger.error(f"Error cleaning up test project: {e}")


def check_pip_installed():
    """Check if pip is installed"""
    try:
        subprocess.run(["pip", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        return True
    except FileNotFoundError:
        return False


def start_label_studio_pip(data_dir, username, password, port, predefined_token=None):
    """Start Label Studio using pip installation"""
    # Ensure data directory exists
    data_dir = os.path.abspath(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    # Generate a fixed token if not provided
    if not predefined_token:
        import uuid

        predefined_token = str(uuid.uuid4())
        logger.info(f"Generated fixed API token: {predefined_token}")

    # Set environment variables
    env = os.environ.copy()
    env["LABEL_STUDIO_USERNAME"] = username
    env["LABEL_STUDIO_PASSWORD"] = password
    env["LABEL_STUDIO_USER_TOKEN"] = predefined_token
    env["LABEL_STUDIO_BASE_DATA_DIR"] = data_dir
    env["LABEL_STUDIO_PORT"] = str(port)

    # Start Label Studio
    logger.info(f"Starting Label Studio on port {port}...")
    try:
        # Use Popen to start the process in the background
        process = subprocess.Popen(
            ["label-studio", "start", "--no-browser", f"--port={port}"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
        )

        # Register cleanup function to terminate the process on exit
        atexit.register(lambda: process.terminate())

        # Monitor the output for startup completion
        server_url = f"http://localhost:{port}"
        max_wait_time = 300  # 5 minutes
        start_time = time.time()

        logger.info("Waiting for Label Studio to initialize...")

        # Make stdout non-blocking
        fd = process.stdout.fileno()
        fl = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

        # Initialize variables for tracking server status
        server_started = False

        # Process output in real-time
        while time.time() - start_time < max_wait_time:
            # Check if process is still running
            if process.poll() is not None:
                logger.error("Label Studio process exited unexpectedly.")
                return None, None

            # Use select to check if there's data to read without blocking
            ready_to_read, _, _ = select.select([process.stdout], [], [], 0.1)

            if ready_to_read:
                # Read all available output
                output = process.stdout.read()
                if output:
                    # Process each line
                    for line in output.splitlines():
                        if line.strip():
                            logger.info(f"LABEL_STUDIO: {line.strip()}")

                            # Check for successful startup message
                            if "Label Studio is up and running" in line:
                                logger.info(f"Label Studio is running at {server_url}")
                                server_started = True

            # Check if the server is responding
            try:
                response = requests.get(f"{server_url}/health", timeout=2)
                if response.status_code == 200:
                    logger.info(f"Label Studio is running at {server_url}")
                    return server_url, predefined_token
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                pass  # Expected during startup

            # If we've detected server startup in the logs, give it a moment to fully initialize
            if server_started:
                time.sleep(2)
                try:
                    response = requests.get(f"{server_url}/health", timeout=2)
                    if response.status_code == 200:
                        logger.info(f"Label Studio is running at {server_url}")
                        return server_url, predefined_token
                except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                    # If health check fails, continue monitoring
                    server_started = False

        logger.error("Timed out waiting for Label Studio to start.")
        return None, None

    except Exception as e:
        logger.error(f"Error starting Label Studio: {e}")
        return None, None


def create_test_project(server_url, api_token):
    """Create a test project in Label Studio using SDK"""
    try:
        from label_studio_sdk import Client

        # Initialize the Label Studio client
        client = Client(url=server_url, api_key=api_token)

        # Create a test project
        project = client.create_project(
            title="Test Classification Project",
            description="A test project for text classification",
            label_config="""
<View>
  <Text name="text" value="$text"/>
  <Choices name="sentiment" toName="text" choice="single">
    <Choice value="Positive"/>
    <Choice value="Neutral"/>
    <Choice value="Negative"/>
  </Choices>
</View>
""",
        )

        if project:
            logger.info(f"Created test project: {project.title} (ID: {project.id})")

            # Create a sample task using import_tasks instead of create_task
            task_data = [{"data": {"text": "This is a sample text for annotation. Please classify the sentiment."}}]

            try:
                imported_tasks = project.import_tasks(task_data)
                if imported_tasks:
                    logger.info("Added sample task to the project")
                else:
                    logger.error("Failed to add sample task to the project")
            except Exception as e:
                logger.error(f"Error adding sample task: {e}")
                # Clean up the project if task creation fails
                try:
                    client.delete_project(project.id)
                    logger.info("Cleaned up test project due to task creation failure")
                except Exception as cleanup_error:
                    logger.error(f"Error cleaning up test project: {cleanup_error}")
                return None

            return project.id

        logger.error("Failed to create test project")
        return None

    except Exception as e:
        logger.error(f"Error creating test project using SDK: {e}")
        return None


def main():
    args = parse_args()

    # Initialize logging
    global logger
    logger = setup_logging(args.data_dir)

    logger.info("Label Studio Service starting")

    # Handle status request
    if args.status:
        if check_container_running(args.container_name):
            logger.info(f"Container {args.container_name} is running.")
            return 0
        elif check_container_exists(args.container_name):
            logger.info(f"Container {args.container_name} exists but is not running.")
            return 1
        else:
            logger.info(f"Container {args.container_name} does not exist.")
            return 1

    # Handle stop request
    if args.stop:
        stop_container(args.container_name, args.remove_volumes)
        return 0

    # Handle kill request
    if args.kill:
        kill_container(args.container_name)
        if args.remove_volumes:
            volume_name = f"{args.container_name}-data"
            remove_volume(volume_name)
        return 0

    # Load existing connection info if available
    existing_connection = load_connection_info()

    # Initialize variables with values from connection file
    api_token = None
    project_id = None
    username_from_file = None
    password_from_file = None
    server_url = None

    if existing_connection:
        if "api_token" in existing_connection:
            api_token = existing_connection["api_token"]

        if "username" in existing_connection:
            username_from_file = existing_connection["username"]

        if "password" in existing_connection:
            password_from_file = existing_connection["password"]

        if "project_id" in existing_connection:
            project_id = existing_connection["project_id"]

        if "api_url" in existing_connection:
            server_url = existing_connection["api_url"]

    # Override with command line arguments if provided
    if args.api_token:
        api_token = args.api_token
        logger.info(f"Overriding API token with command line argument")

    # Use username from command line or connection file, or prompt if neither is available
    if args.username:
        logger.info(f"Using username from command line: {args.username}")
    elif username_from_file:
        args.username = username_from_file
    else:
        import getpass

        logger.info("Label Studio requires an admin username.")
        args.username = input("Enter admin email: ").strip()
        if not args.username:
            args.username = "admin@example.com"
            logger.info(f"Using default username: {args.username}")

    # Use password from command line or connection file, or prompt if neither is available
    if args.password:
        logger.info("Using password from command line")
    elif password_from_file:
        args.password = password_from_file
    else:
        import getpass

        logger.info("Label Studio requires an admin password.")
        args.password = getpass.getpass("Enter admin password: ").strip()
        if not args.password:
            import random
            import string

            # Generate a random password if none provided
            args.password = "".join(random.choices(string.ascii_letters + string.digits, k=12))
            logger.info(f"Using generated password: {args.password}")

    # Validate email format
    import re

    email_pattern = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
    if not email_pattern.match(args.username):
        logger.info(f"Warning: '{args.username}' doesn't appear to be a valid email address.")
        logger.info("Label Studio requires a valid email format for the username.")
        use_anyway = input("Use this username anyway? (y/n): ").strip().lower()
        if use_anyway != "y":
            logger.info("Please restart with a valid email address as the username.")
            return 1

    # Determine whether to use Docker or pip based on args and availability
    if args.use_pip:
        logger.info("Using pip installation as requested")
        use_docker = False
    else:
        # Check if Docker is installed and running
        use_docker = check_docker_installed()
        if not use_docker:
            logger.warning("Docker is not available. Falling back to pip installation.")

    # Start Label Studio based on chosen method
    if use_docker:
        logger.info("Using Docker to run Label Studio (recommended)")
        server_url, api_token = start_label_studio_with_docker(args, api_token)
    else:
        logger.info("Using pip to install and run Label Studio")
        if not check_pip_installed():
            logger.error("Error: pip is not available. Please install pip and try again.")
            return 1

        # Start Label Studio using pip
        server_url, api_token = start_label_studio_pip(
            args.data_dir,
            args.username,
            args.password,
            args.port,
            api_token,  # Use token from connection file or command line
        )

    if not server_url:
        logger.error("Failed to start Label Studio. Exiting.")
        return 1

    # Create test project if requested
    if args.create_test_project and api_token:
        project_id = create_test_project(server_url, api_token)
        logger.info(f"Created test project with ID: {project_id}")

        # Clean up test project immediately after creation
        if project_id:
            logger.info("Cleaning up test project...")
            cleanup_test_project(server_url, api_token, project_id)
            project_id = None  # Reset project_id since it's been cleaned up

    # Display connection information but don't save it
    if api_token:
        connection_info = {"api_url": server_url, "api_key": api_token, "project_id": project_id}
        logger.info("\nConnection information:")
        logger.info(json.dumps(connection_info, indent=2))
        logger.info("\nNote: Connection information is not being saved automatically.")
        logger.info(
            "If you need to save this information, you can create a 'label_studio_localhost_connection.json' file manually."
        )
    else:
        logger.info("\nNo API token available.")
        logger.info(f"You can still access Label Studio manually at {server_url}")

    logger.info("\nLabel Studio is running. Press Ctrl+C to stop.")
    logger.info(f"Access the web interface at: {server_url}")
    logger.info(f"Login with:")
    logger.info(f"  Username: {args.username}")
    logger.info(f"  Password: {args.password}")
    if api_token:
        logger.info(f"  API Token: {api_token}")

    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\nStopping Label Studio...")
        # Clean up test project if it exists
        if project_id:
            cleanup_test_project(server_url, api_token, project_id)
        if use_docker:
            stop_container(args.container_name)
            logger.info("Label Studio container stopped.")
        else:
            logger.info("Label Studio process terminated.")

    return 0


def start_label_studio_with_docker(args, api_token):
    """
    Start Label Studio using Docker with the specified arguments
    """
    # Set of images to try if the previous one fails
    images_to_try = [
        args.image,  # Try the specified/default image first
        "heartexlabs/label-studio:1.7.3",  # Then try a specific older version
        "heartexlabs/label-studio:1.6.0",  # Then try an even older version
    ]

    # If args.image is not the default, add the default to the list
    default_image = "heartexlabs/label-studio:latest"
    if args.image != default_image and args.image not in images_to_try:
        images_to_try.append(default_image)

    # Try each image until one works
    server_url = None
    for image in images_to_try:
        logger.info(f"Attempting to run Label Studio with image: {image}")

        # Pull the Docker image
        if not pull_docker_image(image):
            logger.error(f"Failed to pull Docker image: {image}")
            logger.info("Trying next image if available...")
            continue  # Try the next image

        # Start container with appropriate mounting strategy based on user preference
        if args.use_host_mount:
            logger.info("Using host directory mount as requested...")
            server_url, api_token = start_label_studio_container(
                args.port,
                args.data_dir,
                args.username,
                args.password,
                image,
                args.container_name,
                args.network,
                api_token,  # Use token from connection file or command line
                use_host_mount=True,
            )
        else:
            # Default approach: Docker volumes
            logger.info("Using Docker named volume approach (default)...")
            server_url, api_token = start_label_studio_container(
                args.port,
                args.data_dir,
                args.username,
                args.password,
                image,
                args.container_name,
                args.network,
                api_token,  # Use token from connection file or command line
                use_host_mount=False,
            )

        if server_url:
            logger.info(f"Successfully started Label Studio using image: {image}")
            break  # If successful, no need to try other images
        else:
            logger.info(f"Failed to start using image: {image}")
            if image != images_to_try[-1]:
                logger.info("Trying next image...")

    return server_url, api_token


if __name__ == "__main__":
    sys.exit(main())
