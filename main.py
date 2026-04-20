#!/usr/bin/env python3
import subprocess
import sys
import os
import signal
import time
from typing import Dict, List
import threading

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Map of service directory to port and color code for nice logging
SERVICES = {
    "ingestion-service": {"port": "8004", "color": "\033[96m", "name": "INGEST"},
    "llm-service":       {"port": "8001", "color": "\033[93m", "name": "LLM   "},
    "retrieval-service": {"port": "8002", "color": "\033[94m", "name": "RETRVL"},
    "kg-service":        {"port": "8003", "color": "\033[95m", "name": "KG    "},
    "orchestrator":      {"port": "8000", "color": "\033[92m", "name": "ORCHST"},
    "gateway":           {"port": "8080", "color": "\033[97m", "name": "GATEWY"},
}

processes: List[subprocess.Popen] = []

def stream_output(process: subprocess.Popen, name: str, color: str):
    """Reads lines from the process stdout and prints them with a colored prefix."""
    reset_color = "\033[0m"
    prefix = f"{color}[{name}]{reset_color} | "
    for line in iter(process.stdout.readline, b''):
        sys.stdout.write(f"{prefix}{line.decode('utf-8', errors='replace')}")

def start_services():
    print("\n🚀 Starting all microservices...\n")
    venv_python = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")
    if not os.path.exists(venv_python):
        print(f"❌ Error: Python virtual environment not found at {venv_python}")
        sys.exit(1)

    for svc_dir, config in SERVICES.items():
        port = config["port"]
        name = config["name"]
        color = config["color"]
        cwd = os.path.join(PROJECT_ROOT, "Src", svc_dir)

        # Load environment variables if .env exists
        env = os.environ.copy()
        env_file = os.path.join(cwd, ".env")
        if os.path.exists(env_file):
            with open(env_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        key, val = line.split("=", 1)
                        env[key.strip()] = val.strip().strip('"').strip("'")

        cmd = [venv_python, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", port, "--log-level", "warning"]

        try:
            p = subprocess.Popen(
                cmd,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
            )
            processes.append(p)
            
            # Start a background thread to read and prefix this process's output
            t = threading.Thread(target=stream_output, args=(p, name, color), daemon=True)
            t.start()
            
            print(f"{color}✅ Started {svc_dir} on port {port}{'\033[0m'}")
        except Exception as e:
            print(f"❌ Failed to start {svc_dir}: {e}")

    print("\n==============================================")
    print("  All services are launching!")
    print("  Gateway API Docs: http://localhost:8080/docs")
    print("  Press Ctrl+C to stop all services.")
    print("==============================================\n")

def shutdown_handler(signum, frame):
    print("\n🛑 Shutting down all services gracefully...")
    for p in processes:
        if p.poll() is None:
            p.terminate()
            
    # Wait for all processes to terminate
    for p in processes:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()  # Force kill if they hang
            
    print("✅ All services stopped.")
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    
    start_services()
    
    # Keep main thread alive waiting for processes or interrupts
    try:
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        pass # Handled by signal handler
