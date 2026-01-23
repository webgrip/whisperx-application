from __future__ import annotations

import os
import socket
import sys

def main() -> int:
    # A basic connectivity check: try resolving and connecting to a well-known host.
    # In a truly offline environment, this should fail.
    host = os.environ.get("OFFLINE_TEST_HOST", "huggingface.co")
    try:
        socket.gethostbyname(host)
        s = socket.create_connection((host, 443), timeout=2.0)
        s.close()
        print(f"WARNING: Network connectivity to {host}:443 appears to be available.")
        return 1
    except Exception:
        print("OK: Network connectivity check failed as expected (offline).")
        return 0

if __name__ == "__main__":
    raise SystemExit(main())
