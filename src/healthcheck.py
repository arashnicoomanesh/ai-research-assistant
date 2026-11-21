"""
Healthcheck module for AI Research Assistant.

Checks connectivity to Qdrant and Ollama services.
Exits 0 if both services are reachable, non-zero otherwise.
"""
import os
import sys
import requests
from urllib.parse import urlparse


def check_qdrant(qdrant_url: str, timeout: int = 5) -> bool:
    """Check if Qdrant is reachable and healthy."""
    try:
        # Parse URL to get base
        parsed = urlparse(qdrant_url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"

        # Check collections endpoint (lightweight)
        response = requests.get(f"{base_url}/collections", timeout=timeout)
        if response.status_code == 200:
            return True
        print(f"Qdrant healthcheck failed: status {response.status_code}", file=sys.stderr)
        return False
    except requests.exceptions.RequestException as e:
        print(f"Qdrant healthcheck failed: {e}", file=sys.stderr)
        return False


def check_ollama(ollama_host: str, timeout: int = 5) -> bool:
    """Check if Ollama is reachable and has models loaded."""
    try:
        # Parse URL to get base
        parsed = urlparse(ollama_host)
        base_url = f"{parsed.scheme}://{parsed.netloc}"

        # Check tags endpoint
        response = requests.get(f"{base_url}/api/tags", timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            if models:
                return True
            print("Ollama healthcheck: no models loaded", file=sys.stderr)
            return False
        print(f"Ollama healthcheck failed: status {response.status_code}", file=sys.stderr)
        return False
    except requests.exceptions.RequestException as e:
        print(f"Ollama healthcheck failed: {e}", file=sys.stderr)
        return False


def main() -> int:
    """Main healthcheck entry point.

    Returns:
        0 if both Qdrant and Ollama are healthy, 1 otherwise.
    """
    # Get URLs from environment with defaults
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    # Short timeouts for healthcheck
    timeout = int(os.getenv("HEALTHCHECK_TIMEOUT", "5"))

    print(f"Checking Qdrant at {qdrant_url}...", file=sys.stderr)
    qdrant_ok = check_qdrant(qdrant_url, timeout)

    print(f"Checking Ollama at {ollama_host}...", file=sys.stderr)
    ollama_ok = check_ollama(ollama_host, timeout)

    if qdrant_ok and ollama_ok:
        print("Healthcheck PASSED", file=sys.stderr)
        return 0
    else:
        print("Healthcheck FAILED", file=sys.stderr)
        return 1


if __name__ == "__main__":
    import os
    sys.exit(main())