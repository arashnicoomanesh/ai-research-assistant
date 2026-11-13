#!/usr/bin/env python3
"""
Environment Check Script for AI Research Assistant.

Checks that all required services, models, and directories are available
before running the pipeline. Prints clear pass/fail messages.
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.config import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("check_environment")


def check_ollama_reachable() -> bool:
    """Check if Ollama server is reachable."""
    import urllib.request
    try:
        req = urllib.request.Request(f"{config.ollama.host}/api/tags")
        req.add_header("Accept", "application/json")
        response = urllib.request.urlopen(req, timeout=5)
        response.read()
        logger.info("Ollama server reachable at %s", config.ollama.host)
        return True
    except Exception as e:
        logger.warning("Ollama server not reachable at %s: %s", config.ollama.host, e)
        return False


def check_ollama_models() -> bool:
    """Check if required Ollama models are available."""
    import urllib.request

    required_models = [
        config.ollama.llm_model,
        config.ollama.embedding_model,
    ]
    all_available = True

    try:
        req = urllib.request.Request(f"{config.ollama.host}/api/tags")
        req.add_header("Accept", "application/json")
        response = urllib.request.urlopen(req, timeout=5)
        models_data = response.read()
        import json
        available_models = json.loads(models_data).get("models", [])

        # Extract just the model names
        available_names = [m.get("name", "") for m in available_models]

        for required in required_models:
            if any(required in name for name in available_names):
                logger.info("Model available: %s", required)
            else:
                logger.warning("Model NOT available: %s (available: %s)", required, available_names[:5])
                all_available = False

        return all_available
    except Exception as e:
        logger.warning("Cannot check Ollama models: %s", e)
        return False


def check_qdrant_reachable() -> bool:
    """Check if Qdrant server is reachable."""
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        client = QdrantClient(host=config.qdrant.host, port=config.qdrant.port)
        # Try to get collection info
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]

        if config.qdrant.collection in collection_names:
            logger.info("Qdrant reachable at %s:%d, collection '%s' exists",
                       config.qdrant.host, config.qdrant.port, config.qdrant.collection)
            return True
        else:
            logger.warning("Qdrant reachable at %s:%d but collection '%s' not found",
                         config.qdrant.host, config.qdrant.port, config.qdrant.collection)
            logger.info("Available collections: %s", collection_names)
            return False

    except Exception as e:
        logger.warning("Qdrant not reachable at %s:%d: %s",
                     config.qdrant.host, config.qdrant.port, e)
        return False


def check_data_directories() -> bool:
    """Check that required data directories exist (or can be created)."""
    all_exist = True
    directories = [
        config.paths.papers_dir,
        config.paths.processed_dir,
    ]

    for d in directories:
        if d.exists():
            logger.info("Directory exists: %s", d)
        else:
            logger.warning("Directory NOT found: %s - will be created if needed", d)
            try:
                d.mkdir(parents=True, exist_ok=True)
                logger.info("Created directory: %s", d)
            except Exception as e:
                logger.error("Cannot create directory %s: %s", d, e)
                all_exist = False

    return all_exist


def check_qdrant_storage() -> bool:
    """Check Qdrant storage directory exists."""
    # Qdrant storage is typically at project_root/qdrant_storage
    project_root = Path(__file__).parent.parent
    storage_path = project_root / "qdrant_storage"
    if storage_path.exists():
        logger.info("Qdrant storage directory exists: %s", storage_path)
        return True
    else:
        logger.warning("Qdrant storage directory NOT found: %s", storage_path)
        logger.info("Create with: docker run -v $(pwd)/qdrant_storage:/qdrant/storage ...")
        return False


def main() -> int:
    """Run all environment checks and report results."""
    logger.info("=" * 60)
    logger.info("AI Research Assistant - Environment Check")
    logger.info("=" * 60)

    results = []

    # Check Ollama reachability
    ollama_reachable = check_ollama_reachable()
    results.append(("Ollama reachable", ollama_reachable))

    # Check Ollama models
    ollama_models = check_ollama_models()
    results.append(("Ollama models available", ollama_models))

    # Check Qdrant reachability
    qdrant_reachable = check_qdrant_reachable()
    results.append(("Qdrant reachable", qdrant_reachable))

    # Check Qdrant storage
    qdrant_storage = check_qdrant_storage()
    results.append(("Qdrant storage", qdrant_storage))

    # Check data directories
    data_dirs = check_data_directories()
    results.append(("Data directories", data_dirs))

    # Summary
    logger.info("=" * 60)
    logger.info("ENVIRONMENT CHECK SUMMARY")
    logger.info("=" * 60)

    all_pass = True
    for check_name, passed in results:
        status = "PASS" if passed else "FAIL"
        logger.info("  [%s] %s", status, check_name)
        if not passed:
            all_pass = False

    logger.info("=" * 60)
    if all_pass:
        logger.info("All checks passed! You're ready to run the assistant.")
    else:
        logger.warning("Some checks failed. See above for details.")
        logger.info("Run: cp .env.example .env  # to configure settings")
        logger.info("    Start Ollama: ollama serve")
        logger.info("    Start Qdrant: docker run -d -p 6333:6333 -p 6334:6334 \\")
        logger.info("                      -v $(pwd)/qdrant_storage:/qdrant/storage \\")
        logger.info("                      qdrant/qdrant")
        logger.info("    Pull models: ollama pull qwen2.5-coder:3b && ollama pull nomic-embed-text:v1.5")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())