# Makefile for AI Research Assistant Docker Operations
# Usage: make <target>
# All targets run via docker compose

.PHONY: help up down logs ps restart build pull clean ingest rebuild-bm25 eval health reset

# Default target
help:
	@echo "AI Research Assistant - Docker Commands"
	@echo ""
	@echo "Infrastructure:"
	@echo "  make up           - Start all services (detached)"
	@echo "  make down         - Stop and remove containers"
	@echo "  make logs         - Follow logs for all services"
	@echo "  make ps           - Show container status"
	@echo "  make restart      - Restart all services"
	@echo "  make build        - Build app image"
	@echo "  make pull         - Pull base images"
	@echo ""
	@echo "Data & Ingestion:"
	@echo "  make ingest       - Ingest papers (incremental)"
	@echo "  make rebuild-bm25 - Rebuild BM25 keyword index"
	@echo ""
	@echo "Evaluation:"
	@echo "  make eval         - Run evaluation (limit 3)"
	@echo "  make eval-full    - Run full evaluation"
	@echo ""
	@echo "Health & Debug:"
	@echo "  make health       - Run healthcheck (Qdrant + Ollama)"
	@echo "  make shell        - Open shell in app container"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean        - Remove containers and volumes (WARNING: data loss)"
	@echo "  make reset        - Full reset: down + clean + up"

# ============================================================
# Infrastructure
# ============================================================
up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

ps:
	docker compose ps

restart:
	docker compose restart

build:
	docker compose build app

pull:
	docker compose pull

# ============================================================
# Data & Ingestion
# ============================================================
ingest:
	docker compose run --rm app python scripts/ingest_papers.py

rebuild-bm25:
	docker compose run --rm app python scripts/ingest_papers.py --rebuild-bm25

build-raptor:
	docker compose run --rm app python scripts/build_raptor.py --all

build-graph:
	docker compose run --rm app python scripts/build_graph.py --limit 8

# ============================================================
# Evaluation
# ============================================================
eval:
	docker compose run --rm app python scripts/run_eval.py --limit 3

eval-full:
	docker compose run --rm app python scripts/run_eval.py

# ============================================================
# Health & Debug
# ============================================================
health:
	docker compose run --rm app python -m src.healthcheck

shell:
	docker compose run --rm app /bin/bash

# ============================================================
# Maintenance
# ============================================================
clean:
	docker compose down -v --remove-orphans
	docker volume prune -f

reset: clean up
	@echo "Waiting for services to be healthy..."
	@sleep 10
	@$(MAKE) health