#!/usr/bin/env bash
# scripts/demo_queries.sh
# Run curated demo queries to showcase automatic strategy selection.
# Prerequisites: Ollama + Qdrant running, papers ingested, optional: RAPTOR + GraphRAG built.

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

echo -e "${BLUE}=== AI Research Assistant — Strategy Demo ===${NC}"
echo ""

# --- Environment checks ---
echo -e "${YELLOW}Environment checks:${NC}"
if ! command -v ollama &> /dev/null; then
  echo -e "${RED}✗ ollama not in PATH${NC}"
  exit 1
fi
if ! ollama list 2>/dev/null | grep -q "qwen2.5-coder:3b"; then
  echo -e "${YELLOW}⚠ qwen2.5-coder:3b not pulled (run: ollama pull qwen2.5-coder:3b)${NC}"
fi
if ! ollama list 2>/dev/null | grep -q "nomic-embed-text:v1.5"; then
  echo -e "${YELLOW}⚠ nomic-embed-text:v1.5 not pulled (run: ollama pull nomic-embed-text:v1.5)${NC}"
fi
if ! curl -sf http://localhost:6333/ >/dev/null 2>&1; then
  echo -e "${RED}✗ Qdrant not reachable at http://localhost:6333 (start with docker run ... qdrant/qdrant)${NC}"
  exit 1
fi
echo -e "${GREEN}✓ Ollama + Qdrant reachable${NC}"
echo ""

# --- Run demo queries ---
run_query() {
  local label="$1"
  shift
  echo -e "${BLUE}────────────────────────────────────────${NC}"
  echo -e "${YELLOW}▶ ${label}${NC}"
  echo -e "${BLUE}────────────────────────────────────────${NC}"
  echo -e "Command: python src/main.py \"$*\""
  echo ""
  python src/main.py "$@" || echo -e "${RED}(query failed — check logs)${NC}"
  echo ""
}

run_query "Definition → single_shot / CRAG path" \
  "What is Self-RAG?"

run_query "Technique extraction → graph_rag tendency" \
  "What techniques does Self-RAG introduce?"

run_query "Survey / overview → crag_raptor tendency" \
  "Give me an overview of advanced RAG methods in these papers"

run_query "Repeat query → cache hit (instant)" \
  "What is Self-RAG?"

run_query "Relational (experimental) → graph_rag tendency" \
  "How is Self-RAG related to CRAG?"

echo -e "${BLUE}=== Demo complete ===${NC}"
echo ""
echo "Strategy labels appear in answer headers, e.g.:"
echo "  ⚡🕸️ Answer [cached_graph_rag_relational]"
echo "  ⚡📚 Answer [crag_refine]"
echo "  ⚡🌳 Answer [raptor_survey]"
echo ""
echo "Tips:"
echo "  • First run per query: 1–4 min on CPU (3B model)"
echo "  • Repeat runs: <2 sec via semantic cache"
echo "  • Build GraphRAG first for better technique/relational answers:"
echo "      python scripts/build_graph.py --limit 8"
echo "  • Build RAPTOR for better survey answers:"
echo "      python scripts/build_raptor.py --all"
echo "  • Run environment check:"
echo "      python scripts/check_environment.py"