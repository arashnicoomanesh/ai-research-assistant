<div align="center">

# AI Research Assistant

A local-first, framework-free RAG laboratory that implements Adaptive routing, CRAG, multi-step retrieval, RAPTOR-inspired summaries, semantic cache, and Stage-A GraphRAG in one coherent CPU-runnable system.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ollama](https://img.shields.io/badge/Ollama-local-black?logo=ollama&logoColor=white)](https://ollama.ai)
[![Qdrant](https://img.shields.io/badge/Qdrant-vector%20DB-blueviolet)](https://qdrant.tech/)
[![Local-First](https://img.shields.io/badge/Privacy-Local--First-success)](https://github.com/arashnicoomanesh/ai-research-assistant)
[![No LangChain](https://img.shields.io/badge/No-LangChain%2FLlamaIndex-informational)](https://github.com/arashnicoomanesh/ai-research-assistant)
[![Version](https://img.shields.io/badge/Version-v0.4.0-blue)](https://github.com/arashnicoomanesh/ai-research-assistant/releases/tag/v0.4.0)

</div>

---

## Why this repo is different

- **Explicit implementations** — Core RAG logic written from scratch; no LangChain/LlamaIndex in the critical path
- **Multi-strategy RAG in one local system** — Adaptive routing, CRAG, multi-step, RAPTOR-inspired, GraphRAG (Stage A), semantic cache
- **CPU-runnable defaults** — 3B LLM + embeddings run on 8 GB RAM; `CPU_SAFE_MODE` keeps latency acceptable
- **Inspectable routing decisions** — Every answer shows the chosen strategy (`graph_rag_relational`, `raptor_survey`, `crag_refine`, `cached_*`)
- **Honest status + experimental GraphRAG Stage A** — Capability map below is the target scope; status audit is separate

## Positioning

This repository is a **RAG laboratory** — a complete, hackable system where several RAG techniques are implemented explicitly and run locally. It is not a wrapper around a framework; it is the framework, laid bare for study and extension. Use it to learn RAG internals, answer questions over AI research papers, or prototype new retrieval strategies without cloud APIs or hidden abstractions.

---

## Capability Map (Full Target Scope)

*What the lab covers — organized by subsystem. No status column here; see Implementation Status for the honest audit.*

### A. Data & Indexing
- PDF ingestion pipeline (structure-aware parsing via pdfplumber/Docling)
- Parent-child hierarchical chunking (sections preserved)
- Incremental ingestion ledger (SHA-256 + SQLite record manager)
- Qdrant dense vector indexing (child chunks searchable, parent chunks expandable)
- CPU-safe embedding policy (abstract-first / children-only modes)
- Metadata preservation (paper_id, section, chunk_type, token_count)

### B. Core Query Pipeline
- CLI QA interface (positional, `--question`, interactive)
- Adaptive query routing (heuristics + optional LLM classifier)
- Query rewriting (entity-aware, sometimes over-generalizes)
- Hybrid retrieval: dense vector (Qdrant `query_points`) + BM25 keyword search with RRF fusion
- Parent expansion (child→parent lookup via payload `chunk_id`)
- Cited answer generation (grounded, per-source citations)
- Semantic cache (similarity-based; optional TTL; invalidates on ingest/model change)
- CPU-Safe Mode (default ON; defers expensive ops)

### C. Advanced RAG Modules
- **Adaptive RAG** — Route per query type to the right strategy
- **Corrective RAG (CRAG)** — Iterative retrieval with rewrite/merge when quality low
- **Multi-step retrieval** — Decompose → sub-questions → per-sub retrieval → synthesize
- **RAPTOR-inspired hierarchical summaries** — Build/merge summary layers from child chunks (limited hierarchy; not full RAPTOR clustering tree)
- **Document grading** — LLM-based relevance filter (OFF by default in CPU mode)
- **Faithfulness self-check** — Answer-vs-sources validation (OFF by default in CPU mode)
- **Semantic cache / cost optimization** — Dedupe identical/similar queries; max-entries eviction; optional TTL; auto-invalidates on corpus re-ingest or LLM/embedding model change

### D. GraphRAG Stack
- Entity/relation extraction (LLM with 3-stage CPU fallback)
- Local graph store (NetworkX MultiDiGraph + JSON persistence)
- Alias canonicalization (e.g., "Self-RAG"/"SELF-RAG"/"Self Reflective RAG" → `self-rag`)
- `graph_rag` routing path (technique/relational queries)
- Heuristic entity linking (exact + fuzzy + token overlap)
- 1-hop neighborhood traversal + triple collection
- Supporting Qdrant chunk fusion for grounded answers
- Graph-backed answer generation
- **Edges require both endpoints; relation IDs include paper/chunk provenance** (no dangling edges, no overwriting across papers)
- **Stage B roadmap:** Community detection + global summary reports

### E. Evaluation, Product & Ops
- Environment checker (`scripts/check_environment.py`)
- Evaluation harness (latency, cache hit rate, citation rate, strategy distribution)
- CI (GitHub Actions)
- Demo script (`scripts/demo_queries.sh`)
- Architecture diagrams (`docs/architecture.webp`, `docs/graphrag-stage-a.webp`)
- Reproducible local setup (Docker Qdrant + Ollama)
- Future: strategy override CLI flag / optional UI

---

## Current Implementation Status

> Honest snapshot as of `2026-11-15`. Status labels: ✓ Working · ⚠️ Partial · 🧪 Experimental

### A. Data & Indexing
| Capability | Status | Notes |
|---|---|---|
| PDF ingestion | ✓ Working | 6 papers ingested |
| Structure-aware parsing | ⚠️ Partial | pdfplumber path works; Docling optional/fallback |
| Parent-child chunking | ✓ Working | Child for search, parent for expansion |
| Incremental indexing (hash + SQLite) | ✓ Working | Record manager tracks papers |
| Qdrant vector index | ✓ Working | `ai_papers` collection live |
| CPU-safe indexing | ✓ Working | Abstract/children-only modes |
| RAPTOR summary nodes | ✓ Working | Build + retrieval verified; summaries > 0 |

### B. Core Query Pipeline
| Capability | Status | Notes |
|---|---|---|
| CLI question interface | ✓ Working | positional / `--question` / interactive |
| Semantic cache | ✓ Working | Exact repeat = cache HIT |
| Query rewriting | ⚠️ Partial | Sometimes over-generalizes entities |
| Adaptive routing | ✓ Working | `no_retrieval` / `baseline` / `single_shot` / `multi_step` / `crag_raptor` / `graph_rag` |
| Hybrid retrieval (dense + BM25 + RRF) | ✓ Working | BM25 index persisted; RRF fusion (k=60) |
| Cross-encoder rerank | 🧪 Optional, OFF by default | `RERANK_ENABLED=false` default; requires sentence-transformers; not for weak CPU |
| Dense retrieval (`query_points`) | ✓ Working | Modern Qdrant API |
| Parent expansion | ✓ Working | Lookup by payload `chunk_id` |
| Answer generation + citations | ✓ Working | Grounded answers returned |
| CPU-Safe Mode | ✓ Working | Default ON for target hardware |

### C. Advanced RAG Modules
| Module | Status | Evidence |
|---|---|---|
| Adaptive RAG | ✓ Implemented | Router chooses strategy by query type |
| Corrective RAG (CRAG) | ✓ Implemented | Rewrite/merge path used (lightweight in CPU mode) |
| Multi-Step Retrieval | ✓ Implemented | Decomposition + per-sub retrieval code path |
| RAPTOR-inspired | ✓ Working | Builder + retriever paths verified; limited hierarchy |
| Document grading | 🧪 Implemented, OFF by default | LLM grader disabled in CPU-Safe Mode |
| Faithfulness self-check | 🧪 Implemented, OFF by default | Disabled in CPU-Safe Mode |
| Semantic Caching | ✓ Implemented | HIT path verified; similarity-based, max-entries eviction |

### D. GraphRAG Stack
| Capability | Status | Notes |
|---|---|---|
| Entity/relation extraction | 🧪 Experimental | LLM with 3-stage CPU fallback |
| Local graph store (NetworkX + JSON) | ✓ Working | Persists to `data/graph/graph.json` |
| Alias canonicalization | ✓ Working | Self-RAG / SELF-RAG / Self Reflective → `self-rag` |
| `graph_rag` routing | ⚠️ Partial | Heuristics + LLM; selects for technique/relational |
| Entity linking | 🧪 Experimental | Heuristic (exact/fuzzy/token overlap); can over-link |
| 1-hop traversal + triples | ✓ Working | Neighborhood + supporting chunk fusion |
| Grounded graph answers | 🧪 Experimental | Quality varies with extraction noise |
| Stage B (communities/global) | 🚫 Not implemented | Roadmap item |

### E. Engineering / Ops
| Capability | Status | Notes |
|---|---|---|
| Environment checker | ✓ Working | `scripts/check_environment.py` |
| Ingest / stats CLI | ✓ Working | Stats vector count display still buggy |
| Eval harness | ✓ Working | Code present |
| CI | ✓ Working | GitHub Actions passing |
| Architecture diagrams | ✓ Added | `docs/architecture.webp`, `docs/graphrag-stage-a.webp` |

### What works well now
- End-to-end QA with citations on local hardware (Ollama + Qdrant)
- RAPTOR-inspired hierarchical summaries built and retrieved correctly
- GraphRAG answers technique/relational queries (e.g., "What techniques does X introduce?")
- CPU-Safe defaults keep the system responsive on 3B models
- Semantic cache makes repeat queries instant

### Known limitations (GraphRAG)
- Entity extraction noise: some spurious entities/relations in `graph.json`
- Entity linking is heuristic (exact + fuzzy + token overlap) — can over-link or miss variants
- Graph built from CPU-model extractions; quality improves with larger models
- No community detection / global report generation (Microsoft-style GraphRAG not implemented)
- `--dedupe` cleans aliases post-hoc but does not fix extraction errors

---

## 🏗️ Architecture

![Architecture](docs/architecture.webp)

---

## 🕸️ GraphRAG Stage A


**Scope:** Local entity–relation graph extracted from paper chunks.

**Flow:** extract → graph store → entity linking → 1-hop neighborhood traversal → supporting chunks → grounded answer

**Stage A status (current):** Implemented end-to-end; **experimental, opt-in via `EXPERIMENTAL_ROUTING=true`** (Reliable Mode default forces `single_shot` for non-relational queries)
- `scripts/build_graph.py` builds the graph (LLM extraction with 3-stage CPU fallback)
- Entity IDs normalized; aliases merged (e.g., "Self-RAG"/"SELF-RAG" → `self-rag`)
- Graph stored as NetworkX JSON (`data/graph/graph.json`, local runtime artifact)
- Adaptive router selects `graph_rag` for technique/relational queries when experimental routing is enabled
- Retrieval links query entities, expands 1-hop neighborhood, fetches supporting Qdrant chunks

**Stage B (planned):** Community detection + global summary generation (Microsoft-style GraphRAG)

**Practical commands:**
```bash
# Abstracts-first, small batch (fast on CPU)
python scripts/build_graph.py --limit 8

# Deduplicate nodes by alias map (no LLM, ~1s)
python scripts/build_graph.py --dedupe

# Show node/edge counts and type breakdown
python scripts/build_graph.py --stats
```

> **Notes:** GraphRAG is experimental. Entity linking is heuristic and can be noisy; extraction quality improves with larger models. Use `--limit` and abstracts-first on CPU. `graph.json` is not committed to git (ignored via `data/graph/*.json`).

---

## Supported Query Strategies

The adaptive router selects a strategy per query. Each answer header shows the strategy label.

| Strategy | Description | Typical Trigger |
|---|---|---|
| `no_retrieval` | Answer from model knowledge without retrieval. | Simple factual / definitional |
| `baseline` | Dense retrieve top-k + generate. No CRAG rewrite loop. | Forced via `FORCE_STRATEGY=baseline` |
| `single_shot` | CRAG path — single retrieval + optional corrective step. Default in Reliable Mode. | Standard fact-seeking |
| `multi_step` | Decompose → sub-questions → per-sub retrieval → synthesize. Requires `EXPERIMENTAL_ROUTING=true`. | Complex multi-part |
| `crag_raptor` | RAPTOR-inspired hierarchical retrieval via summary layers. Requires `EXPERIMENTAL_ROUTING=true`. | Survey / overview / deep-dive |
| `graph_rag` | Entity linking → 1-hop traversal → grounded answer. Requires `EXPERIMENTAL_ROUTING=true`. | Technique list / relational |
| `cached_*` | Served from semantic cache (instant). | Exact repeat query |

**Reliable Mode (default: `RELIABLE_MODE=true`, `EXPERIMENTAL_ROUTING=false`):** Forces `single_shot` (CRAG) for all queries except `no_retrieval` and `baseline`. Experimental strategies (`multi_step`, `crag_raptor`, `graph_rag`) only activate when `EXPERIMENTAL_ROUTING=true`. Use `FORCE_STRATEGY=baseline` to explicitly select the vanilla retrieve+generate path.

---

## 🎬 Demo: Automatic Strategy Selection

The router picks the strategy automatically. Run these and watch the strategy label in the answer header/logs.

**Curated demo queries:**

```bash
# Definition-style → routes to single_shot / CRAG path (Reliable Mode default)
python src/main.py "What is Self-RAG?"

# Technique extraction → routes to graph_rag (requires EXPERIMENTAL_ROUTING=true)
EXPERIMENTAL_ROUTING=true python src/main.py "What techniques does Self-RAG introduce?"

# Survey / overview → routes to crag_raptor (RAPTOR path, requires EXPERIMENTAL_ROUTING=true)
EXPERIMENTAL_ROUTING=true python src/main.py "Give me an overview of advanced RAG methods in these papers"

# Cache demo (second run should be fast)
python src/main.py "What is Self-RAG?"

# Optional relational demo (experimental — graph may be noisy)
EXPERIMENTAL_ROUTING=true python src/main.py "How is Self-RAG related to CRAG?"
```

**Expected routing tendencies** (not guarantees — router uses heuristics + LLM; Reliable Mode forces single_shot unless EXPERIMENTAL_ROUTING=true):

| Query pattern | Typical strategy | What you see |
|---|---|---|
| "What is X?" | `single_shot` / `cached_*` | Definition + citations |
| "What techniques does X introduce?" | `graph_rag` (if experimental) | Techniques from graph edges |
| "How is X related to Y?" | `graph_rag` (if experimental) | Path/edges between entities |
| "Overview of Z" / "Survey of …" | `crag_raptor` (if experimental) | Summary from RAPTOR layers |
| Repeat identical query | `cached_*` | Near-instant cached answer |

**Timing notes (CPU, 3B model):**
- First answers: 1–4 minutes (model loading + retrieval + generation)
- Cache hits: <2 seconds
- Strategy name appears in answer header (e.g., `graph_rag_relational`, `raptor_survey`, `crag_refine`)

> Note: forced `--strategy` CLI flag is not the main interface yet; auto-routing + env flags (`CPU_SAFE_MODE`, `RELIABLE_MODE`, `EXPERIMENTAL_ROUTING`) control behavior.

---

## 📑 Quick Start

> **⚡ CPU Note:** CPU-Safe Mode is ON by default. First answers take 1–4 min; repeats are instant via cache. Ollama + Qdrant must be running before you query.

```bash
# 1. Clone the repo
git clone https://github.com/arashnicoomanesh/ai-research-assistant.git
cd ai-research-assistant

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Start Ollama and pull models
ollama serve
ollama pull qwen2.5-coder:3b
ollama pull nomic-embed-text:v1.5

# 4. Start Qdrant (Docker)
docker run -d -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant

# 5. Verify environment (optional but recommended)
python scripts/check_environment.py

# 6. Configure environment
cp .env.example .env

# 7. Ingest papers (place PDFs in data/papers/ first)
python scripts/ingest_papers.py

# 8. Build RAPTOR summaries (optional, ~2–5 min per paper)
python scripts/build_raptor.py --all

# 9. Build GraphRAG graph (optional, abstracts-first ~30–60s)
python scripts/build_graph.py --limit 8

# 10. Ask a question
python src/main.py "What is the attention mechanism in transformers?"
```

**Prerequisites:** Python 3.10+, [Ollama](https://ollama.ai), [Docker](https://docker.com) (for Qdrant).

---

## 🛠️ Installation

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Install & Start Ollama
```bash
# Download from https://ollama.ai, then:
ollama serve
ollama pull qwen2.5-coder:3b      # LLM for generation
ollama pull nomic-embed-text:v1.5 # Embedding model for search
ollama list
```

### 3. Install & Start Qdrant (via Docker)
```bash
docker run -d -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
curl -s http://localhost:6333/ | head -5
```

### 4. Configure Environment
```bash
cp .env.example .env
# Edit .env if needed (defaults work for most cases)
```

### 5. (Optional) Download Additional Models
See `.env.example` for available model names.

---

## 🚀 Usage

### 📥 Ingest Papers
```bash
# Place PDFs in data/papers/ first
python scripts/ingest_papers.py                 # Incremental
python scripts/ingest_papers.py --force         # Re-index all
python scripts/ingest_papers.py --stats         # Show stats
python scripts/ingest_papers.py --clean         # Clean deleted
```

### 🌳 Build RAPTOR Summaries
```bash
python scripts/build_raptor.py                  # Build all
python scripts/build_raptor.py --paper ai_papers
python scripts/build_raptor.py --limit 5        # Test subset
python scripts/build_raptor.py --check          # Inspect existing
python scripts/build_raptor.py --all --limit 3  # Cap groups
```

> **Note:** This is a RAPTOR-inspired implementation with limited hierarchy (section-level and paper-level summaries only), not a full recursive RAPTOR clustering tree.

### 🕸️ Build GraphRAG Knowledge Graph
```bash
# Recommended: abstracts-first (CPU-friendly)
python scripts/build_graph.py --limit 8
python scripts/build_graph.py --rebuild-abstracts --limit 12

# Full / incremental
python scripts/build_graph.py                   # All papers
python scripts/build_graph.py --paper ai_papers
python scripts/build_graph.py --stats           # Node/edge counts
python scripts/build_graph.py --dedupe          # Merge aliases (no LLM)
```

### 💬 Ask Questions
```bash
python src/main.py "What is the attention mechanism in transformers?"
python src/main.py --interactive
python src/main.py -v "Compare transformer architectures"
```

### 🌐 UI (Streamlit)
```bash
pip install -r requirements.txt
streamlit run ui/app.py
```

### 📊 Run Evaluations
```bash
python scripts/run_eval.py
python scripts/run_eval.py --limit 5
python scripts/run_eval.py --output my_report.json
python scripts/run_eval.py data/eval/questions.jsonl --limit 10
```

---

## 📂 Project Structure

```
ai-research-assistant/
├── data/
│   ├── papers/              # Raw PDF papers (user-provided)
│   ├── processed/           # Processed chunks (JSON) — gitignored
│   ├── eval/                # Evaluation questions and reports
│   ├── cache/               # Semantic cache JSON — gitignored
│   └── graph/               # GraphRAG runtime artifacts — gitignored
│       ├── graph.json       # NetworkX graph (local only)
│       └── processed_chunks.json
├── scripts/
│   ├── check_environment.py # Environment validation
│   ├── ingest_papers.py     # Incremental paper ingestion pipeline
│   ├── build_raptor.py      # RAPTOR hierarchical summary builder
│   ├── build_graph.py       # GraphRAG knowledge graph builder
│   ├── run_eval.py          # Evaluation harness
│   └── demo_queries.sh      # Strategy selection demo
├── src/
│   ├── main.py              # CLI entry point
│   ├── core/
│   │   ├── config.py        # Configuration management
│   │   ├── llm.py           # Ollama LLM client
│   │   └── embeddings.py    # Ollama embeddings client
│   ├── graph/
│   │   ├── __init__.py      # GraphRAG package
│   │   ├── schema.py        # Entity, Relation, GraphEvidence models
│   │   ├── entity_extractor.py # LLM-based entity/relation extraction
│   │   ├── graph_store.py   # NetworkX graph persistence
│   │   └── graph_retriever.py # Graph retrieval (entity linking + traversal)
│   ├── ingestion/
│   │   ├── pdf_parser.py    # PDF text extraction
│   │   ├── chunker.py       # Structure-aware chunking
│   │   ├── indexer.py       # Qdrant indexing
│   │   └── raptor_builder.py # RAPTOR summary building
│   ├── retrieval/
│   │   ├── hybrid_retriever.py   # Dense vector search + parent expansion
│   │   ├── grader.py             # Relevance grading
│   │   ├── query_rewriter.py     # Query rewriting
│   │   ├── semantic_cache.py     # Caching
│   │   ├── corrective_retriever.py # CRAG
│   │   ├── multi_step_retriever.py # Multi-step retrieval
│   │   ├── raptor_retriever.py     # RAPTOR retrieval
│   │   └── graph_retriever.py      # GraphRAG retrieval
│   ├── routing/
│   │   └── adaptive_router.py # Query routing
│   ├── generation/
│   │   └── answer_generator.py # Cited answer generation
│   └── evaluation/
│       ├── metrics.py         # Metrics computation
│       └── evaluator.py       # Evaluation orchestration
├── requirements.txt
├── .env.example
├── README.md
└── docs/
    ├── architecture.webp
    └── graphrag-stage-a.webp
```

---

## ⚠️ Limitations & Notes on CPU Usage

| Aspect | Note |
|---|---|
| **Model sizes** | Default LLM `qwen2.5-coder:3b` and embeddings `nomic-embed-text:v1.5` work on CPU but are slower. Larger models or GPU improve quality/speed. |
| **Memory** | Multiple model instances consume RAM. Config tuned for 8 GB+. |
| **Qdrant** | Vector search on CPU is functional but slower than GPU-accelerated. |
| **RAPTOR** | Building summary layers needs multiple LLM calls. Use `--limit` for large collections. Limited hierarchy (section-level summaries); not full RAPTOR clustering tree. |
| **GraphRAG extraction** | Entity/relation extraction ~1–2 min/chunk on 3B CPU. Prefer `--limit` and abstracts-first. |
| **Multi-step** | Complex questions generate multiple sub-questions; latency scales. |
| **Cache** | Semantic cache speeds repeats; max-entries eviction (oldest removed); no TTL expiration. |
| **CPU_SAFE_MODE** | Default ON: disables grader, self-check, RAPTOR self-check for latency. |
| **RELIABLE_MODE** | Default ON: forces `single_shot` (CRAG) unless `EXPERIMENTAL_ROUTING=true`. |
| **First-run latency** | First answers 1–4 min; repeats <2 s via cache. |

---

## 🔒 Security Notes

- **Grounded answers only:** The answer generator system prompt explicitly instructs the model to use *only* the provided context and to ignore any instructions embedded in retrieved text that attempt to override system rules. This mitigates prompt-injection via retrieved content.
- **No external network calls at query time:** All inference runs locally via Ollama; no data leaves your machine during query processing.

---

## Roadmap

- **R2** GraphRAG linker/router hardening; strategy override CLI flag
- **Eval** Baseline reports + strategy distribution dashboards
- **Stage B** Community detection + global summaries for GraphRAG
- Optional lightweight UI for interactive exploration

---

## 🤝 Contributing

1. **Report bugs** — Clear description + reproduction steps
2. **Suggest features** — Use case + proposed solution
3. **Submit PRs** — Fork → feature branch → clear description
4. **Improve docs** — Fix typos, clarify instructions, add examples

Follow existing code style; add tests where applicable.

### Running Tests

```bash
# Run smoke tests (no Ollama/Qdrant required)
pytest -q
```

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
