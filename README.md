# AI Research Assistant

A local, privacy-first RAG (Retrieval-Augmented Generation) system for AI research papers.
Built from scratch without LangChain or LlamaIndex to provide full transparency and
learning value. Supports adaptive routing, query rewriting, document grading, corrective
RAG, multi-step retrieval, RAPTOR hierarchical summaries, semantic caching, and
faithfulness self-checks.

---

## Overview

This project implements a complete RAG pipeline for querying AI research papers entirely
on your local machine. Using Ollama for LLM/embedding services and Qdrant as the vector
database, the system provides:

- **Adaptive routing** - automatically selects the best retrieval strategy for each query
- **Query rewriting** - reformulates questions for better retrieval results
- **Document grading** - filters irrelevant retrieved chunks
- **Corrective RAG (CRAG)** - iterative retrieval with relevance evaluation
- **Multi-step retrieval** - decomposes complex questions into sub-questions
- **RAPTOR hierarchical summaries** - builds summary layers from child chunks
- **Semantic cache** - caches previous answers for fast repeat queries
- **Faithfulness self-check** - validates generated answers against sources
- **Evaluation harness** - runs structured evaluations and produces reports

---

## Features

| Category | Description |
|---|---|
| **Local-first** | Runs entirely on your machine with Ollama + Qdrant (no cloud APIs) |
| **Structure-aware chunking** | Preserves paper sections (Abstract, Introduction, Methods, Results, etc.) |
| **Hybrid retrieval** | Dense vector search (Qdrant) + BM25 keyword search |
| **Adaptive routing** | Chooses retrieval strategy based on query type/intent |
| **Document grading** | Filters irrelevant retrieved chunks; only relevant chunks used |
| **Cited answers** | Generates responses with paper citations and source metadata |
| **Corrective RAG** | Iterative retrieval with automatic correction when results are poor |
| **Multi-step retrieval** | Decomposes complex questions into sub-questions, retrieves per sub-question |
| **RAPTOR summaries** | Builds hierarchical summary layers from child chunks for improved retrieval |
| **Semantic cache** | Caches question→answer mappings for repeat queries (fast, no re-computation) |
| **Faithfulness self-check** | Validates generated answers are supported by retrieved sources |
| **Evaluation harness** | Run structured evaluations with metrics (latency, faithfulness, citation rate) |

---

## Architecture

![Architecture Overview](docs/architecture.webp)

---

## Supported Query Strategies

The adaptive router selects a strategy for each query. Supported strategies:

| Strategy | Description | When Used |
|---|---|---|
| `no_retrieval` | Answer from model knowledge without retrieval | Simple factual questions, general knowledge |
| `single_shot` | Corrective RAG (CRAG) - iterative retrieval with grading | Standard queries needing paper evidence |
| `multi_step` | Decompose question into sub-questions, retrieve per sub-question | Complex multi-part questions |
| `crag_raptor` | RAPTOR-enhanced hierarchical retrieval using summary layers | Deep research questions needing structure |
| `cached_*` | Answer served from semantic cache (fast path) | Repeat queries |

Each strategy produces a labeled answer (e.g., `crag_refine`, `multi_step_3sub_2retrievals`,
`raptor_survey`) so you can see which path was taken.

---

## Installation

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Install Ollama
#    - Download from: https://ollama.ai
#    - Start the service: ollama serve

# 3. Install Qdrant (via Docker)
#    - Install Docker: https://docker.com
#    - Start Qdrant: docker run -d -p 6333:6333 -p 6334:6334 \
#        -v $(pwd)/qdrant_storage:/qdrant/storage \
#        qdrant/qdrant

# 4. Configure environment
cp .env.example .env
#    Edit .env if needed (default settings should work for most cases)

# 5. Download models
ollama pull qwen2.5-coder:3b     # LLM for generation
ollama pull nomic-embed-text:v1.5  # Embedding model

# 6. (Optional) Download additional models
#    See .env.example for available model names
```

---

## How to Start Ollama + Qdrant

### Ollama

```bash
# Start the Ollama server
ollama serve

# In another terminal, pull required models
ollama pull qwen2.5-coder:3b    # LLM model for generation
ollama pull nomic-embed-text:v1.5  # Embedding model for search

# Verify models are available
ollama list
```

### Qdrant

```bash
# Start Qdrant via Docker (one-time setup)
docker run -d -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant

# Verify Qdrant is running
# Visit: http://localhost:6333 in your browser, or:
curl -s http://localhost:6333/ | head -5
```

---

## How to Ingest Papers

1. **Place PDF papers** in `data/papers/` directory
2. **Run ingestion:**

```bash
python scripts/ingest_papers.py
```

3. **Options:**

```bash
# Incremental (default) - only processes new/changed papers
python scripts/ingest_papers.py

# Force re-index all papers (ignores hashes)
python scripts/ingest_papers.py --force

# Show statistics only
python scripts/ingest_papers.py --stats

# Clean records for deleted files
python scripts/ingest_papers.py --clean
```

4. **What happens during ingestion:**

   - PDFs are parsed using docling (structure-aware)
   - Text is chunked using parent-child hierarchy (parent chunks for summaries, child chunks for retrieval)
   - Chunks are indexed into Qdrant vector database
   - Record metadata tracks processed files and hashes

---

## How to Build RAPTOR Summaries

RAPTOR builds hierarchical summary layers from child chunks, enabling
better retrieval for deep research questions.

```bash
# Build all summaries for all papers
python scripts/build_raptor.py

# Build for one paper by ID
python scripts/build_raptor.py --paper ai_papers

# Limit number of groups (useful for testing)
python scripts/build_raptor.py --limit 5

# Check existing summaries without building
python scripts/build_raptor.py --check

# Build all but limit to first N groups
python scripts/build_raptor.py --all --limit 3
```

**RAPTOR process:**

1. Retrieves all child chunks from Qdrant
2. Groups chunks by paper_id and section
3. Iteratively builds summary nodes using the LLM, creating a hierarchy
4. Summary nodes are stored in Qdrant for RAPTOR-enhanced retrieval
5. Each level of summary provides increasingly abstract representations

---

## How to Ask Questions

```bash
# Basic query (interactive or single-shot)
python src/main.py "What is the attention mechanism in transformers?"

# Interactive mode
python src/main.py --interactive

# With verbose logging
python src/main.py -v "Compare transformer architectures"

# Using specific strategy (bypass router)
# The router automatically selects the best strategy, but you can observe which one is chosen
```

**Query flow:**

1. Semantic cache check (if HIT, returns cached answer immediately)
2. Query rewriting for better retrieval
3. Adaptive routing selects strategy
4. Based on strategy:
   - `no_retrieval` → Answer from model knowledge
   - `single_shot` → Corrective RAG retrieval + generation
   - `multi_step` → Decompose + retrieve per sub-question
   - `crag_raptor` → RAPTOR-enhanced hierarchical retrieval
5. Answer generation with citations
6. Faithfulness self-check
7. Answer stored in cache

---

## How to Run Evaluation

The evaluation harness runs questions from a JSONL file through the pipeline and collects metrics.

```bash
# Run all questions in the default file
python scripts/run_eval.py

# Run first 5 questions only
python scripts/run_eval.py --limit 5

# Run and save detailed report
python scripts/run_eval.py --output my_report.json

# Run with custom questions file
python scripts/run_eval.py data/eval/questions.jsonl --limit 10
```

**Evaluation output:**

- Summary metrics: average latency, cache hit rate, faithfulness rate, citation rate,
  retrieval success rate
- Strategy distribution across all questions
- Per-question details: strategy, latency, cache hit/fail, faithfulness
- Detailed JSON report saved to specified output path

**Example questions JSONL format:**

```jsonl
{"id": "1", "question": "What is attention mechanism?", "type": "factoid", "must_cite": true}
{"id": "2", "question": "Compare transformer and LSTM architectures", "type": "comparative", "must_cite": true}
{"id": "3", "question": "How does backpropagation work?", "type": "explain", "must_cite": false}
```

---

## Project Structure

```
ai-research-assistant/
├── data/
│   ├── papers/                # Raw PDF papers
│   ├── processed/             # Processed chunks (JSON)
│   ├── eval/                  # Evaluation questions and reports
│   └── cache/                 # Semantic cache JSON
├── scripts/
│   ├── check_environment.py   # NEW - Environment validation
│   ├── ingest_papers.py       # Incremental paper ingestion pipeline
│   ├── build_raptor.py        # RAPTOR hierarchical summary builder
│   └── run_eval.py            # Evaluation harness
├── requirements.txt           # Python dependencies
├── .env.example              # Environment configuration template
├── README.md                 # This file
└── src/
    ├── main.py               # CLI entry point
    ├── core/
    │   ├── config.py         # Configuration management
    │   ├── llm.py            # Ollama LLM client
    │   └── embeddings.py     # Ollama embeddings client
    ├── ingestion/
    │   ├── pdf_parser.py     # PDF text extraction
    │   ├── chunker.py        # Structure-aware chunking
    │   ├── indexer.py        # Qdrant indexing
    │   └── raptor_builder.py # RAPTOR summary building
    ├── retrieval/
    │   ├── hybrid_retriever.py # Dense + BM25 search
    │   ├── grader.py         # Relevance grading
    │   ├── query_rewriter.py # Query rewriting
    │   ├── semantic_cache.py # Caching
    │   ├── corrective_retriever.py # CRAG
    │   ├── multi_step_retriever.py # Multi-step retrieval
    │   └── raptor_retriever.py # RAPTOR retrieval
    ├── routing/
    │   └── adaptive_router.py # Query routing
    ├── generation/
    │   └── answer_generator.py # Cited answer generation
    └── evaluation/
        ├── metrics.py         # Metrics computation
        └── evaluator.py       # Evaluation orchestration
```

---

## Notes on CPU Usage and Limitations

| Aspect | Note |
|---|---|
| **Model sizes** | Default LLM `qwen2.5-coder:3b` and embeddings `nomic-embed-text:v1.5` work on CPU but are slower. For better performance, consider larger models or GPU acceleration. |
| **Memory** | Loading multiple model instances consumes RAM. Default configuration is tuned for 8GB+ RAM. |
| **Qdrant** | Vector search on CPU is functional but slower than GPU-accelerated alternatives. |
| **RAPTOR** | Building summary layers requires multiple LLM calls. For large paper collections, use `--limit` to process incrementally. |
| **Multi-step** | Complex questions may generate multiple sub-questions, increasing latency proportionally. |
| **Cache** | Semantic cache provides speedup for repeat queries but cache entries expire with paper re-ingestion. |
| **No LangChain/LlamaIndex** | This project is self-contained with no external RAG frameworks. All logic is explicitly implemented. |
| **CPU-only** | The system is designed to work on CPU-only machines. Ollama + Qdrant both run locally without requiring cloud credits or GPU resources. |

---

## License

MIT
