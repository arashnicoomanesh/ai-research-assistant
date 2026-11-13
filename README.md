# 🧠 AI Research Assistant

> **A local, privacy-first RAG system for AI research papers — built from scratch without LangChain or LlamaIndex.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ollama](https://img.shields.io/badge/Ollama-local-black?logo=ollama&logoColor=white)](https://ollama.ai)
[![Qdrant](https://img.shields.io/badge/Qdrant-vector%20DB-blueviolet)](https://qdrant.tech/)
[![Local-First](https://img.shields.io/badge/Privacy-Local--First-success)](https://github.com/arashnicoomanesh/ai-research-assistant)
[![No LangChain](https://img.shields.io/badge/No-LangChain%2FLlamaIndex-informational)](https://github.com/arashnicoomanesh/ai-research-assistant)

---

## 📑 Table of Contents

- [Why This Project?](#-why-this-project)
- [Features](#-features)
- [Architecture](#-architecture)
- [Supported Query Strategies](#-supported-query-strategies)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
  - [Ingest Papers](#-ingest-papers)
  - [Build RAPTOR Summaries](#-build-raptor-summaries)
  - [Ask Questions](#-ask-questions)
  - [Run Evaluations](#-run-evaluations)
- [Project Structure](#-project-structure)
- [Limitations & Notes on CPU Usage](#-limitations--notes-on-cpu-usage)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Why This Project?

Most RAG tutorials rely on frameworks like LangChain or LlamaIndex, which abstract away the core logic. This project takes the opposite approach: **every component is implemented explicitly** so you can see exactly how a production-grade RAG pipeline works. It runs entirely on your local machine using **Ollama** for LLM/embedding services and **Qdrant** as the vector database—no cloud APIs, no hidden magic, no data leaving your computer.

Whether you're learning RAG internals, researching AI papers, or building a privacy-sensitive document QA system, this repository provides a transparent, hackable foundation.

---

## ✨ Features

| Category | Description |
|---|---|
| **🔒 Local-first** | Runs entirely on your machine with Ollama + Qdrant (no cloud APIs). |
| **📄 Structure-aware chunking** | Preserves paper sections (Abstract, Introduction, Methods, Results, etc.). |
| **🔍 Hybrid retrieval** | Dense vector search (Qdrant) + BM25 keyword search. |
| **🧭 Adaptive routing** | Chooses the retrieval strategy based on query type/intent. |
| **📊 Document grading** | Filters irrelevant retrieved chunks; only relevant chunks are used. |
| **📝 Cited answers** | Generates responses with paper citations and source metadata. |
| **🔄 Corrective RAG (CRAG)** | Iterative retrieval with automatic correction when results are poor. |
| **🧩 Multi-step retrieval** | Decomposes complex questions into sub-questions and retrieves per sub-question. |
| **🌳 RAPTOR summaries** | Builds hierarchical summary layers from child chunks for improved retrieval. |
| **⚡ Semantic cache** | Caches question→answer mappings for repeat queries (fast, no re-computation). |
| **✅ Faithfulness self-check** | Validates that generated answers are supported by retrieved sources. |
| **📈 Evaluation harness** | Runs structured evaluations with metrics (latency, faithfulness, citation rate). |

---

## 🏗️ Architecture

![Architecture](docs/architecture.webp)
---

## 🧭 Supported Query Strategies

The adaptive router selects a strategy for each query. Supported strategies:

| Strategy | Description | When Used |
|---|---|---|
| `no_retrieval` | Answer from model knowledge without retrieval. | Simple factual questions, general knowledge. |
| `single_shot` | Corrective RAG (CRAG) — iterative retrieval with grading. | Standard queries needing paper evidence. |
| `multi_step` | Decompose question into sub-questions, retrieve per sub-question. | Complex multi-part questions. |
| `crag_raptor` | RAPTOR-enhanced hierarchical retrieval using summary layers. | Deep research questions needing structure. |
| `cached_*` | Answer served from semantic cache (fast path). | Repeat queries. |

Each strategy produces a **labeled answer** (e.g., `crag_refine`, `multi_step_3sub_2retrievals`, `raptor_survey`) so you can see exactly which path was taken.

---



## 📑 Quick Start

For the impatient, here's the minimal path to a working system:

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

# 5. Configure environment
cp .env.example .env

# 6. Ingest papers (place PDFs in data/papers/ first)
python scripts/ingest_papers.py

# 7. Ask a question
python src/main.py "What is the attention mechanism in transformers?"
```

> **Prerequisites:** Python 3.10+, [Ollama](https://ollama.ai), [Docker](https://docker.com) (for Qdrant).

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

# Pull required models
ollama pull qwen2.5-coder:3b      # LLM for generation
ollama pull nomic-embed-text:v1.5 # Embedding model for search

# Verify models are available:
ollama list
```

### 3. Install & Start Qdrant (via Docker)

```bash
# Install Docker from https://docker.com, then:
docker run -d -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant

# Verify Qdrant is running:
curl -s http://localhost:6333/ | head -5
# Or visit http://localhost:6333 in your browser
```

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env if needed (default settings should work for most cases)
```

### 5. (Optional) Download Additional Models

See `.env.example` for available model names.

---

## 🚀 Usage

### 📥 Ingest Papers

1. Place PDF papers in the `data/papers/` directory.
2. Run ingestion:

```bash
# Incremental (default) — only processes new/changed papers
python scripts/ingest_papers.py

# Force re-index all papers (ignores hashes)
python scripts/ingest_papers.py --force

# Show statistics only
python scripts/ingest_papers.py --stats

# Clean records for deleted files
python scripts/ingest_papers.py --clean
```

**What happens during ingestion:**
- PDFs are parsed using `docling` (structure-aware).
- Text is chunked using a parent-child hierarchy (parent chunks for summaries, child chunks for retrieval).
- Chunks are indexed into Qdrant vector database.
- Record metadata tracks processed files and hashes.

### 🌳 Build RAPTOR Summaries

RAPTOR builds hierarchical summary layers from child chunks, enabling better retrieval for deep research questions.

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
- Retrieves all child chunks from Qdrant.
- Groups chunks by `paper_id` and `section`.
- Iteratively builds summary nodes using the LLM, creating a hierarchy.
- Summary nodes are stored in Qdrant for RAPTOR-enhanced retrieval.
- Each level of summary provides increasingly abstract representations.

### 💬 Ask Questions

```bash
# Basic query (interactive or single-shot)
python src/main.py "What is the attention mechanism in transformers?"

# Interactive mode
python src/main.py --interactive

# With verbose logging
python src/main.py -v "Compare transformer architectures"
```

**Query flow:**
1. Semantic cache check (if HIT, returns cached answer immediately).
2. Query rewriting for better retrieval.
3. Adaptive routing selects the best strategy.
4. Retrieval, grading, and generation.
5. Faithfulness self-check.
6. Answer returned with citations.

### 📊 Run Evaluations

The evaluation harness reads a JSONL file of questions and collects metrics.

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
- Summary metrics: average latency, cache hit rate, faithfulness rate, citation rate, retrieval success rate.
- Strategy distribution across all questions.
- Per-question details: strategy, latency, cache hit/fail, faithfulness.
- Detailed JSON report saved to the specified output path.

**Example questions JSONL format:**

```json
{"id": "1", "question": "What is attention mechanism?", "type": "factoid", "must_cite": true}
{"id": "2", "question": "Compare transformer and LSTM architectures", "type": "comparative", "must_cite": true}
{"id": "3", "question": "How does backpropagation work?", "type": "explain", "must_cite": false}
```

---

## 📂 Project Structure

```
ai-research-assistant/
├── data/
│   ├── papers/              # Raw PDF papers
│   ├── processed/           # Processed chunks (JSON)
│   ├── eval/                # Evaluation questions and reports
│   └── cache/               # Semantic cache JSON
├── scripts/
│   ├── check_environment.py # Environment validation
│   ├── ingest_papers.py     # Incremental paper ingestion pipeline
│   ├── build_raptor.py      # RAPTOR hierarchical summary builder
│   └── run_eval.py          # Evaluation harness
├── src/
│   ├── main.py              # CLI entry point
│   ├── core/
│   │   ├── config.py        # Configuration management
│   │   ├── llm.py           # Ollama LLM client
│   │   └── embeddings.py    # Ollama embeddings client
│   ├── ingestion/
│   │   ├── pdf_parser.py    # PDF text extraction
│   │   ├── chunker.py       # Structure-aware chunking
│   │   ├── indexer.py       # Qdrant indexing
│   │   └── raptor_builder.py # RAPTOR summary building
│   ├── retrieval/
│   │   ├── hybrid_retriever.py   # Dense + BM25 search
│   │   ├── grader.py             # Relevance grading
│   │   ├── query_rewriter.py     # Query rewriting
│   │   ├── semantic_cache.py     # Caching
│   │   ├── corrective_retriever.py # CRAG
│   │   ├── multi_step_retriever.py # Multi-step retrieval
│   │   └── raptor_retriever.py     # RAPTOR retrieval
│   ├── routing/
│   │   └── adaptive_router.py # Query routing
│   ├── generation/
│   │   └── answer_generator.py # Cited answer generation
│   └── evaluation/
│       ├── metrics.py         # Metrics computation
│       └── evaluator.py       # Evaluation orchestration
├── requirements.txt          # Python dependencies
├── .env.example              # Environment configuration template
├── README.md                 # This file
└── ...
```

---

## ⚠️ Limitations & Notes on CPU Usage

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

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Report bugs** — Open an issue with a clear description and reproduction steps.
2. **Suggest features** — Open an issue explaining the use case and proposed solution.
3. **Submit pull requests** — Fork the repo, create a feature branch, and submit a PR with a clear description of changes.
4. **Improve documentation** — Fix typos, clarify instructions, or add examples.

Please ensure your code follows the existing style and includes tests where applicable.

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.




