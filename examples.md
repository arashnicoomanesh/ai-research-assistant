# Examples

Practical examples and workflows for the AI Research Assistant.

---

## Quick Start: Complete Workflow

```bash
# 1. Start services
ollama serve &
docker run -d -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant

# 2. Pull models
ollama pull qwen2.5-coder:3b
ollama pull nomic-embed-text:v1.5

# 3. Configure
cp .env.example .env

# 4. Add papers
cp ~/papers/*.pdf data/papers/

# 5. Ingest
python scripts/ingest_papers.py

# 6. Build RAPTOR summaries
python scripts/build_raptor.py

# 7. Ask questions
python src/main.py "What is the attention mechanism?"

# 8. Or interactive mode
python src/main.py --interactive
```

---

## Example Questions by Strategy

### No Retrieval (General Knowledge)

```bash
python src/main.py "What is machine learning?"
python src/main.py "Define neural network"
python src/main.py "What year was the transformer paper published?"
```

### Single-Shot / CRAG (Paper-Specific Facts) — Default in Reliable Mode

```bash
python src/main.py "What is the attention mechanism in the Attention Is All You Need paper?"
python src/main.py "How does BERT differ from GPT?"
python src/main.py "What datasets were used in the Llama 2 paper?"
```

### Multi-Step (Complex/Comparative) — Requires EXPERIMENTAL_ROUTING=true

```bash
EXPERIMENTAL_ROUTING=true python src/main.py "Compare transformer and RNN architectures for sequence modeling"
EXPERIMENTAL_ROUTING=true python src/main.py "What are the tradeoffs between dense and sparse attention?"
EXPERIMENTAL_ROUTING=true python src/main.py "Summarize the evolution from BERT to GPT-4"
```

### RAPTOR-inspired (Deep Research / Survey Questions) — Requires EXPERIMENTAL_ROUTING=true

```bash
EXPERIMENTAL_ROUTING=true python src/main.py "What are the main approaches to efficient transformers?"
EXPERIMENTAL_ROUTING=true python src/main.py "Survey the literature on retrieval-augmented generation"
EXPERIMENTAL_ROUTING=true python src/main.py "How has model scaling evolved from 2017 to 2024?"
```

---

## Ingestion Examples

```bash
# Incremental (default) - only new/changed
python scripts/ingest_papers.py

# Force re-index everything
python scripts/ingest_papers.py --force

# Show stats without ingesting
python scripts/ingest_papers.py --stats

# Clean deleted file records
python scripts/ingest_papers.py --clean

# Custom papers directory
python scripts/ingest_papers.py --papers-dir /path/to/papers

# Verbose logging
python scripts/ingest_papers.py -v
```

---

## RAPTOR Building Examples

```bash
# Build all summaries
python scripts/build_raptor.py

# Build for one paper
python scripts/build_raptor.py --paper attention_is_all_you_need

# Quick test - limit to 5 groups
python scripts/build_raptor.py --limit 5

# Check existing summaries only
python scripts/build_raptor.py --check

# One paper with limit
python scripts/build_raptor.py --paper attention_is_all_you_need --limit 3
```

---

## Query Examples

```bash
# Single question
python src/main.py "How does self-attention work?"

# Interactive mode
python src/main.py --interactive

# Verbose logging (shows routing, retrieval, cache details)
python src/main.py -v "What is LoRA?"

# Cache statistics
python src/main.py --cache-stats

# Clear cache
python src/main.py --clear-cache
```

---

## Evaluation Examples

```bash
# Run all evaluation questions
python scripts/run_eval.py

# Quick test with 5 questions
python scripts/run_eval.py --limit 5

# Custom questions file + save report
python scripts/run_eval.py data/eval/my_questions.jsonl --output results.json

# Run with specific limit
python scripts/run_eval.py --limit 10
```

---

## Environment Check

```bash
# Validate entire environment
python scripts/check_environment.py
```

Expected output:
```
PASS Ollama reachable
PASS Ollama models available
PASS Qdrant reachable
PASS Data directories
```

---

## Configuration Examples

### Using Different Models

Edit `.env`:

```bash
# Faster, smaller models
OLLAMA_LLM_MODEL=phi3:3.8b
OLLAMA_EMBEDDING_MODEL=all-minilm:33m

# Higher quality (needs more RAM)
OLLAMA_LLM_MODEL=mistral:7b
OLLAMA_EMBEDDING_MODEL=mxbai-embed-large:335m
```

### Custom Retrieval Settings

```bash
# More results for complex queries
TOP_K_DENSE=20
TOP_K_FINAL=10

# Larger chunks for better context
CHUNK_SIZE=1024
CHUNK_OVERLAP=100
```

---

## Troubleshooting

### Ollama Not Reachable

```bash
# Check if running
curl http://localhost:11434/api/tags

# Start if not
ollama serve
```

### Models Not Found

```bash
ollama pull qwen2.5-coder:3b
ollama pull nomic-embed-text:v1.5
```

### Qdrant Collection Missing

```bash
# Will be created automatically on first ingestion
python scripts/ingest_papers.py
```

### No Papers Found

```bash
# Add PDFs to data/papers/
ls data/papers/
# Should show .pdf files
```

### Slow Performance

- Use smaller models (phi3:3.8b, all-minilm:33m)
- Reduce TOP_K values
- Limit RAPTOR groups with `--limit`
- Ensure SSD storage for Qdrant

---

## Advanced: Custom Evaluation Questions

Create `data/eval/questions.jsonl`:

```jsonl
{"id": "q1", "question": "What is self-attention?", "type": "factoid", "must_cite": true}
{"id": "q2", "question": "Compare attention mechanisms in recent papers", "type": "comparative", "must_cite": true}
{"id": "q3", "question": "How does positional encoding work?", "type": "explain", "must_cite": true}
{"id": "q4", "question": "What are the limitations of transformers?", "type": "survey", "must_cite": true}
{"id": "q5", "question": "Explain the scaling laws paper findings", "type": "detail", "must_cite": true}
```

Run:
```bash
python scripts/run_eval.py data/eval/questions.jsonl --output eval_results.json
```

---

## Integration with Other Tools

### Using as a Module

```python
from src.main import ResearchAssistant

assistant = ResearchAssistant()
answer = assistant.ask("Your question here")
print(answer.answer)
print(answer.sources)
```

### Programmatic Ingestion

```python
from scripts.ingest_papers import IngestionPipeline
from pathlib import Path

pipeline = IngestionPipeline(papers_dir=Path("my_papers"))
stats = pipeline.run()
print(f"Indexed {stats['processed']} papers, {stats['total_chunks']} chunks")
```

---

## Performance Tips

| Setting | Fast CPU | Balanced | Best Quality |
|---------|----------|----------|--------------|
| LLM | phi3:3.8b | qwen2.5-coder:3b | mistral:7b |
| Embeddings | all-minilm:33m | nomic-embed-text:v1.5 | mxbai-embed-large |
| CHUNK_SIZE | 256 | 512 | 1024 |
| TOP_K_DENSE | 5 | 10 | 20 |
| RAPTOR | --limit 10 | default | --limit 100 |

---

## Common Patterns

### Research Paper Analysis
```bash
# 1. Ingest a batch of papers on a topic
python scripts/ingest_papers.py

# 2. Build RAPTOR-inspired summaries for deep retrieval
python scripts/build_raptor.py

# 3. Ask survey-style questions (requires EXPERIMENTAL_ROUTING=true)
EXPERIMENTAL_ROUTING=true python src/main.py "What are the main approaches to X?"
EXPERIMENTAL_ROUTING=true python src/main.py "Compare methods A, B, and C for Y"
```

### Fact Checking
```bash
# Direct questions about specific papers
python src.main.py "What dataset did paper X use?"
python src.main.py "What was the main finding of paper Y?"
```

### Iterative Research
```bash
# Start broad
python src.main.py "Overview of efficient transformers"

# Drill down
python src/main.py "Details on FlashAttention implementation"

# Compare
python src.main.py "FlashAttention vs xFormers vs memory-efficient attention"
```

---

## Next Steps

- Add your papers to `data/papers/`
- Run `python scripts/ingest_papers.py`
- Run `python scripts/build_raptor.py`
- Start asking questions!