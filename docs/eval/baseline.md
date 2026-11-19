# Reliable-Path Baseline Evaluation Notes

> **Disclaimer**: This is a **lab baseline** for internal ablation comparisons, **not a benchmark claim**. Results are corpus-dependent, CPU-bound, and tied to the specific model/hardware below. Do not cite as SOTA.

---

## Run Metadata

| Field | Value |
|-------|-------|
| Date | 2026-11-19 |
| Commit | `7e8a045` (and `81c6e68` for P1.1 baseline strategy) |
| Branch | `main` |
| Hardware | CPU-only (local), Ollama on localhost |
| LLM | `qwen2.5-coder:3b` |
| Embeddings | `nomic-embed-text:v1.5` |
| Corpus | 6 AI research papers (ingested under `data/papers/`) |
| Qdrant | Local Docker, collection `ai_papers` |

---

## Reproduce (Reliable Default Path)

Default flags (set in `.env` or shell):

```bash
export CPU_SAFE_MODE=true
export RELIABLE_MODE=true
export EXPERIMENTAL_ROUTING=false
# (these are the defaults in src/core/config.py)
```

Run the existing eval harness (uses questions under `data/eval/questions.jsonl`):

```bash
# Quick run (first 3 questions from the suite)
python scripts/run_eval.py --limit 3

# Full suite
python scripts/run_eval.py

# Optional: force the vanilla baseline strategy for ablation
FORCE_STRATEGY=baseline python scripts/run_eval.py --limit 3
```

Output goes to `data/eval/reports/evaluation_report.json`.

---

## Last Known Run (from `data/eval/reports/evaluation_report.json`)

Run date: 2026-11-19 (same as commit above)  
Questions evaluated: 3 (subset for CPU time)

### Summary

| Metric | Value |
|--------|-------|
| Total questions | 3 |
| Avg latency (seconds) | 431.5 |
| Cache hit rate | 0.0 |
| Faithfulness rate | 66.7% |
| Citation rate | 100% |
| Retrieval success rate | 100% |

### Strategy Distribution

| Strategy | Share |
|----------|-------|
| `crag_rewrite_merge` | 66.7% |
| `crag_refine` | 33.3% |

> Note: Under Reliable Mode (`RELIABLE_MODE=true`, `EXPERIMENTAL_ROUTING=false`) the router forces `single_shot` (which runs CRAG) for all retrieval-requiring queries. The strategy labels `crag_rewrite_merge` / `crag_refine` come from the corrective retriever's post-hoc correction type.

### Per-Question Detail

| qid | Question | Type | Strategy | Latency (s) | Faithful | Citation | Retrieval OK |
|-----|----------|------|----------|-------------|----------|----------|--------------|
| q001 | What is attention mechanism in transformer neural networks? | no_retrieval | crag_rewrite_merge | 360.5 | ✅ | ✅ | ✅ |
| q002 | What method did the BERT paper propose for masked language modeling? | single_shot | crag_refine | 422.9 | ❌ (generation timeout) | ✅ | ✅ |
| q003 | Compare the architecture of BERT and GPT-2 models | multi_step | crag_rewrite_merge | 511.1 | ✅ | ✅ | ✅ |

---

## Vanilla Baseline Ablation (P1.1)

The codebase now includes an explicit `baseline` strategy (`FORCE_STRATEGY=baseline`) that does **dense retrieve → optional grade → generate** without the CRAG rewrite loop. Use this for controlled ablation:

```bash
# Compare reliable default (CRAG) vs vanilla baseline
python scripts/run_eval.py --limit 3                    # default (CRAG)
FORCE_STRATEGY=baseline python scripts/run_eval.py --limit 3  # vanilla
```

Expected differences:
- Baseline skips CRAG evaluation/rewrite → lower latency
- Baseline may have lower retrieval quality on ambiguous queries
- Both paths share: sludge filter (Q2.1), parent expansion with score inheritance (Q2.3), context cap (Q2.4)

---

## Evaluation Harness Notes

- `scripts/run_eval.py` reads `data/eval/questions.jsonl` (JSONL with fields: `id`, `question`, `type`, `must_cite`).
- The evaluator (`src/evaluation/evaluator.py`) runs each question through `ResearchAssistant.ask()`, records latency, strategy, faithfulness (self-check), citation presence, and retrieval success.
- Reports are written to `data/eval/reports/evaluation_report.json`.
- CPU timeouts on 3B model are common; the generator falls back to extractive summaries when LLM generation times out (see q002 above).

---

## Caveats & Limitations

1. **Tiny sample**: 3 questions only — not statistically meaningful.
2. **CPU only**: Latencies ~6–8 min/question on 3B model; GPU or smaller models would differ.
3. **Corpus bias**: 6 papers, AI/ML domain; results don't generalize.
4. **No BM25/RRF**: Retrieval is dense-only (Qdrant HNSW).
5. **No TTL cache**: Semantic cache evicts by max-entries only.
6. **GraphRAG Stage A only**: `graph_rag` strategy requires `EXPERIMENTAL_ROUTING=true`.
7. **RAPTOR-inspired**: Not full RAPTOR clustering tree.
8. **Faithfulness self-check**: Off by default in CPU-Safe Mode; when on, uses same 3B model (circular).

---

## Files Referenced

- Config defaults: `src/core/config.py`
- Reliable mode logic: `src/main.py` (around line 317)
- Baseline strategy: `src/main.py` (search `FORCE_STRATEGY=baseline`)
- Eval runner: `scripts/run_eval.py`
- Questions: `data/eval/questions.jsonl`
- Reports: `data/eval/reports/evaluation_report.json`