# Ablation Matrix: Baseline vs Reliable CRAG

> **Purpose**: Compare baseline (vanilla retrieve→generate) vs reliable default (CRAG) on the **same question set**. Only the strategy flag changes; everything else (corpus, models, cache, index) stays identical.

> **Disclaimer**: This is a **lab ablation** on a 6-paper CPU corpus with `qwen2.5-coder:3b`. Not a benchmark, not SOTA. Results are hardware/corpus dependent.

---

## Reproduce Matrix

Run each configuration on the **same question subset** (e.g., `--limit 5` from `data/eval/questions.jsonl`). Save each report to a distinct file.

| Config | Description | Env Flags | Command |
|--------|-------------|-----------|---------|
| **A** | Reliable default (CRAG/single_shot) | `RELIABLE_MODE=true` `EXPERIMENTAL_ROUTING=false` | `python scripts/run_eval.py --limit 5 --output data/eval/reports/ablation_crag.json` |
| **B** | Vanilla baseline (no CRAG) | `FORCE_STRATEGY=baseline` | `FORCE_STRATEGY=baseline python scripts/run_eval.py --limit 5 --output data/eval/reports/ablation_baseline.json` |
| **C** (optional) | Experimental routing (multi_step/raptor/graph) | `RELIABLE_MODE=false` `EXPERIMENTAL_ROUTING=true` | `RELIABLE_MODE=false EXPERIMENTAL_ROUTING=true python scripts/run_eval.py --limit 5 --output data/eval/reports/ablation_experimental.json` |

> **Note**: Config A uses defaults (`RELIABLE_MODE=true`, `EXPERIMENTAL_ROUTING=false`). Config B forces the vanilla baseline. Config C enables experimental strategies.

---

## How to Run

```bash
# 1. Ensure services running (Ollama + Qdrant)
# 2. Ingest papers if not done
python scripts/ingest_papers.py

# 3. Run ablation A: Reliable default (CRAG)
python scripts/run_eval.py --limit 5 --output data/eval/reports/ablation_crag.json

# 4. Run ablation B: Vanilla baseline
FORCE_STRATEGY=baseline python scripts/run_eval.py --limit 5 --output data/eval/reports/ablation_baseline.json

# 5. Optional: Run ablation C: Experimental routing
RELIABLE_MODE=false EXPERIMENTAL_ROUTING=true python scripts/run_eval.py --limit 5 --output data/eval/reports/ablation_experimental.json
```

---

## Expected Report Structure

Each run produces a JSON report at the `--output` path with this structure:

```json
{
  "questions_evaluated": 5,
  "summary": {
    "total_questions": 5,
    "avg_latency_seconds": 400.2,
    "cache_hit_rate": 0.0,
    "faithfulness_rate": 0.6,
    "citation_rate": 1.0,
    "retrieval_success_rate": 1.0,
    "strategy_distribution": {
      "crag_rewrite_merge": 0.6,
      "crag_refine": 0.4
    }
  },
  "detailed_results": {
    "q001": {"strategy": "crag_rewrite_merge", "latency": 350.2, "faithful": true, "citation_present": true, "retrieval_success": true, ...},
    ...
  }
}
```

---

## Comparison Template (fill from actual JSON outputs)

### Summary Comparison

| Metric | A: CRAG (default) | B: Baseline | C: Experimental |
|--------|-------------------|-------------|-----------------|
| Avg latency (s) | | | |
| Faithfulness rate | | | |
| Citation rate | | | |
| Retrieval success | | | |
| Cache hit rate | | | |
| Total questions | 5 | 5 | 5 |

### Strategy Distribution

| Strategy | A (CRAG) | B (Baseline) | C (Experimental) |
|----------|----------|--------------|------------------|
| `crag_rewrite_merge` | | N/A | |
| `crag_refine` | | N/A | |
| `baseline` | N/A | | |
| `multi_step` | N/A | N/A | |
| `crag_raptor` | N/A | N/A | |
| `graph_rag` | N/A | N/A | |
| `no_retrieval` | | | |

### Per-Question Detail (example template)

| qid | Question | A: Strategy | A: Latency | A: Faithful | B: Strategy | B: Latency | B: Faithful |
|-----|----------|-------------|------------|-------------|-------------|------------|-------------|
| q001 | What is attention? | | | | | | |
| q002 | Compare BERT vs GPT-2 | | | | | | |
| ... | | | | | | | |

---

## Key Differences to Expect

| Aspect | A: CRAG (Reliable) | B: Baseline (Vanilla) |
|--------|--------------------|-----------------------|
| **Latency** | Higher (CRAG rewrite loop) | Lower (single dense retrieve) |
| **Retrieval Quality** | Better on ambiguous queries | May miss on low-recall queries |
| **Strategy Label** | `crag_rewrite_merge` / `crag_refine` | `baseline` |
| **Failure Mode** | CRAG fallback to rewrite | Single-shot only |

---

## Disclaimer

> **CPU lab, small corpus, not a leaderboard.**
>
> - Corpus: 6 AI papers, dense-only retrieval (Qdrant HNSW)
> - Model: `qwen2.5-coder:3b` + `nomic-embed-text:v1.5` on CPU
> - Cache: max-entries eviction, no TTL
> - GraphRAG: Stage A only (requires `EXPERIMENTAL_ROUTING=true`)
> - RAPTOR: inspired / limited hierarchy (not full clustering tree)
> - Faithfulness self-check: off in CPU-Safe mode
>
> Do not cite these numbers as SOTA. They are internal lab baselines for controlled ablation only.

---

## Files Referenced

- Eval runner: `scripts/run_eval.py`
- Questions: `data/eval/questions.jsonl`
- Reports: `data/eval/reports/ablation_*.json`
- Baseline notes: `docs/eval/baseline.md`
- Config defaults: `src/core/config.py`