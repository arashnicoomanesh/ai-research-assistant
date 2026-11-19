# Release Notes: v0.3.0

> **Current Version: v0.3.0** — "Reliable QA + Fixed Paths + Honest Docs + Hybrid + Baseline + Smoke Tests"

Date: 2026-11-19  
Commit: `e021430` (tagged as v0.3.0)

---

## What's New

This release marks the completion of a full cycle of **core reliability fixes**, **quality-of-life improvements**, **honest documentation**, and **ablation-ready evaluation infrastructure**. The system is now a stable, CPU-runnable RAG laboratory with explicit implementations and no framework dependencies.

---

## Major Sections

### P0: Core Integrity Fixes (Sprint B)
| ID | Fix | Description |
|----|-----|-------------|
| P0.3 | **Ingestion Integrity** | Stale vectors removed on re-ingest (`delete_paper_chunks` before upsert); batch upsert failures no longer leave silent success in record manager; PDF removal clears Qdrant vectors via `cleanup_deleted_files(indexer)` |
| P0.7 | **Temperature=0.0 Preservation** | `temperature=temperature or 0.7` → explicit `None` checks in `src/core/llm.py` (4 sites); `max_tokens=0` also preserved |

### Q2: Quality-of-Life (Sprint C)
| ID | Feature | Description |
|----|---------|-------------|
| Q2.1 | **Sludge Filter** | `src/retrieval/sludge_filter.py` drops table dumps / OCR noise / glued tokens via cheap heuristics (letter-ratio, digit-ratio, punctuation-ratio, min words); applied before context packing |
| Q2.2 | **Entity-Preserving Rewrite** | Query rewriter keeps named entities (Self-RAG, CRAG, BERT, GPT-3, FlashAttention, LoRA) via token-based extraction + post-process prepend |
| Q2.3 | **Parent Score Inheritance** | `expand_to_parents` assigns parent chunk the max score of triggering children so parents sort alongside high-scoring children |
| Q2.4 | **Context Cap** | `_limit_context` enforces hard `max_context_chunks` (default 5 via `CONTEXT_TOP_K`) before token budget; sorted by relevance score |

### P1: Extended Features
| ID | Feature | Description |
|----|---------|-------------|
| P1.1 | **Explicit Vanilla Baseline** | `FORCE_STRATEGY=baseline` env var selects dense retrieve → grade → generate (no CRAG rewrite loop); Reliable Mode default unchanged |
| P1.2 | **BM25 + RRF Hybrid** | `src/retrieval/bm25_index.py` builds persisted BM25 index; `HybridRetriever` fuses dense + BM25 via RRF (k=60); falls back to dense-only if index missing |
| P1.5 | **Graph Provenance** | Relations require both endpoints exist; relation IDs include `paper_id` + `chunk_id` (e.g., `source_type_target_paper_chunk`) so same entities from different papers don't overwrite |
| P1.6 | **Semantic Cache TTL + Invalidate** | `CACHE_TTL_SECONDS` (0 = no TTL); model identity in cache key; auto-clears on re-ingest / model change; `invalidate()` method wired in ingest pipeline |

### P3: Smoke Tests + License
| Item | Description |
|------|-------------|
| **LICENSE** | MIT license file added (matches README badge) |
| **tests/** | 67 offline tests: temperature preservation, RRF fusion, sludge filter, entity preservation, metrics, BM25 fallback, graph provenance, router experimental |
| **pytest -q** | Green without Ollama/Qdrant |

### P2: Ablation Documentation
| File | Purpose |
|------|---------|
| `docs/eval/ablations.md` | 3-config matrix: A) CRAG default, B) Baseline, C) Experimental; reproduce commands, comparison templates |
| `docs/eval/baseline.md` | Links to ablations.md; honest baseline notes |

---

## Non-Goals (Explicitly Not Done)

| Area | Status |
|------|--------|
| GraphRAG Stage B (communities/global summaries) | 🚫 Not implemented |
| Cross-encoder reranking | 🚫 Not implemented |
| Full RAPTOR clustering tree | 🚫 RAPTOR-inspired only (limited hierarchy) |
| BM25 as default / required | 🚫 Optional; falls back to dense-only |
| SOTA benchmark claims | 🚫 Honest: CPU lab, small corpus, not a leaderboard |

---

## Version Badge

README now includes: `[![Version](https://img.shields.io/badge/Version-v0.3.0-blue)](https://github.com/arashnicoomanesh/ai-research-assistant/releases/tag/v0.3.0)`

---

## Files Changed in This Release

**Code (7 files):**
- `src/retrieval/semantic_cache.py` — TTL, model identity, invalidate
- `src/core/config.py` — CacheConfig, paths
- `scripts/ingest_papers.py` — cache.invalidate on ingest, --rebuild-bm25
- `src/main.py` — cache identity, baseline strategy
- `src/retrieval/bm25_index.py` — new module
- `src/retrieval/hybrid_retriever.py` — RRF fusion
- `src/routing/adaptive_router.py` — comparison → multi_step heuristic

**Tests (7 new files):**
- `tests/test_llm_temperature.py`
- `tests/test_rrf.py`
- `tests/test_sludge_filter.py`
- `tests/test_query_rewriter_entities.py`
- `tests/test_metrics.py`
- `tests/test_bm25_fallback.py`
- `tests/test_semantic_cache.py`
- `tests/test_graph_provenance.py`
- `tests/test_router_experimental.py`

**Docs (4 files):**
- `LICENSE` — MIT
- `docs/eval/baseline.md` — baseline notes + ablation link
- `docs/eval/ablations.md` — matrix with commands/templates
- `README.md` — version badge, honest positioning

**Config:**
- `.env.example` — CACHE_TTL_SECONDS, HYBRID_ENABLED, BM25_INDEX_DIR

---

## Migration Notes

- **Default behavior unchanged**: `RELIABLE_MODE=true`, `EXPERIMENTAL_ROUTING=false` still forces `single_shot` (CRAG)
- **To try baseline**: `FORCE_STRATEGY=baseline python scripts/run_eval.py --limit 5`
- **To enable hybrid**: `HYBRID_ENABLED=true` (default) + run `python scripts/ingest_papers.py --rebuild-bm25`
- **Cache TTL**: set `CACHE_TTL_SECONDS=3600` for 1-hour expiry

---

## Verification

```bash
# All tests pass offline
pytest -q
# 67 passed in ~5s

# Smoke eval
python scripts/run_eval.py --limit 3
```

---

## License

MIT — see [LICENSE](LICENSE)

---

*Tagged at commit `e021430` as `v0.3.0`*