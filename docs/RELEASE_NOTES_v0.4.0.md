# Release Notes: v0.4.0

> **Current Version: v0.4.0** — "Hybrid Retrieval + RAPTOR-Inspired Summaries + Cache TTL + Graph Provenance + Ablation Docs"

Date: 2026-11-20  
Commit: `cf155d9` (tagged as v0.4.0)

---

## What's New

This release delivers the remaining P1 fidelity items beyond v0.3.0: **BM25+RRF hybrid retrieval**, **optional cross-encoder reranking (default OFF)**, **RAPTOR-inspired query-similar summaries with bounded detail attach**, **GraphRAG provenance IDs**, **semantic cache TTL + model/ingest invalidation**, **experimental multi-step routing**, and **ablation documentation**. All implemented without frameworks; CPU-runnable defaults unchanged.

---

## Major Sections Since v0.3.0

### P1.2: BM25 + RRF Hybrid Retrieval (Completed)
| Item | Description |
|------|-------------|
| `src/retrieval/bm25_index.py` | Persisted BM25 index built from child chunks; incremental updates via `--rebuild-bm25` |
| `src/retrieval/hybrid_retriever.py` | Dense (Qdrant) + BM25 fused via Reciprocal Rank Fusion (k=60); dense-only fallback if BM25 index missing |
| `HYBRID_ENABLED=true` | Default ON; set `false` to run dense-only |

### P1.3: Optional Cross-Encoder Rerank (Default OFF)
| Item | Description |
|------|-------------|
| `src/retrieval/reranker.py` | Cross-encoder reranking (sentence-transformers) behind `RERANK_ENABLED` flag |
| Default | `RERANK_ENABLED=false` — not required, not on by default, not for weak CPU |
| Config | `RERANK_TOP_K=20`, `RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2` |

### P1.4: RAPTOR-Inspired Query-Similar Summaries + Detail Budget
| Item | Description |
|------|-------------|
| `src/retrieval/raptor_retriever.py` | Summary nodes retrieved by **query embedding similarity** (vector search), not static level sort |
| Detail attach | Each selected summary gets ≤ `DEFAULT_K_DETAIL_PER_SUMMARY=3` child chunks from its `source_chunk_ids` |
| Total budget | `MAX_DETAIL_CHUNKS_TOTAL=10` across all summaries; context stays bounded |
| Generator `text` field | Populated on both summary (`node.summary`) and detail chunks for cited generation |
| Tests | `tests/test_raptor_inspired.py` — 11 offline mock tests (similarity ranking, budgets, text field) |
| Scope | **RAPTOR-inspired**: limited hierarchy (section=1, paper=2); no recursive clustering tree |

### P1.5: GraphRAG Endpoint + Provenance IDs
| Item | Description |
|------|-------------|
| `src/graph/graph_retriever.py` | Graph retrieval path: entity linking → 1-hop traversal → triple collection → supporting chunk fusion |
| Provenance | Relation IDs embed `paper_id` + `chunk_id` (e.g., `source_type_target_paper_chunk`) so same entities from different papers don't overwrite |
| Edge validity | Relations require both endpoints to exist; no dangling edges |
| Tests | `tests/test_graph_provenance.py` — verifies provenance scheme |

### P1.6: Semantic Cache TTL + Ingest/Model Invalidate
| Item | Description |
|------|-------------|
| `src/retrieval/semantic_cache.py` | `CACHE_TTL_SECONDS` (0 = no expiry); model identity (LLM + embedding) in cache key |
| Auto-invalidate | Cache clears on re-ingest (`scripts/ingest_papers.py`) or model change (detected on load) |
| Manual invalidate | `cache.invalidate(reason="...")` wired in ingest pipeline |
| Tests | `tests/test_semantic_cache.py` — TTL, model mismatch, invalidate persistence |

### Experimental: Comparison → Multi-Step Routing
| Item | Description |
|------|-------------|
| `src/routing/adaptive_router.py` | Heuristic: "compare X vs Y" / "difference between" / "versus" routes to `multi_step` when `EXPERIMENTAL_ROUTING=true` |
| Default | Reliable Mode (`RELIABLE_MODE=true`) still forces `single_shot` (CRAG); experimental opt-in only |

### P2: Ablation Documentation
| File | Purpose |
|------|---------|
| `docs/eval/ablations.md` | 3-config matrix: A) CRAG default, B) Baseline (dense-only), C) Experimental; reproduce commands, comparison templates |
| `docs/eval/baseline.md` | Honest baseline notes + link to ablations |

### P3: Smoke Test Growth
| Item | Description |
|------|-------------|
| Total tests | **86** offline tests (was 67 in v0.3.0) |
| New test files | `tests/test_raptor_inspired.py` (11), `tests/test_rerank.py` (5) |
| `pytest -q` | Green without Ollama/Qdrant (~6s) |

---

## Non-Goals (Explicitly Not Done)

| Area | Status |
|------|--------|
| GraphRAG Stage B (communities/global summaries) | 🚫 Not implemented |
| Full RAPTOR recursive clustering tree | 🚫 RAPTOR-inspired only (limited 2-level hierarchy) |
| Cross-encoder reranking as default | 🚫 Optional, OFF by default (`RERANK_ENABLED=false`) |
| BM25 as required / no fallback | 🚫 Falls back to dense-only gracefully |
| SOTA benchmark claims | 🚫 Honest: CPU lab, small corpus, not a leaderboard |

---

## Version Badge

README updated: `[![Version](https://img.shields.io/badge/Version-v0.4.0-blue)](https://github.com/arashnicoomanesh/ai-research-assistant/releases/tag/v0.4.0)`

---

## Files Changed in This Release

**Code (4 files):**
- `src/retrieval/raptor_retriever.py` — query-similar summary retrieval, bounded detail attach, config constants
- `src/retrieval/reranker.py` — cross-encoder reranking (optional, default OFF)
- `src/routing/adaptive_router.py` — comparison → multi_step heuristic
- `src/retrieval/semantic_cache.py` — TTL, model identity, invalidate (from v0.3.0, stabilized)

**Tests (2 new files):**
- `tests/test_raptor_inspired.py` — 11 offline tests
- `tests/test_rerank.py` — 5 offline tests

**Docs (3 files):**
- `docs/RELEASE_NOTES_v0.4.0.md` — this file
- `docs/eval/ablations.md` — ablation matrix + commands
- `docs/eval/baseline.md` — baseline notes + ablation link
- `README.md` — version badge v0.4.0, RAPTOR-inspired clarifying note

**Config:**
- `.env.example` — `RERANK_ENABLED`, `RERANK_TOP_K`, `RERANK_MODEL`

---

## Migration Notes

- **Default behavior unchanged**: `RELIABLE_MODE=true`, `EXPERIMENTAL_ROUTING=false` still forces `single_shot` (CRAG)
- **To try baseline**: `FORCE_STRATEGY=baseline python scripts/run_eval.py --limit 5`
- **To enable hybrid**: `HYBRID_ENABLED=true` (default) + `python scripts/ingest_papers.py --rebuild-bm25`
- **To enable rerank**: `RERANK_ENABLED=true` (requires `sentence-transformers`, not for weak CPU)
- **Cache TTL**: set `CACHE_TTL_SECONDS=3600` for 1-hour expiry
- **To enable experimental routing**: `EXPERIMENTAL_ROUTING=true` (unlocks `multi_step`, `crag_raptor`, `graph_rag`)

---

## Verification

```bash
# All tests pass offline
pytest -q
# 86 passed in ~6s

# Smoke eval
python scripts/run_eval.py --limit 3
```

---

## License

MIT — see [LICENSE](LICENSE)

---

*Tagged at commit `cf155d9` as `v0.4.0`*