#!/usr/bin/env python3
"""
AI Research Assistant - Main CLI Entry Point (Phase 4: GraphRAG + RAPTOR + CRAG).

Orchestrates the full pipeline with GraphRAG, Corrective RAG, and RAPTOR-enhanced retrieval:
1. Check Semantic Cache
2. If miss -> Rewrite query
3. Route the query (adaptive router with GraphRAG + RAPTOR support)
4. If retrieval needed:
   - no_retrieval: Answer from model knowledge
   - baseline: Dense retrieve + generate (no corrective loop)
   - single_shot: Corrective RAG (CRAG)
   - multi_step: Decompose + retrieve per sub-question
   - crag_raptor: RAPTOR-enhanced hierarchical retrieval
   - graph_rag: Knowledge graph retrieval for relational queries
5. Answer generation + self-check
6. Store in cache
7. Print result cleanly
"""
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import config
from src.core.llm import get_llm
from src.core.embeddings import get_embeddings
from src.retrieval.hybrid_retriever import get_retriever
from src.retrieval.grader import get_grader
from src.retrieval.query_rewriter import get_rewriter
from src.retrieval.semantic_cache import get_cache
from src.retrieval.corrective_retriever import get_corrective_retriever, CorrectiveRetrievalResult
from src.retrieval.multi_step_retriever import get_multi_step_retriever, MultiStepRetriever
from src.retrieval.raptor_retriever import get_raptor_retriever, RaptorRetrievalResult
from src.graph import get_graph_retriever, GraphEvidence
from src.routing.adaptive_router import get_router, RoutingDecision
from src.generation.answer_generator import get_generator, GeneratedAnswer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


def _get_parser():
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        usage="python src/main.py [options] [question]",
        description="AI Research Assistant - Query your paper collection with GraphRAG + CRAG + RAPTOR pipeline.\n\n"
                    "Supports adaptive routing, query rewriting, corrective RAG, multi-step retrieval,\n"
                    "RAPTOR-enhanced summaries, knowledge graph retrieval, semantic caching,\n"
                    "and faithfulness self-checks.\n\n"
                    "If no question is provided, interactive mode is started.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/main.py "What is attention mechanism?"
    Ask a question against your paper collection.

  python src/main.py "Compare transformer architectures in recent papers"
    Complex query: will use multi-step or RAPTOR strategy.

  python src/main.py "How is Self-RAG related to CRAG?"
    Relational query: will use GraphRAG strategy.

  python src/main.py --interactive
    Start interactive question loop.

  python src/main.py --cache-stats
    Show semantic cache statistics and exit.

  python src/main.py --clear-cache
    Clear semantic cache and exit.

  python scripts/build_raptor.py --all
    Build RAPTOR summaries for all papers.

  python scripts/build_raptor.py --paper ai_papers --limit 5
    Build RAPTOR summaries for one paper, limited to 5 groups.

  python scripts/build_graph.py --all
    Build GraphRAG knowledge graph for all papers.

  python scripts/build_graph.py --stats
    Show graph statistics.

  scripts/check_environment.py
    Validate that Ollama, Qdrant, and data directories are properly configured.
"""
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Question to ask (omit for interactive mode)"
    )
    parser.add_argument(
        "-q", "--question",
        dest="query",
        help="Question to ask (alternative to positional argument)"
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Run in interactive mode (question loop)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--cache-stats",
        action="store_true",
        help="Show cache statistics and exit"
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear semantic cache and exit"
    )
    parser.add_argument(
        "--cache-path",
        default="data/cache/semantic_cache.json",
        help="Path to semantic cache file (default: data/cache/semantic_cache.json)"
    )
    return parser


class ResearchAssistant:
    """Main orchestrator for the CRAG + RAPTOR pipeline."""

    def __init__(self, verbose: bool = False, cache_path: str = "data/cache/semantic_cache.json"):
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        # Initialize components
        self.llm = get_llm()
        self.embeddings = get_embeddings()
        self.retriever = get_retriever(top_k=config.cpu_safe.retrieve_top_k)
        self.grader = get_grader()
        self.rewriter = get_rewriter()
        self.router = get_router()
        self.corrective_retriever = get_corrective_retriever(
            retriever=self.retriever,
            grader=self.grader,
            rewriter=self.rewriter,
        )
        self.multi_step_retriever = get_multi_step_retriever(
            corrective_retriever=self.corrective_retriever,
        )
        self.raptor_retriever = get_raptor_retriever(
            embeddings=self.embeddings,
        )
        self.graph_retriever = get_graph_retriever(
            embeddings=self.embeddings,
            retriever=self.retriever,
        )
        self.generator = get_generator(
            max_context_chunks=config.cpu_safe.context_top_k,
            do_self_check=config.cpu_safe.enable_self_check,
        )
        self.cache = get_cache(
            embeddings=self.embeddings,
            similarity_threshold=config.cache.similarity_threshold,
            max_entries=config.cache.max_entries,
            cache_path=cache_path,
            ttl_seconds=config.cache.ttl_seconds,
            llm_model=config.ollama.llm_model,
            embedding_model=config.ollama.embedding_model,
        )

        # Check services
        self._check_services()

    def _check_services(self) -> None:
        """Verify Ollama and Qdrant are available."""
        # Check Ollama
        if not self.llm.is_available():
            logger.warning("Ollama server not reachable at %s", config.ollama.host)
        else:
            logger.info("Ollama: OK (model: %s)", config.ollama.llm_model)

        # Check embeddings
        if not self.embeddings.is_available():
            logger.warning("Embedding model not available: %s", config.ollama.embedding_model)
        else:
            logger.info("Embeddings: OK (model: %s)", config.ollama.embedding_model)

        # Cache stats
        stats = self.cache.stats()
        logger.info("Cache: %d entries, %d total hits", stats["entries"], stats["total_hits"])

    @staticmethod
    def _to_dict(chunk: Any) -> Dict[str, Any]:
        """Convert chunk to dict (handles both dict and dataclass)."""
        if isinstance(chunk, dict):
            return chunk
        # Assume dataclass with attributes (GradedChunk)
        return {
            "chunk_id": getattr(chunk, "chunk_id", None),
            "paper_id": getattr(chunk, "paper_id", ""),
            "title": getattr(chunk, "title", ""),
            "section": getattr(chunk, "section", ""),
            "chunk_type": getattr(chunk, "chunk_type", ""),
            "text": getattr(chunk, "text", ""),
            "parent_id": getattr(chunk, "parent_id", None),
            "chunk_index": getattr(chunk, "chunk_index", 0),
            "token_count": getattr(chunk, "token_count", 0),
            "relevance_score": getattr(chunk, "relevance_score", 0),
            "relevance_label": getattr(chunk, "relevance_label", ""),
            "reasoning": getattr(chunk, "reasoning", ""),
        }

    def _graph_evidence_to_chunks(self, evidence: GraphEvidence) -> List[Dict[str, Any]]:
        """
        Convert GraphEvidence to chunk format for answer generator.

        Builds a strict context block that the generator must grounded in.
        Never invents relations not present in the graph.

        Args:
            evidence: GraphEvidence from graph retriever

        Returns:
            List of chunk dicts compatible with generator
        """
        chunks = []

        # Build strict graph context block
        graph_lines = ["[Graph Evidence]"]

        if evidence.entities:
            entity_names = [e.name for e in evidence.entities]
            graph_lines.append(f"- Entities: {', '.join(entity_names)}")

        if evidence.relations:
            rel_lines = []
            for r in evidence.relations:
                source = self.graph_retriever.graph_store.get_entity(r.source_id)
                target = self.graph_retriever.graph_store.get_entity(r.target_id)
                s_name = source.name if source else r.source_id
                t_name = target.name if target else r.target_id
                rel_lines.append(f"- {s_name} -[{r.relation_type.value}]-> {t_name}: {r.evidence}")
            graph_lines.append("Relations: " + "; ".join(rel_lines))
        else:
            graph_lines.append("Relations: none found")

        # Explicit note if no direct relation found between query entities
        linked_ids = evidence.metadata.get("linked_entity_ids", [])
        if len(linked_ids) >= 2:
            has_direct_edge = evidence.metadata.get("direct_edges", 0) > 0
            if not has_direct_edge:
                graph_lines.append(
                    "*No direct relation edge was found in the local graph. "
                    "The above shows available entities and their relations."
                )

        if evidence.supporting_chunks:
            graph_lines.append("")
            graph_lines.append("[Supporting Chunks]")
            for i, chunk in enumerate(evidence.supporting_chunks, 1):
                title = chunk.get("title", "Unknown")
                section = chunk.get("section", "")
                text = chunk.get("text", "")[:200]
                graph_lines.append(f"  {i}. [{title}] {section}: {text}...")

        graph_lines.append("")
        graph_chunk = {
            "chunk_id": "graph_context",
            "paper_id": "graph",
            "title": "Knowledge Graph Context",
            "section": "Graph Evidence",
            "chunk_type": "graph_context",
            "text": "\n".join(graph_lines),
            "relevance_score": 1.0,
            "metadata": {
                "entity_count": len(evidence.entities),
                "relation_count": len(evidence.relations),
                "direct_edge": evidence.metadata.get("direct_edges", 0) > 0,
            },
        }
        chunks.append(graph_chunk)

        # Add supporting vector chunks (without modifying their structure)
        for chunk in evidence.supporting_chunks:
            # Ensure required fields
            chunk_copy = dict(chunk)
            chunk_copy.setdefault("chunk_type", "child")
            chunk_copy.setdefault("relevance_score", chunk.get("score", 0.5))
            chunks.append(chunk_copy)

        return chunks

    def ask(self, query: str) -> GeneratedAnswer:
        """
        Process a query through the full CRAG + RAPTOR pipeline.

        Args:
            query: User question

        Returns:
            GeneratedAnswer with answer, sources, and metadata
        """
        logger.info("Processing query: %s", query)

        # Step 1: Check Semantic Cache
        cached = self.cache.get(query)
        if cached:
            logger.info("Cache HIT - returning cached answer")
            return GeneratedAnswer(
                answer=cached["answer"],
                sources=cached["sources"],
                query=query,
                strategy=f"cached_{cached['strategy']}",
                faithful=True,
                confidence=cached.get("similarity", 1.0),
                self_check_reasoning=f"Cache hit (similarity: {cached.get('similarity', 0):.4f})",
            )

        # Step 2: Rewrite query for better retrieval
        rewritten = self.rewriter.rewrite(query)
        search_query = rewritten.rewritten
        logger.info("Query rewritten: '%s' -> '%s'", query, search_query)
        if rewritten.alternatives:
            logger.debug("Alternatives: %s", rewritten.alternatives)

        # Step 3: Route the (rewritten) query, passing original for definition pattern check
        routing = self.router.route(search_query, original_query=query)
        
        # Phase Q1: Reliable mode override - force single_shot for normal questions
        # when Reliable Mode is ON (default), unless Experimental Routing is enabled
        if config.cpu_safe.reliable_mode and not config.cpu_safe.experimental_routing:
            # Only force single_shot if not already single_shot and not no_retrieval
            if routing.strategy not in ("single_shot", "no_retrieval"):
                # Preserve the original reasoning but change the strategy
                routing = RoutingDecision(
                    strategy="single_shot",
                    confidence=0.75,
                    reasoning=routing.reasoning + " (reliable mode forced to single_shot)",
                    query_type=routing.query_type,
                    use_raptor=False,
                    use_graph=False,
                )
                logger.info("Reliable mode: overridden strategy from %s to single_shot", routing.strategy)
        logger.info("Routing: strategy=%s, confidence=%.2f, type=%s, reason=%s, use_raptor=%s",
                    routing.strategy, routing.confidence, routing.query_type, routing.reasoning,
                    routing.use_raptor)

        # Force strategy override via environment variable (for testing/baseline)
        force_strategy = os.getenv("FORCE_STRATEGY")
        if force_strategy:
            valid_strategies = ("no_retrieval", "baseline", "single_shot", "multi_step", "crag_raptor", "graph_rag")
            if force_strategy in valid_strategies:
                logger.info("FORCE_STRATEGY: overriding strategy from %s to %s", routing.strategy, force_strategy)
                routing = RoutingDecision(
                    strategy=force_strategy,
                    confidence=1.0,
                    reasoning=f"Forced via FORCE_STRATEGY={force_strategy}",
                    query_type=routing.query_type,
                    use_raptor=force_strategy in ("crag_raptor",),
                    use_graph=force_strategy in ("graph_rag",),
                )
            else:
                logger.warning("FORCE_STRATEGY=%s is not a valid strategy, ignoring", force_strategy)

        # Step 4: Handle based on strategy
        if routing.strategy == "no_retrieval":
            answer = self.generator.generate_no_retrieval(query)
            # Store in cache
            self.cache.put(query, answer.answer, answer.sources, answer.strategy)
            return answer

        elif routing.strategy == "baseline":
            logger.info("Baseline strategy engaged - dense retrieve + generate (no corrective loop)")
            # Simple dense retrieval without CRAG rewrite loop
            retrieval_result = self.retriever.retrieve_multi_type(
                query=search_query,
                chunk_types=["child", "abstract"],
            )

            if not retrieval_result.chunks:
                answer = GeneratedAnswer(
                    answer="No relevant information found using baseline retrieval.",
                    sources=[],
                    query=query,
                    strategy="baseline_no_results",
                    faithful=True,
                    confidence=0.0,
                    self_check_reasoning="Baseline retrieval returned no results"
                )
                self.cache.put(query, answer.answer, answer.sources, answer.strategy)
                return answer

            # Grade chunks for relevance (optional, but keeps quality filters)
            graded_chunks = self.grader.grade(query, retrieval_result.chunks)
            graded_chunks = [self._to_dict(c) for c in graded_chunks] if hasattr(self, '_to_dict') else graded_chunks

            # Expand to parents (inherits child scores from Q2.3) and apply sludge filter (Q2.1)
            # The generator will handle parent expansion and sludge filter via its own logic
            answer = self.generator.generate(
                query=query,
                graded_chunks=graded_chunks,
                retriever=self.retriever,
            )

            answer.strategy = "baseline"

            # Store in cache
            self.cache.put(query, answer.answer, answer.sources, answer.strategy)
            return answer

        elif routing.strategy == "multi_step":
            logger.info("Multi-step strategy engaged - using MultiStepRetriever")
            ms_result = self.multi_step_retriever.retrieve(query=search_query)

            # Log the multi-step process
            if ms_result.gap_identified:
                logger.info("Multi-step gap identified - follow-up question generated")
            logger.info("Multi-step process: %d sub-questions, %d retrievals, %d chunks",
                       len(ms_result.sub_questions), ms_result.total_retrievals, len(ms_result.all_chunks))
            for sq in ms_result.sub_questions:
                logger.info("  Sub-question %d: %s", sq.index, sq.question)

            if ms_result.gap_identified:
                logger.info("Multi-step gap identified - follow-up question generated: '%s'",
                           ms_result.follow_up_question[:60] if ms_result.follow_up_question else '')

            # Pass the collected evidence chunks to the generator
            final_chunks = ms_result.all_chunks

            if not final_chunks:
                answer = GeneratedAnswer(
                    answer="No relevant information found across all retrieval steps.",
                    sources=[],
                    query=query,
                    strategy="multi_step_no_results",
                    faithful=True,
                    confidence=0.0,
                    self_check_reasoning="Multi-step retrieval completed with no results"
                )
                self.cache.put(query, answer.answer, answer.sources, answer.strategy)
                return answer

            # Generate answer with multi-step mode
            answer = self.generator.generate(
                query=query,
                graded_chunks=final_chunks,
                retriever=self.retriever,
                multi_step_mode=True,
            )

            # Add multi-step strategy label
            answer.strategy = f"multi_step_{len(ms_result.sub_questions)}sub_{ms_result.total_retrievals}retrievals"

            # Include multi-step metadata
            ms_meta = getattr(ms_result, "multi_step_metadata", None)
            if ms_meta:
                answer.multi_step_metadata = ms_meta
            else:
                # Build fallback metadata from result attributes
                answer.multi_step_metadata = {
                    "sub_questions": [
                        {
                            "index": sq.index,
                            "question": sq.question,
                            "retrieval_success": sq.retrieval_success,
                            "relevance_score": sq.relevance_score,
                        }
                        for sq in ms_result.sub_questions
                    ],
                    "follow_up_question": ms_result.follow_up_question,
                    "gap_identified": ms_result.gap_identified,
                    "total_retrievals": ms_result.total_retrievals,
                    "num_chunks": len(ms_result.all_chunks),
                    "process_log": list(ms_result.process_log),
                }

            # Store in cache
            self.cache.put(query, answer.answer, answer.sources, answer.strategy)
            return answer

        elif routing.strategy == "single_shot":
            # Step 5: Corrective Retrieval (CRAG) instead of plain retrieve+grade
            logger.info("Starting Corrective RAG retrieval...")
            corrective_result = self.corrective_retriever.retrieve(
                query=search_query,
                chunk_types=["child", "abstract"],
            )

            # Log corrective retrieval details
            eval_info = corrective_result.evaluation
            if eval_info:
                logger.info("CRAG Evaluation: %s (confidence=%.2f, chunks=%d, avg_rel=%.2f) - %s",
                            eval_info.evaluation, eval_info.confidence, eval_info.num_chunks,
                            eval_info.avg_relevance, eval_info.reasoning)
            logger.info("CRAG Correction: %s (total retrievals: %d)",
                        corrective_result.correction_applied, corrective_result.total_retrievals)
            if corrective_result.rewrite_used:
                logger.info("CRAG Rewrite used: '%s'", corrective_result.rewrite_used)

            final_chunks = corrective_result.chunks

            if not final_chunks:
                answer = GeneratedAnswer(
                    answer="No relevant information found after corrective retrieval.",
                    sources=[],
                    query=query,
                    strategy="crag_no_results",
                    faithful=True,
                    confidence=0.0,
                    self_check_reasoning=f"CRAG evaluation: {eval_info.evaluation if eval_info else 'unknown'}"
                )
                self.cache.put(query, answer.answer, answer.sources, answer.strategy)
                return answer

            # Step 6: Generate answer with citations + self-check
            answer = self.generator.generate(
                query=query,
                graded_chunks=final_chunks,
                retriever=self.retriever,
            )

            # Add CRAG metadata to answer
            answer.strategy = f"crag_{corrective_result.correction_applied}"

            # Step 7: Store in cache
            self.cache.put(query, answer.answer, answer.sources, answer.strategy)

            return answer

        elif routing.strategy == "crag_raptor":
            logger.info("RAPTOR-enhanced retrieval engaged")
            raptor_result = self.raptor_retriever.retrieve(
                query=search_query,
            )

            # Log RAPTOR retrieval details
            logger.info("RAPTOR retrieval: query_type=%s, summaries=%d, details=%d, combined=%d",
                       raptor_result.query_type,
                       len(raptor_result.summary_nodes),
                       len(raptor_result.detail_chunks),
                       len(raptor_result.combined_chunks))

            for node in raptor_result.summary_nodes[:3]:
                logger.info("  Summary: %s [%s] (level %d)",
                           node.chunk_id, node.section, node.level)

            # Pass the combined evidence to the answer generator
            final_chunks = raptor_result.combined_chunks

            if not final_chunks:
                answer = GeneratedAnswer(
                    answer="No relevant information found using RAPTOR-enhanced retrieval.",
                    sources=[],
                    query=query,
                    strategy="raptor_no_results",
                    faithful=True,
                    confidence=0.0,
                    self_check_reasoning="RAPTOR retrieval returned no results"
                )
                self.cache.put(query, answer.answer, answer.sources, answer.strategy)
                return answer

            # Generate answer with multi-step mode (since we have mixed evidence)
            answer = self.generator.generate(
                query=query,
                graded_chunks=final_chunks,
                retriever=self.retriever,
                multi_step_mode=True,
            )

            # Add RAPTOR strategy label
            answer.strategy = f"raptor_{raptor_result.query_type}"

            # Include RAPTOR metadata
            if raptor_result.retrieval_stats:
                answer.multi_step_metadata = {
                    "query_type": raptor_result.retrieval_stats.get("query_type"),
                    "summary_count": raptor_result.retrieval_stats.get("num_summaries"),
                    "detail_count": raptor_result.retrieval_stats.get("num_details"),
                    "combined_count": raptor_result.retrieval_stats.get("num_combined"),
                }

            # Store in cache
            self.cache.put(query, answer.answer, answer.sources, answer.strategy)
            return answer

        elif routing.strategy == "graph_rag":
            logger.info("GraphRAG retrieval engaged")
            if not config.graph.enabled:
                logger.warning("GraphRAG is disabled in config, falling back to single_shot")
                # Fall through to single_shot
            else:
                graph_evidence = self.graph_retriever.retrieve(
                    query=search_query,
                )

                # Log GraphRAG retrieval details
                logger.info("GraphRAG retrieval: linked_entities=%d, relations=%d, supporting_chunks=%d",
                           len(graph_evidence.entities),
                           len(graph_evidence.relations),
                           len(graph_evidence.supporting_chunks))

                for entity in graph_evidence.entities[:3]:
                    logger.info("  Linked entity: %s (%s)", entity.name, entity.type.value)

                for rel in graph_evidence.relations[:3]:
                    logger.info("  Relation: %s -[%s]-> %s",
                               rel.source_id, rel.relation_type.value, rel.target_id)

                # Fallback if graph evidence is empty
                if not graph_evidence.entities and not graph_evidence.relations:
                    logger.warning("GraphRAG returned empty evidence, falling back to single_shot (CRAG)")
                    # Actually run the corrective retriever (same as single_shot path)
                    logger.info("Starting Corrective RAG retrieval as fallback...")
                    corrective_result = self.corrective_retriever.retrieve(
                        query=search_query,
                        chunk_types=["child", "abstract"],
                    )

                    # Log corrective retrieval details
                    eval_info = corrective_result.evaluation
                    if eval_info:
                        logger.info("CRAG Evaluation: %s (confidence=%.2f, chunks=%d, avg_rel=%.2f) - %s",
                                    eval_info.evaluation, eval_info.confidence, eval_info.num_chunks,
                                    eval_info.avg_relevance, eval_info.reasoning)
                    logger.info("CRAG Correction: %s (total retrievals: %d)",
                                corrective_result.correction_applied, corrective_result.total_retrievals)
                    if corrective_result.rewrite_used:
                        logger.info("CRAG Rewrite used: '%s'", corrective_result.rewrite_used)

                    final_chunks = corrective_result.chunks

                    if not final_chunks:
                        answer = GeneratedAnswer(
                            answer="No relevant information found after corrective retrieval.",
                            sources=[],
                            query=query,
                            strategy="graph_rag_fallback_crag_no_results",
                            faithful=True,
                            confidence=0.0,
                            self_check_reasoning=f"GraphRAG fallback CRAG evaluation: {eval_info.evaluation if eval_info else 'unknown'}"
                        )
                        self.cache.put(query, answer.answer, answer.sources, answer.strategy)
                        return answer

                    # Generate answer with citations + self-check
                    answer = self.generator.generate(
                        query=query,
                        graded_chunks=final_chunks,
                        retriever=self.retriever,
                    )

                    # Add fallback strategy label
                    answer.strategy = f"graph_rag_fallback_crag_{corrective_result.correction_applied}"

                    # Store in cache
                    self.cache.put(query, answer.answer, answer.sources, answer.strategy)

                    return answer
                else:
                    # Convert GraphEvidence to chunks for generator
                    final_chunks = self._graph_evidence_to_chunks(graph_evidence)

                    if not final_chunks:
                        answer = GeneratedAnswer(
                            answer="No relevant information found using GraphRAG retrieval.",
                            sources=[],
                            query=query,
                            strategy="graph_rag_no_results",
                            faithful=True,
                            confidence=0.0,
                            self_check_reasoning="GraphRAG retrieval returned no results"
                        )
                        self.cache.put(query, answer.answer, answer.sources, answer.strategy)
                        return answer

                    # Generate answer with multi-step mode (mixed evidence)
                    answer = self.generator.generate(
                        query=query,
                        graded_chunks=final_chunks,
                        retriever=self.retriever,
                        multi_step_mode=True,
                    )

                    # Add GraphRAG strategy label
                    answer.strategy = f"graph_rag_{graph_evidence.metadata.get('query_type', 'relational')}"

                    # Include GraphRAG metadata
                    answer.multi_step_metadata = {
                        "query_type": "relational",
                        "linked_entities": graph_evidence.metadata.get("linked_entity_ids", []),
                        "expanded_entity_count": graph_evidence.metadata.get("expanded_entity_count", 0),
                        "relation_count": graph_evidence.metadata.get("relation_count", 0),
                        "supporting_chunk_count": graph_evidence.metadata.get("supporting_chunk_count", 0),
                        "hops": graph_evidence.metadata.get("hops", 1),
                        "graph_context": graph_evidence.to_context_string(),
                    }

                    # Store in cache
                    self.cache.put(query, answer.answer, answer.sources, answer.strategy)
                    return answer

        # Fallback if no strategy matched
        answer = self.generator.generate_no_retrieval(query)
        self.cache.put(query, answer.answer, answer.sources, "no_retrieval_fallback")
        return answer

    def interactive(self) -> None:
        """Run interactive CLI loop."""
        print("\n" + "=" * 70)
        print("AI Research Assistant - Interactive Mode (Phase 4: GraphRAG + RAPTOR + CRAG)")
        print("Features: Semantic Cache | Query Rewriting | Adaptive Routing | CRAG | Multi-Step | RAPTOR | GraphRAG | Self-Check")
        print("Type 'exit', 'quit', or Ctrl+C to quit")
        print("=" * 70 + "\n")

        while True:
            try:
                query = input("Question: ").strip()
                if not query:
                    continue
                if query.lower() in ("exit", "quit", "q"):
                    print("Goodbye!")
                    break

                answer = self.ask(query)
                self._print_answer(answer)

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except EOFError:
                print("\nGoodbye!")
                break
            except Exception as e:
                logger.error("Error processing query: %s", e)
                print(f"Error: {e}\n")

    def _print_answer(self, answer: GeneratedAnswer) -> None:
        """Print formatted answer with sources and metadata."""
        # Strategy indicator
        strategy_icons = {
            "no_retrieval": "🧠",
            "retrieval_augmented": "🔍",
            "multi_step_3sub_2retrievals": "🔍🧩",
            "crag_refine": "🔍✨",
            "crag_rewrite_merge": "🔍🔄",
            "crag_strong_rewrite": "🔍💪",
            "raptor_survey": "🗺️",
            "raptor_detail": "🗺️📄",
            "graph_rag_relational": "🕸️",
            "cached_no_retrieval": "⚡",
            "cached_retrieval_augmented": "⚡",
            "cached_single_shot": "⚡",
            "cached_multi_step_3sub_2retrievals": "⚡🧩",
            "cached_raptor_survey": "⚡🗺️",
            "cached_raptor_detail": "⚡🗺️📄",
            "cached_graph_rag_relational": "⚡🕸️",
            "baseline": "📄",
            "baseline_no_results": "❌",
            "no_results": "❌",
            "no_relevant": "⚠️",
            "multi_step_no_results": "❌",
            "raptor_no_results": "❌",
            "graph_rag_no_results": "❌",
            "error": "💥",
        }
        icon = strategy_icons.get(answer.strategy, "📝")

        print("\n" + "─" * 70)
        print(f"{icon} Answer [{answer.strategy}] (faithful={answer.faithful}, confidence={answer.confidence:.2f})")
        print("─" * 70)
        print(answer.answer)

        # Self-check info
        if answer.self_check_reasoning and answer.strategy not in ("no_retrieval",):
            if not answer.faithful:
                print(f"\n⚠️  Self-check: {answer.self_check_reasoning}")
            else:
                print(f"\n✅ Self-check: {answer.self_check_reasoning}")

        if answer.sources:
            print("\n📚 Sources:")
            print("─" * 70)
            for i, src in enumerate(answer.sources, 1):
                if src.get("source") == "model_knowledge":
                    print(f"  {i}. Model knowledge (no retrieval)")
                else:
                    title = src.get("title", "Unknown")
                    paper_id = src.get("paper_id", "")
                    section = src.get("section", "")
                    score = src.get("relevance_score", 0)
                    section_str = f" [{section}]" if section else ""
                    print(f"  {i}. {title} [{paper_id}]{section_str} (relevance: {score:.2f})")

        # Multi-step/RAPTOR info
        if getattr(answer, 'multi_step_metadata', None):
            ms = answer.multi_step_metadata
            print(f"\n🔍 Multi-step/RAPTOR: type={ms.get('query_type')}, "
                  f"summaries={ms.get('summary_count', 0)}, details={ms.get('detail_count', 0)}, "
                  f"combined={ms.get('combined_count', 0)}")

        # Cache info
        if "cached" in answer.strategy:
            print("\n💾 Served from semantic cache")

        print("\n")


def main():
    parser = _get_parser()
    args = parser.parse_args()

    assistant = ResearchAssistant(verbose=args.verbose, cache_path=args.cache_path)

    if args.cache_stats:
        stats = assistant.cache.stats()
        print("\n📊 Semantic Cache Statistics:")
        print("─" * 40)
        for k, v in stats.items():
            print(f"  {k}: {v}")
        return

    if args.clear_cache:
        assistant.cache.clear()
        print("Cache cleared.")
        return

    if args.interactive or not args.query:
        assistant.interactive()
    else:
        answer = assistant.ask(args.query)
        assistant._print_answer(answer)


if __name__ == "__main__":
    main()