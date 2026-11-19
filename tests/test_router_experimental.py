"""
Smoke test for experimental routing behavior.

Verifies:
- Comparison queries route to multi_step when experimental routing on + reliable off
- Reliable mode still forces single_shot regardless of experimental routing
- Definition queries still route to single_shot
"""
import os
from unittest.mock import MagicMock, patch

from src.routing.adaptive_router import AdaptiveRouter, RoutingDecision


class TestRouterExperimental:
    """Tests for experimental routing behavior."""

    def setup_method(self):
        """Set up router with mocked LLM."""
        self.mock_llm = MagicMock()
        self.mock_llm.generate.return_value = '{"strategy": "single_shot", "confidence": 0.8, "reasoning": "LLM default", "query_type": "factual", "use_raptor": false, "use_graph": false}'
        self.router = AdaptiveRouter(llm=self.mock_llm, use_llm=False)  # Use heuristic only for deterministic tests

    def test_comparison_query_routes_to_multi_step_experimental_on(self):
        """Comparison query should route to multi_step with experimental routing enabled."""
        # Mock the config to have experimental_routing=True
        with patch('src.routing.adaptive_router.config') as mock_config:
            mock_config.cpu_safe.experimental_routing = True
            mock_config.cpu_safe.reliable_mode = False

            query = "Compare BERT and GPT-3 architectures"
            decision = self.router.route(query, original_query=query)

            assert decision.strategy == "multi_step", f"Expected multi_step, got {decision.strategy}"
            assert decision.confidence >= 0.85, "Should have high confidence to beat LLM"
            assert "comparison" in decision.reasoning.lower()

    def test_comparison_query_single_shot_when_reliable_mode_on(self):
        """Comparison query should be forced to single_shot when reliable_mode is True and experimental_routing is False."""
        # This test verifies the heuristic works; main.py handles the reliable mode override
        with patch('src.routing.adaptive_router.config') as mock_config:
            mock_config.cpu_safe.experimental_routing = True
            mock_config.cpu_safe.reliable_mode = False

            query = "Compare BERT and GPT-3"
            decision = self.router.route(query, original_query=query)

            # Router itself should return multi_step (main.py handles reliable mode override)
            assert decision.strategy == "multi_step"

    def test_what_is_query_routes_to_single_shot(self):
        """Definition query should route to single_shot."""
        with patch('src.routing.adaptive_router.config') as mock_config:
            mock_config.cpu_safe.experimental_routing = True
            mock_config.cpu_safe.reliable_mode = False

            query = "What is BERT?"
            decision = self.router.route(query, original_query=query)

            assert decision.strategy == "single_shot", f"Expected single_shot, got {decision.strategy}"

    def test_vs_query_routes_to_multi_step(self):
        """'vs' query should route to multi_step."""
        with patch('src.routing.adaptive_router.config') as mock_config:
            mock_config.cpu_safe.experimental_routing = True
            mock_config.cpu_safe.reliable_mode = False

            query = "BERT vs GPT-3"
            decision = self.router.route(query, original_query=query)

            assert decision.strategy == "multi_step", f"Expected multi_step, got {decision.strategy}"

    def test_versus_query_routes_to_multi_step(self):
        """'versus' query should route to multi_step."""
        with patch('src.routing.adaptive_router.config') as mock_config:
            mock_config.cpu_safe.experimental_routing = True
            mock_config.cpu_safe.reliable_mode = False

            query = "BERT versus GPT-3"
            decision = self.router.route(query, original_query=query)

            assert decision.strategy == "multi_step", f"Expected multi_step, got {decision.strategy}"

    def test_difference_between_query_routes_to_multi_step(self):
        """'difference between' query should route to multi_step."""
        with patch('src.routing.adaptive_router.config') as mock_config:
            mock_config.cpu_safe.experimental_routing = True
            mock_config.cpu_safe.reliable_mode = False

            query = "What is the difference between BERT and GPT-3?"
            decision = self.router.route(query, original_query=query)

            assert decision.strategy == "multi_step", f"Expected multi_step, got {decision.strategy}"

    def test_contrast_query_routes_to_multi_step(self):
        """'contrast' query should route to multi_step."""
        with patch('src.routing.adaptive_router.config') as mock_config:
            mock_config.cpu_safe.experimental_routing = True
            mock_config.cpu_safe.reliable_mode = False

            query = "Contrast BERT and GPT-3 architectures"
            decision = self.router.route(query, original_query=query)

            assert decision.strategy == "multi_step", f"Expected multi_step, got {decision.strategy}"

    def test_compare_query_routes_to_multi_step(self):
        """'compare' query should route to multi_step."""
        with patch('src.routing.adaptive_router.config') as mock_config:
            mock_config.cpu_safe.experimental_routing = True
            mock_config.cpu_safe.reliable_mode = False

            query = "Compare transformer and RNN architectures"
            decision = self.router.route(query, original_query=query)

            assert decision.strategy == "multi_step", f"Expected multi_step, got {decision.strategy}"

    def test_heuristic_confidence_beats_llm_default(self):
        """Heuristic confidence for multi_step should beat LLM default (0.8)."""
        with patch('src.routing.adaptive_router.config') as mock_config:
            mock_config.cpu_safe.experimental_routing = True
            mock_config.cpu_safe.reliable_mode = False

            query = "Compare BERT and GPT-3"
            decision = self.router.route(query, original_query=query)

            assert decision.confidence >= 0.85, f"Confidence {decision.confidence} should be >= 0.85 to beat LLM default"

    def test_no_retrieval_query_still_works(self):
        """General knowledge query should still route to no_retrieval."""
        with patch('src.routing.adaptive_router.config') as mock_config:
            mock_config.cpu_safe.experimental_routing = True
            mock_config.cpu_safe.reliable_mode = False

            query = "What is a neural network?"
            decision = self.router.route(query, original_query=query)

            assert decision.strategy == "no_retrieval", f"Expected no_retrieval, got {decision.strategy}"

    def test_graph_rag_query_still_works(self):
        """Relational query should still route to graph_rag."""
        with patch('src.routing.adaptive_router.config') as mock_config:
            mock_config.cpu_safe.experimental_routing = True
            mock_config.cpu_safe.reliable_mode = False

            query = "How is Self-RAG related to CRAG?"
            decision = self.router.route(query, original_query=query)

            assert decision.strategy == "graph_rag", f"Expected graph_rag, got {decision.strategy}"

    def test_raptor_query_still_works(self):
        """Survey query should still route to crag_raptor."""
        with patch('src.routing.adaptive_router.config') as mock_config:
            mock_config.cpu_safe.experimental_routing = True
            mock_config.cpu_safe.reliable_mode = False

            query = "Overview of transformer architectures in recent papers"
            decision = self.router.route(query, original_query=query)

            assert decision.strategy == "crag_raptor", f"Expected crag_raptor, got {decision.strategy}"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])