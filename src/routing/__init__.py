"""
Routing package for AI Research Assistant.
"""
from src.routing.adaptive_router import AdaptiveRouter, RoutingDecision, get_router

__all__ = [
    "AdaptiveRouter",
    "RoutingDecision",
    "get_router",
]