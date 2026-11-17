"""Answer card component for displaying RAG answers."""

import streamlit as st
from typing import Any, List, Dict, Optional


def render_answer_card(
    answer: str,
    strategy: str,
    cache_hit: bool = False,
    sources: Optional[List[Dict[str, Any]]] = None,
    latency: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Render a styled answer card with strategy badge and sources."""
    # Strategy badge
    strategy_icons = {
        "no_retrieval": "🧠",
        "single_shot": "🔍",
        "crag_refine": "🔍✨",
        "crag_rewrite_merge": "🔍🔄",
        "crag_strong_rewrite": "🔍💪",
        "raptor_survey": "🗺️",
        "raptor_detail": "🗺️📄",
        "graph_rag_relational": "🕸️",
        "graph_rag_technique": "🕸️",
        "cached_*": "⚡",
        "multi_step": "🔍🧩",
        "crag_na": "❌",
        "raptor_na": "❌",
        "graph_rag_no_results": "❌",
    }

    # Determine icon
    icon = "📝"
    for key, val in strategy_icons.items():
        if key == "*":
            continue
        if key in strategy:
            icon = val
            break

    # Also try substring match
    if icon == "📝":
        for key in strategy_icons:
            if key in strategy:
                icon = strategy_icons[key]
                break

    # Determine cache badge
    cache_badge = "⚡ Cache hit" if cache_hit else "🔍 Fresh retrieval"

    # Generation status info
    generation_status = metadata.get("generation_status", "ok") if metadata else "ok"
    fallback_type = metadata.get("fallback_type", "none") if metadata else "none" if metadata else ""

    with st.container():
        st.markdown("---")
        # Header row with strategy and cache
        col1, col2 = st.columns([6, 1])
        with col1:
            st.markdown(f"**{icon} {strategy.replace('_', ' ').title()}**  —  {cache_badge}")
        with col2:
            if latency is not None:
                st.caption(f"≈ {latency:.1f}s")

        # Answer text
        st.markdown(answer)

        # Generation status badge (only show if not ok)
        if generation_status != "ok":
            status_color = "🔴" if generation_status == "failed" else "🟡"
            st.caption(f"{status_color} Generation: {generation_status} ({fallback_type})")

        # Sources
        if sources:
            with st.expander("Sources", expanded=False):
                for i, src in enumerate(sources, 1):
                    title = src.get("title", "Unknown")
                    paper_id = src.get("paper_id", "")
                    section = src.get("section", "")
                    score = src.get("relevance_score", "")
                    section_str = f" [{section}]" if section else ""
                    pid_str = f" [{paper_id}]" if paper_id else ""
                    score_str = f" (relevance: {score:.2f})" if score else ""
                    st.markdown(f"**{i}. {title}{pid_str}{section_str}{score_str}**")

        # Metadata expander
        if metadata:
            with st.expander("Details", expanded=False):
                for k, v in metadata.items():
                    if k in ("generation_status", "fallback_type"):
                        continue  # already shown above
                    st.write(f"**{k}**: {v}")


def render_strategy_badge(strategy: str) -> None:
    """Render a small standalone strategy badge."""
    st.markdown(
        f'<span style="background:#e2e6ea; padding:2px 6px; '
        f'border-radius:3px; font-size:0.85em; '
        f'color:#485563;">{strategy.replace("_", " ").title()}</span>',
        unsafe_allow_html=True,
    )