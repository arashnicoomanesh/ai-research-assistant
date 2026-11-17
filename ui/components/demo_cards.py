"""Demo cards component for the Demo tab."""

import streamlit as st
from typing import Callable, List, Dict, Any


def render_demo_cards(
    on_query_selected: Callable[[str], None],
    selected_tab_setter: Callable[[str], None],
) -> None:
    """Render clickable demo query cards."""
    st.markdown("### 🎬 Demo Queries")

    demo_queries: List[Dict[str, Any]] = [
        {
            "title": "🔍 Definition",
            "query": "What is Self-RAG?",
            "note": "Adaptive QA / single_shot tendency",
        },
        {
            "title": "🕸️ GraphRAG technique",
            "query": "What techniques does Self-RAG introduce?",
            "note": "Experimental GraphRAG Stage A",
        },
        {
            "title": "📊 RAPTOR overview",
            "query": "Give me an overview of advanced RAG methods in these papers",
            "note": "Hierarchical RAPTOR tendency",
        },
        {
            "title": "⚡ Cache repeat",
            "query": "What is Self-RAG?",
            "note": "Second run should be faster via semantic cache",
        },
    ]

    # Display in columns or expanders
    for i, demo in enumerate(demo_queries, 1):
        with st.container():
            col1, col2 = st.columns([1, 5])
            with col1:
                st.markdown(f"**{demo['title']}**")
            with col2:
                st.write(demo["note"])
            if st.button(f"Run", key=f"demo_{i}", use_container_width=True):
                on_query_selected(demo["query"])
                selected_tab_setter("Ask")