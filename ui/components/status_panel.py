"""Status panel component for system health."""

import streamlit as st
from typing import Any, Dict, List, Optional


def render_status_panel(
    ollama_ok: bool = False,
    qdrant_ok: bool = False,
    graph_available: bool = False,
    cache_entries: int = 0,
    cpu_safe: bool = True,
) -> None:
    """Render system health status cards."""
    st.markdown("### System Status")

    # Ollama card
    with st.container():
        st.markdown("**🦙 Ollama**")
        col1, col2 = st.columns([4, 1])
        with col1:
            status = "✅ Running" if ollama_ok else "❌ Not reachable"
            st.write(status)
        with col2:
            if ollama_ok:
                st.write("OK")
            else:
                st.error("?")
        st.caption("Local LLM server")

    # Qdrant card
    with st.container():
        st.markdown("**🗃️ Qdrant**")
        col1, col2 = st.columns([4, 1])
        with col1:
            status = "✅ Running" if qdrant_ok else "❌ Not reachable"
            st.write(status)
        with col2:
            if qdrant_ok:
                st.write("OK")
            else:
                st.error("?")
        st.caption("Vector database")

    # Graph card
    with st.container():
        st.markdown("**🕸️ GraphRAG**")
        col1, col2 = st.columns([4, 1])
        with col1:
            status = "✅ Available" if graph_available else "⚠️ Experimental Stage A"
            st.write(status)
        with col2:
            if graph_available:
                st.write("OK")
            else:
                st.warning("Stage A")
        st.caption("Knowledge graph (Partial/Experimental)")

    # Cache card
    with st.container():
        st.markdown("**⚡ Semantic Cache**")
        col1, col2 = st.columns([4, 1])
        with col1:
            st.write(f"{cache_entries} entries")
        with col2:
            st.write("—")
        st.caption("Exact-repeat HIT → instant answer")

    # CPU-Safe note
    st.markdown("**💻 CPU-Safe Mode**: Default ON — disables expensive graders/self-checks for responsive latency on 3B models.")


def render_action_buttons(
    clear_cache: bool = False,
    refresh: bool = True,
) -> Dict[str, bool]:
    """Render action buttons and return click states."""
    cols = st.columns([1, 1, 1])
    results = {}

    with cols[0]:
        if st.button("🔄 Refresh", use_container_width=True, disabled=not refresh):
            results["refresh"] = True

    with cols[1]:
        if st.button("🧹 Clear cache", use_container_width=True):
            results["clear_cache"] = True

    with cols[2]:
        # Download or info button placeholder
        results["action"] = False

    return results