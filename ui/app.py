"""AI Research Assistant — Streamlit UI (Phase U1 + U2).

Local-first Advanced RAG workbench with adaptive routing, CRAG, multi-step,
RAPTOR, semantic cache, and Stage-A GraphRAG.
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path so `src` is importable when running
# `streamlit run ui/app.py` from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time
from typing import Any, Dict, List, Optional

# Page config
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🕸️",
    layout="wide",
)

# Custom CSS for clean lab style
st.markdown("""
<style>
/* Strategy badge styling */
.st-bb {background: #e2e6ea !important; color: #485563 !important;}
/* Make answer text more readable */
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
/* Demo card styling */
.demo-card {border: 1px solid #e2e6ea; border-radius: 6px; padding: 1rem; margin-bottom: 1rem;}
</style>
""", unsafe_allow_html=True)

# ─── Imports from backend ──────────────────────────────────────────────
from src.main import ResearchAssistant
from ui.components.answer_card import render_answer_card, render_strategy_badge
from ui.components.status_panel import render_status_panel, render_action_buttons
from ui.components.demo_cards import render_demo_cards

# ─── Session state init ───────────────────────────────────────────────
if "assistant" not in st.session_state:
    st.session_state.assistant = ResearchAssistant(verbose=False)

if "last_answer" not in st.session_state:
    st.session_state.last_answer: Dict[str, Any] = {}

if "last_query" not in st.session_state:
    st.session_state.last_query: str = ""

if "cache_cleared" not in st.session_state:
    st.session_state.cache_cleared = False

if "last_asked_question" not in st.session_state:
    st.session_state.last_asked_question: str = ""

if "pending_demo" not in st.session_state:
    st.session_state.pending_demo: Optional[str] = None

# ─── Helper functions ─────────────────────────────────────────────────
def run_assistant_query(question: str) -> None:
    """Run a question through the backend and store results."""
    assistant: ResearchAssistant = st.session_state.assistant
    try:
        with st.spinner("Running locally — CPU may take 1–4 minutes..."):
            start = time.time()
            answer = assistant.ask(question)
            latency = time.time() - start

        # Store results
        st.session_state.last_answer = {
            "answer": answer.answer,
            "strategy": answer.strategy,
            "cache_hit": False,  # Will be updated if cached
            "sources": getattr(answer, "sources", []),
            "latency": latency,
            "faithful": getattr(answer, "faithful", True),
        }
        st.session_state.last_query = question
        st.rerun()

    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        # Provide helpful guidance
        with st.expander("Troubleshooting"):
            st.markdown(
                "- Ensure Ollama is running: `ollama serve`\n"
                "- Ensure Qdrant is running: `docker run -d -p 6333:6333 ... qdrant/qdrant`\n"
                "- Check environment: `python scripts/check_environment.py`"
            )


def render_ask_tab() -> None:
    """Render the Ask tab."""
    st.header("💬 Ask")

    # Question input
    col_input, col_btn = st.columns([6, 1])

    with col_input:
        question = st.text_input(
            "Your question",
            placeholder="e.g., What is Self-RAG?",
            key="question_input",
            label_visibility="collapsed",
        )

    with col_btn:
        ask_btn = st.button("Ask", use_container_width=True)

    # Sample question chips
    st.markdown("**Quick queries:**")
    cols = st.columns(4)
    sample_queries = [
        "What is Self-RAG?",
        "What techniques does Self-RAG introduce?",
        "How is Self-RAG related to CRAG?",
        "Give me an overview of advanced RAG methods",
    ]
    for i, sq in enumerate(sample_queries):
        with cols[i]:
            if st.button(sq, key=f"sample_{i}", help=sq, use_container_width=True):
                question = sq

    # Handle ask - only on explicit trigger, not on input value retention
    if ask_btn or st.session_state.pending_demo is not None:
        # Determine the question to process
        if st.session_state.pending_demo is not None:
            q = st.session_state.pending_demo
            st.session_state.pending_demo = None  # consume once
        else:
            q = question.strip()
        
        # Only process if there's a valid question
        if q and q.strip():
            st.session_state.last_asked_question = q.strip()
            # Clear any previous cache clear state
            st.session_state.cache_cleared = False
            run_assistant_query(q)

    # Display last answer if available
    if st.session_state.last_answer and st.session_state.last_query:
        last_q = st.session_state.last_query
        last_a = st.session_state.last_answer

        # Only show if the last question matches current input (or just always show)
        st.markdown("---")
        st.markdown("**Answer:**")
        render_strategy_badge(last_a["strategy"])
        render_answer_card(
            answer=last_a["answer"],
            strategy=last_a["strategy"],
            cache_hit=last_a["cache_hit"],
            sources=last_a["sources"],
            latency=last_a["latency"],
            metadata={
                "faithful": last_a["faithful"],
            },
        )


def render_demo_tab() -> None:
    """Render the Demo tab."""
    st.header("🎬 Demo")

    st.markdown(
        "Click a demo query to auto-run it. "
        "GraphRAG demos are experimental."
    )

    render_demo_cards(
        on_query_selected=lambda q: st.session_state.update(
            {"pending_demo": q}
        ),
        selected_tab_setter=lambda t: st.session_state.update(
            {"current_tab": "Ask"}
        ),
    )

    st.markdown("---")
    st.caption(
        "Demo queries illustrate typical strategy selection. "
        "Actual results depend on loaded papers, graph state, and model."
    )


def render_system_tab() -> None:
    """Render the System tab."""
    st.header("⚙️ System")

    assistant: ResearchAssistant = st.session_state.assistant

    # Gather status
    ollama_ok = assistant.llm.is_available() if hasattr(assistant, "llm") else False

    # Qdrant status
    qdrant_ok = False
    try:
        import qdrant_client
        client = qdrant_client.QdrantClient("localhost:6333")
        collections = client.get_collections()
        qdrant_ok = any(c.name == "ai_papers" for c in collections.collections)
    except Exception:
        qdrant_ok = False

    # Graph status
    graph_available = False
    try:
        graph_path = "data/graph/graph.json"
        import os
        if os.path.exists(graph_path):
            import json
            with open(graph_path) as f:
                graph_data = json.load(f)
            graph_available = graph_data.get("nodes", 0) > 0
    except Exception:
        graph_available = False

    # Cache status
    cache_entries = 0
    try:
        cache_entries = len(assistant.cache.stats().get("entries", 0))
    except Exception:
        cache_entries = 0

    # Render status panel
    render_status_panel(
        ollama_ok=ollama_ok,
        qdrant_ok=qdrant_ok,
        graph_available=graph_available,
        cache_entries=cache_entries,
        cpu_safe=True,  # Always on by default
    )

    # Action buttons
    st.markdown("### Actions")
    actions = render_action_buttons(clear_cache=True, refresh=True)

    if actions.get("clear_cache"):
        try:
            assistant.cache.clear()
            st.success("Semantic cache cleared.")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to clear cache: {e}")

    if actions.get("refresh"):
        st.success("Status refreshed.")
        st.rerun()

    # Guidance
    st.markdown("### Guidance")
    st.markdown(
        "- **Ollama**: Start with `ollama serve`, then `ollama pull qwen2.5-coder:3b`\n"
        "- **Qdrant**: Start with `docker run -d -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant`\n"
        "- **Build graph**: `python scripts/build_graph.py --limit 8` (abstracts-first)\n"
        "- **Build RAPTOR**: `python scripts/build_raptor.py --all`\n"
        "- **Clear cache**: uses the button above"
    )


def render_about_tab() -> None:
    """Render the About tab."""
    st.header("ℹ️ About")

    # Positioning statement
    st.markdown(
        "**Positioning**: A local-first, framework-free Advanced RAG lab that "
        "implements Adaptive routing, CRAG, multi-step retrieval, RAPTOR, semantic "
        "cache, and Stage-A GraphRAG in one coherent CPU-runnable system."
    )

    # GraphRAG status note
    st.info(
        "**GraphRAG**: Partial / Experimental Stage A — entity/relation extraction "
        "is heuristic; quality improves with larger models. Use `--limit` and "
        "abstracts-first on CPU."
    )

    # GitHub link
    st.markdown(
        "[GitHub Repository](https://github.com/arashnicoomanesh/ai-research-assistant)  \n"
        f"**Current tag**: `v0.2.0` — Advanced RAG lab with RAPTOR + experimental "
        "GraphRAG Stage A"
    )

    # Architecture images if available
    st.markdown("### Architecture diagrams")
    import os
    img_dir = "docs"
    img_files = []
    if os.path.exists(img_dir):
        for f in os.listdir(img_dir):
            if f.endswith(".webp") or f.endswith(".png") or f.endswith(".jpg"):
                img_files.append(f)

    if img_files:
        cols = st.columns(min(len(img_files), 3))
        for i, img in enumerate(img_files):
            with cols[i % len(cols)]:
                st.image(os.path.join(img_dir, img), caption=img, use_column_width=True)
    else:
        st.caption(
            "No architecture images found in `docs/`. "
            "Expected: `docs/architecture.webp`, `docs/graphrag-stage-a.webp`"
        )

    # Quick start reminder
    st.markdown("### Quick start")
    st.code(
        "pip install -r requirements.txt\n"
        "streamlit run ui/app.py\n"
        "\n"
        "# Then in order:\n"
        "docker run -d -p 6333:6333 -p 6334:6334 \\\n"
        "  -v $(pwd)/qdrant_storage:/qdrant/storage \\\n"
        "  qdrant/qdrant\n"
        "ollama serve\n"
        "ollama pull qwen2.5-coder:3b\n"
        "ollama pull nomic-embed-text:v1.5\n"
        "python scripts/check_environment.py\n"
        "python scripts/ingest_papers.py\n"
        "python scripts/build_graph.py --limit 8\n"
        "python scripts/build_raptor.py --all\n"
        'python src/main.py "What is Self-RAG?"',
        language="bash",
    )


# ─── Main app ─────────────────────────────────────────────────────────
def main() -> None:
    """Main entry point for the Streamlit UI."""
    # Tab navigation
    tabs = st.tabs(["Ask", "Demo", "System", "About"])

    with tabs[0]:
        render_ask_tab()

    with tabs[1]:
        render_demo_tab()

    with tabs[2]:
        render_system_tab()

    with tabs[3]:
        render_about_tab()


if __name__ == "__main__":
    main()