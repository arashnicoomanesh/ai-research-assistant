"""
Configuration management for the AI Research Assistant.

Loads settings from environment variables with sensible defaults.
All configuration is centralized here for easy modification.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


@dataclass
class OllamaConfig:
    """Ollama service configuration."""
    host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    llm_model: str = os.getenv("OLLAMA_LLM_MODEL", "qwen2.5-coder:3b")
    embedding_model: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:v1.5")
    llm_timeout_seconds: int = int(os.getenv("LLM_TIMEOUT_SECONDS", "300"))


@dataclass
class QdrantConfig:
    """Qdrant vector database configuration."""
    host: str = os.getenv("QDRANT_HOST", "localhost")
    port: int = int(os.getenv("QDRANT_PORT", "6333"))
    collection: str = os.getenv("QDRANT_COLLECTION", "ai_papers")


@dataclass
class ChunkingConfig:
    """Text chunking configuration."""
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    min_chunk_size: int = int(os.getenv("MIN_CHUNK_SIZE", "100"))


@dataclass
class RetrievalConfig:
    """Retrieval configuration."""
    top_k_dense: int = int(os.getenv("TOP_K_DENSE", "10"))
    top_k_bm25: int = int(os.getenv("TOP_K_BM25", "10"))
    top_k_final: int = int(os.getenv("TOP_K_FINAL", "5"))
    rerank_top_k: int = int(os.getenv("RERANK_TOP_K", "20"))
    hybrid_enabled: bool = os.getenv("HYBRID_ENABLED", "true").lower() == "true"


@dataclass
class EmbeddingConfig:
    """Embedding configuration for CPU-safe batch processing."""
    batch_size: int = int(os.getenv("EMBED_BATCH_SIZE", "4"))
    timeout_seconds: int = int(os.getenv("EMBED_TIMEOUT_SECONDS", "300"))
    max_retries: int = int(os.getenv("EMBED_MAX_RETRIES", "2"))
    embed_parents: bool = os.getenv("EMBED_PARENTS", "false").lower() == "true"


@dataclass
class GenerationConfig:
    """Answer generation configuration."""
    temperature: float = float(os.getenv("TEMPERATURE", "0.1"))
    max_tokens: int = int(os.getenv("MAX_TOKENS", "2048"))


@dataclass
class CPUSafeConfig:
    """CPU-safe mode configuration for resource-constrained environments."""
    enabled: bool = os.getenv("CPU_SAFE_MODE", "true").lower() == "true"
    enable_query_rewrite: bool = os.getenv("ENABLE_QUERY_REWRITE", "true").lower() == "true"
    enable_llm_grader: bool = os.getenv("ENABLE_LLM_GRADER", "false").lower() == "true"
    enable_retrieval_evaluator: bool = os.getenv("ENABLE_RETRIEVAL_EVALUATOR", "false").lower() == "true"
    enable_self_check: bool = os.getenv("ENABLE_SELF_CHECK", "false").lower() == "true"
    retrieve_top_k: int = int(os.getenv("RETRIEVE_TOP_K", "5"))
    context_top_k: int = int(os.getenv("CONTEXT_TOP_K", "5"))
    reliable_mode: bool = os.getenv("RELIABLE_MODE", "true").lower() == "true"
    experimental_routing: bool = os.getenv("EXPERIMENTAL_ROUTING", "false").lower() == "true"


@dataclass
class PathsConfig:
    """Filesystem paths configuration."""
    project_root: Path = Path(__file__).parent.parent.parent
    papers_dir: Path = Path(os.getenv("PAPERS_DIR", str(project_root / "data" / "papers")))
    processed_dir: Path = Path(os.getenv("PROCESSED_DIR", str(project_root / "data" / "processed")))
    bm25_index_dir: Path = Path(os.getenv("BM25_INDEX_DIR", str(project_root / "data" / "processed" / "bm25")))

    def __post_init__(self):
        self.papers_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.bm25_index_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class GraphConfig:
    """GraphRAG configuration."""
    enabled: bool = os.getenv("GRAPH_ENABLED", "true").lower() == "true"
    graph_dir: str = os.getenv("GRAPH_DIR", "data/graph")
    hops: int = int(os.getenv("GRAPH_HOPS", "1"))
    max_entities_per_text: int = int(os.getenv("GRAPH_MAX_ENTITIES_PER_TEXT", "8"))
    max_relations_per_text: int = int(os.getenv("GRAPH_MAX_RELATIONS_PER_TEXT", "8"))
    supporting_chunks: int = int(os.getenv("GRAPH_SUPPORTING_CHUNKS", "3"))
    build_source: str = os.getenv("GRAPH_BUILD_SOURCE", "abstract_first")


@dataclass
class Config:
    """Main configuration container."""
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    cpu_safe: CPUSafeConfig = field(default_factory=CPUSafeConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)


# Global config instance
config = Config()