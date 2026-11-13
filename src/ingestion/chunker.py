"""
Parent-Child Chunker for AI Research Papers.

Implements hierarchical chunking:
- Parent chunks: Large context windows (full sections or ~1000 tokens)
- Child chunks: Small retrieval units (~300-500 tokens) with parent references
- Abstract: Special high-value chunk type

Each chunk carries rich metadata for retrieval and citation.
"""
import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from src.ingestion.pdf_parser import ParsedPaper, PaperSection

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """
    Represents a text chunk with metadata.

    Attributes:
        chunk_id: Unique identifier
        paper_id: Source paper identifier
        title: Paper title
        section: Section heading
        chunk_type: "child" | "parent" | "abstract"
        text: Chunk content
        parent_id: Parent chunk ID (for child chunks)
        chunk_index: Index within parent (for child chunks)
        token_count: Approximate token count
        metadata: Additional metadata
    """
    chunk_id: str
    paper_id: str
    title: str
    section: str
    chunk_type: str  # "child", "parent", "abstract"
    text: str
    parent_id: Optional[str] = None
    chunk_index: int = 0
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "chunk_id": self.chunk_id,
            "paper_id": self.paper_id,
            "title": self.title,
            "section": self.section,
            "chunk_type": self.chunk_type,
            "text": self.text,
            "parent_id": self.parent_id,
            "chunk_index": self.chunk_index,
            "token_count": self.token_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        """Create from dictionary."""
        return cls(**data)


class ParentChildChunker:
    """
    Creates parent-child chunk hierarchy from parsed papers.

    Strategy:
    1. Each section becomes a parent chunk (or split if too large)
    2. Each parent is split into child chunks (~400 tokens, 50 overlap)
    3. Abstract gets special treatment as high-value chunk
    4. All chunks linked via parent_id for context retrieval
    """

    def __init__(
        self,
        child_chunk_size: int = 400,
        child_chunk_overlap: int = 50,
        parent_chunk_size: int = 1000,
        min_chunk_size: int = 100,
    ):
        """
        Initialize chunker.

        Args:
            child_chunk_size: Target tokens per child chunk
            child_chunk_overlap: Token overlap between child chunks
            parent_chunk_size: Target tokens per parent chunk
            min_chunk_size: Minimum viable chunk size
        """
        self.child_chunk_size = child_chunk_size
        self.child_chunk_overlap = child_chunk_overlap
        self.parent_chunk_size = parent_chunk_size
        self.min_chunk_size = min_chunk_size

    def chunk_paper(self, paper: ParsedPaper) -> List[Chunk]:
        """
        Create parent-child chunks from a parsed paper.

        Args:
            paper: ParsedPaper object

        Returns:
            List of Chunk objects (parents + children + abstract)
        """
        all_chunks = []

        # 1. Create abstract chunk (high priority)
        if paper.abstract and len(paper.abstract.strip()) >= self.min_chunk_size:
            abstract_chunk = self._create_abstract_chunk(paper)
            all_chunks.append(abstract_chunk)

        # 2. Process each section
        for section in paper.sections:
            if not section.text.strip():
                continue

            section_chunks = self._chunk_section(paper, section)
            all_chunks.extend(section_chunks)

        # 3. Fallback: if no sections found, chunk full text
        if not paper.sections and paper.full_text.strip():
            fallback_chunks = self._chunk_full_text(paper)
            all_chunks.extend(fallback_chunks)

        logger.info(f"Paper {paper.paper_id}: {len(all_chunks)} chunks created")
        return all_chunks

    def _create_abstract_chunk(self, paper: ParsedPaper) -> Chunk:
        """Create special abstract chunk."""
        chunk_id = f"{paper.paper_id}_abstract"
        token_count = self._estimate_tokens(paper.abstract)

        return Chunk(
            chunk_id=chunk_id,
            paper_id=paper.paper_id,
            title=paper.title,
            section="Abstract",
            chunk_type="abstract",
            text=paper.abstract.strip(),
            parent_id=None,
            chunk_index=0,
            token_count=token_count,
            metadata={
                "is_abstract": True,
                "page_range": "1",
            },
        )

    def _chunk_section(self, paper: ParsedPaper, section: PaperSection) -> List[Chunk]:
        """Chunk a single section into parent + children."""
        chunks = []
        text = section.text.strip()

        if not text:
            return chunks

        # Create parent chunk(s) - split if too large
        parent_chunks = self._create_parent_chunks(paper, section)

        for parent in parent_chunks:
            chunks.append(parent)
            # Create child chunks from this parent
            children = self._create_child_chunks(parent)
            chunks.extend(children)

        return chunks

    def _create_parent_chunks(self, paper: ParsedPaper, section: PaperSection) -> List[Chunk]:
        """Split section into parent-sized chunks."""
        text = section.text.strip()
        tokens = self._estimate_tokens(text)

        if tokens <= self.parent_chunk_size:
            # Single parent chunk for this section
            chunk_id = f"{paper.paper_id}_parent_{section.heading[:50].replace(' ', '_')}"
            return [Chunk(
                chunk_id=chunk_id,
                paper_id=paper.paper_id,
                title=paper.title,
                section=section.heading,
                chunk_type="parent",
                text=text,
                parent_id=None,
                chunk_index=0,
                token_count=tokens,
                metadata={
                    "section_level": section.level,
                    "page_start": section.page_start,
                    "page_end": section.page_end,
                },
            )]

        # Split large section into multiple parent chunks
        parents = []
        sentences = self._split_sentences(text)
        current_chunk = []
        current_tokens = 0
        parent_index = 0

        for sent in sentences:
            sent_tokens = self._estimate_tokens(sent)
            if current_tokens + sent_tokens > self.parent_chunk_size and current_chunk:
                # Save current parent
                parent_text = " ".join(current_chunk)
                chunk_id = f"{paper.paper_id}_parent_{section.heading[:30].replace(' ', '_')}_{parent_index}"
                parents.append(Chunk(
                    chunk_id=chunk_id,
                    paper_id=paper.paper_id,
                    title=paper.title,
                    section=f"{section.heading} (part {parent_index + 1})",
                    chunk_type="parent",
                    text=parent_text,
                    parent_id=None,
                    chunk_index=parent_index,
                    token_count=current_tokens,
                    metadata={
                        "section_level": section.level,
                        "page_start": section.page_start,
                        "page_end": section.page_end,
                        "is_continuation": parent_index > 0,
                    },
                ))
                parent_index += 1
                current_chunk = [sent]
                current_tokens = sent_tokens
            else:
                current_chunk.append(sent)
                current_tokens += sent_tokens

        # Don't forget last chunk
        if current_chunk:
            parent_text = " ".join(current_chunk)
            chunk_id = f"{paper.paper_id}_parent_{section.heading[:30].replace(' ', '_')}_{parent_index}"
            parents.append(Chunk(
                chunk_id=chunk_id,
                paper_id=paper.paper_id,
                title=paper.title,
                section=f"{section.heading} (part {parent_index + 1})",
                chunk_type="parent",
                text=parent_text,
                parent_id=None,
                chunk_index=parent_index,
                token_count=current_tokens,
                metadata={
                    "section_level": section.level,
                    "page_start": section.page_start,
                    "page_end": section.page_end,
                    "is_continuation": parent_index > 0,
                },
            ))

        return parents

    def _create_child_chunks(self, parent: Chunk) -> List[Chunk]:
        """Split parent chunk into overlapping child chunks."""
        text = parent.text
        tokens = self._estimate_tokens(text)

        if tokens <= self.child_chunk_size:
            # Parent is small enough, create single child
            child_id = f"{parent.chunk_id}_child_0"
            return [Chunk(
                chunk_id=child_id,
                paper_id=parent.paper_id,
                title=parent.title,
                section=parent.section,
                chunk_type="child",
                text=text,
                parent_id=parent.chunk_id,
                chunk_index=0,
                token_count=tokens,
                metadata={**parent.metadata, "parent_chunk_id": parent.chunk_id},
            )]

        # Split into overlapping children
        sentences = self._split_sentences(text)
        children = []
        current_chunk = []
        current_tokens = 0
        child_index = 0
        overlap_buffer = []

        for sent in sentences:
            sent_tokens = self._estimate_tokens(sent)

            # Check if adding this sentence exceeds child size
            if current_tokens + sent_tokens > self.child_chunk_size and current_chunk:
                # Create child chunk
                child_text = " ".join(current_chunk)
                child_id = f"{parent.chunk_id}_child_{child_index}"
                children.append(Chunk(
                    chunk_id=child_id,
                    paper_id=parent.paper_id,
                    title=parent.title,
                    section=parent.section,
                    chunk_type="child",
                    text=child_text,
                    parent_id=parent.chunk_id,
                    chunk_index=child_index,
                    token_count=current_tokens,
                    metadata={**parent.metadata, "parent_chunk_id": parent.chunk_id},
                ))

                # Prepare overlap for next chunk
                overlap_buffer = self._get_overlap_sentences(current_chunk, self.child_chunk_overlap)
                current_chunk = overlap_buffer + [sent]
                current_tokens = sum(self._estimate_tokens(s) for s in current_chunk)
                child_index += 1
            else:
                current_chunk.append(sent)
                current_tokens += sent_tokens

        # Last chunk
        if current_chunk:
            child_text = " ".join(current_chunk)
            child_id = f"{parent.chunk_id}_child_{child_index}"
            children.append(Chunk(
                chunk_id=child_id,
                paper_id=parent.paper_id,
                title=parent.title,
                section=parent.section,
                chunk_type="child",
                text=child_text,
                parent_id=parent.chunk_id,
                chunk_index=child_index,
                token_count=current_tokens,
                metadata={**parent.metadata, "parent_chunk_id": parent.chunk_id},
            ))

        return children

    def _chunk_full_text(self, paper: ParsedPaper) -> List[Chunk]:
        """Fallback: chunk full text when no sections detected."""
        chunks = []
        text = paper.full_text.strip()
        sentences = self._split_sentences(text)

        current_chunk = []
        current_tokens = 0
        chunk_index = 0

        for sent in sentences:
            sent_tokens = self._estimate_tokens(sent)
            if current_tokens + sent_tokens > self.parent_chunk_size and current_chunk:
                parent_text = " ".join(current_chunk)
                parent_id = f"{paper.paper_id}_parent_fallback_{chunk_index}"
                parent = Chunk(
                    chunk_id=parent_id,
                    paper_id=paper.paper_id,
                    title=paper.title,
                    section="Full Text",
                    chunk_type="parent",
                    text=parent_text,
                    parent_id=None,
                    chunk_index=chunk_index,
                    token_count=current_tokens,
                    metadata={"fallback": True},
                )
                chunks.append(parent)
                # Children
                children = self._create_child_chunks(parent)
                chunks.extend(children)

                chunk_index += 1
                current_chunk = [sent]
                current_tokens = sent_tokens
            else:
                current_chunk.append(sent)
                current_tokens += sent_tokens

        if current_chunk:
            parent_text = " ".join(current_chunk)
            parent_id = f"{paper.paper_id}_parent_fallback_{chunk_index}"
            parent = Chunk(
                chunk_id=parent_id,
                paper_id=paper.paper_id,
                title=paper.title,
                section="Full Text",
                chunk_type="parent",
                text=parent_text,
                parent_id=None,
                chunk_index=chunk_index,
                token_count=current_tokens,
                metadata={"fallback": True},
            )
            chunks.append(parent)
            children = self._create_child_chunks(parent)
            chunks.extend(children)

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences (simple but effective)."""
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        # Filter empty and very short
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

    def _get_overlap_sentences(self, sentences: List[str], overlap_tokens: int) -> List[str]:
        """Get last sentences that fit within overlap token budget."""
        overlap = []
        tokens = 0
        for sent in reversed(sentences):
            sent_tokens = self._estimate_tokens(sent)
            if tokens + sent_tokens > overlap_tokens:
                break
            overlap.insert(0, sent)
            tokens += sent_tokens
        return overlap

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (words * 1.3)."""
        return int(len(text.split()) * 1.3)


def chunk_paper(paper: ParsedPaper, **kwargs) -> List[Chunk]:
    """Convenience function to chunk a single paper."""
    chunker = ParentChildChunker(**kwargs)
    return chunker.chunk_paper(paper)


def chunk_papers(papers: List[ParsedPaper], **kwargs) -> List[Chunk]:
    """Chunk multiple papers."""
    chunker = ParentChildChunker(**kwargs)
    all_chunks = []
    for paper in papers:
        all_chunks.extend(chunker.chunk_paper(paper))
    return all_chunks