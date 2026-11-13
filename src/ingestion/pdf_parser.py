"""
PDF Parser for AI Research Papers.

Extracts structured content from research papers using docling (preferred)
with pdfplumber as a fallback. Handles corrupted PDFs gracefully.
"""
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

# Try to import docling first
try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False

import pdfplumber

from src.core.config import config

logger = logging.getLogger(__name__)


@dataclass
class PaperSection:
    """Represents a section of a paper."""
    heading: str
    level: int  # 1=top-level, 2=subsection, etc.
    text: str
    page_start: int
    page_end: int


@dataclass
class ParsedPaper:
    """Container for parsed paper content."""
    paper_id: str
    file_path: str
    file_hash: str
    title: str
    abstract: str
    sections: List[PaperSection]
    full_text: str
    page_count: int
    parsed_at: datetime = field(default_factory=datetime.now)
    parser_used: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "paper_id": self.paper_id,
            "file_path": self.file_path,
            "file_hash": self.file_hash,
            "title": self.title,
            "abstract": self.abstract,
            "sections": [
                {
                    "heading": s.heading,
                    "level": s.level,
                    "text": s.text,
                    "page_start": s.page_start,
                    "page_end": s.page_end,
                }
                for s in self.sections
            ],
            "full_text": self.full_text,
            "page_count": self.page_count,
            "parsed_at": self.parsed_at.isoformat(),
            "parser_used": self.parser_used,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParsedPaper":
        """Create from dictionary."""
        sections = [
            PaperSection(**s) for s in data.get("sections", [])
        ]
        return cls(
            paper_id=data["paper_id"],
            file_path=data["file_path"],
            file_hash=data["file_hash"],
            title=data["title"],
            abstract=data["abstract"],
            sections=sections,
            full_text=data["full_text"],
            page_count=data["page_count"],
            parsed_at=datetime.fromisoformat(data["parsed_at"]),
            parser_used=data.get("parser_used", "unknown"),
            metadata=data.get("metadata", {}),
        )


class PDFParser:
    """
    Parses PDF papers with structure extraction.

    Uses docling for best structure detection (headings, sections, tables)
    Falls back to pdfplumber with heuristic heading detection.
    """

    def __init__(self):
        self.converter = None
        if DOCLING_AVAILABLE:
            try:
                self.converter = DocumentConverter(
                    allowed_formats=[InputFormat.PDF]
                )
                logger.info("Docling initialized successfully")
            except Exception as e:
                logger.warning(f"Docling init failed, will use pdfplumber: {e}")
                self.converter = None
        else:
            logger.info("Docling not available, using pdfplumber")

    def compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file for change detection."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def parse(self, file_path: Path) -> ParsedPaper:
        """
        Parse a PDF file and extract structured content.

        Args:
            file_path: Path to PDF file

        Returns:
            ParsedPaper with title, abstract, sections, full_text

        Raises:
            ValueError: If PDF cannot be parsed
        """
        if not file_path.exists():
            raise ValueError(f"File not found: {file_path}")

        file_hash = self.compute_file_hash(file_path)
        paper_id = file_path.stem

        # Try docling first
        if self.converter:
            try:
                return self._parse_with_docling(file_path, paper_id, file_hash)
            except Exception as e:
                logger.warning(f"Docling parsing failed for {file_path}: {e}, falling back to pdfplumber")

        # Fallback to pdfplumber
        return self._parse_with_pdfplumber(file_path, paper_id, file_hash)

    def _parse_with_docling(self, file_path: Path, paper_id: str, file_hash: str) -> ParsedPaper:
        """Parse using docling for rich structure extraction."""
        result = self.converter.convert(str(file_path))
        doc = result.document

        # Extract title (first heading or from metadata)
        title = self._extract_title_docling(doc, file_path)

        # Extract abstract
        abstract = self._extract_abstract_docling(doc)

        # Extract sections with hierarchy
        sections = self._extract_sections_docling(doc)

        # Full text fallback
        full_text = doc.export_to_markdown()

        page_count = len(doc.pages) if doc.pages else 0

        return ParsedPaper(
            paper_id=paper_id,
            file_path=str(file_path),
            file_hash=file_hash,
            title=title,
            abstract=abstract,
            sections=sections,
            full_text=full_text,
            page_count=page_count,
            parser_used="docling",
        )

    def _extract_title_docling(self, doc, file_path: Path) -> str:
        """Extract title from docling document."""
        # Try to get from first heading
        for element in doc.iterate_items():
            from docling.datamodel.document import DocItemLabel
            if hasattr(element, 'label') and element.label == DocItemLabel.TITLE:
                return element.text.strip()
            if hasattr(element, 'label') and element.label == DocItemLabel.SECTION_HEADER:
                # First section header might be the title
                return element.text.strip()
        # Fallback to filename
        return file_path.stem.replace("_", " ").replace("-", " ").title()

    def _extract_abstract_docling(self, doc) -> str:
        """Extract abstract from docling document."""
        abstract_parts = []
        in_abstract = False

        for element in doc.iterate_items():
            from docling.datamodel.document import DocItemLabel
            if hasattr(element, 'label'):
                label = element.label
                if label == DocItemLabel.SECTION_HEADER:
                    text_lower = element.text.lower().strip()
                    if "abstract" in text_lower and len(text_lower) < 20:
                        in_abstract = True
                        continue
                    elif in_abstract:
                        # Next section header ends abstract
                        break
                elif in_abstract and label in (DocItemLabel.TEXT, DocItemLabel.PARAGRAPH):
                    abstract_parts.append(element.text)

        if abstract_parts:
            return "\n\n".join(abstract_parts).strip()

        # Fallback: first few paragraphs
        first_paras = []
        for element in doc.iterate_items():
            from docling.datamodel.document import DocItemLabel
            if hasattr(element, 'label') and element.label in (DocItemLabel.TEXT, DocItemLabel.PARAGRAPH):
                first_paras.append(element.text)
                if len(first_paras) >= 3:
                    break
        return "\n\n".join(first_paras).strip()

    def _extract_sections_docling(self, doc) -> List[PaperSection]:
        """Extract sections with hierarchy from docling."""
        sections = []
        current_section = None
        current_text = []
        current_level = 0
        page_start = 1

        for element in doc.iterate_items():
            from docling.datamodel.document import DocItemLabel

            if hasattr(element, 'label'):
                label = element.label
                page_no = getattr(element, 'prov', [{}])[0].get('page_no', 1) if hasattr(element, 'prov') else 1

                if label == DocItemLabel.SECTION_HEADER:
                    # Save previous section
                    if current_section and current_text:
                        sections.append(PaperSection(
                            heading=current_section,
                            level=current_level,
                            text="\n\n".join(current_text).strip(),
                            page_start=page_start,
                            page_end=page_no,
                        ))
                    # Start new section
                    current_section = element.text.strip()
                    # Guess level from formatting or numbering
                    current_level = self._guess_heading_level(element.text)
                    current_text = []
                    page_start = page_no
                elif label in (DocItemLabel.TEXT, DocItemLabel.PARAGRAPH, DocItemLabel.LIST_ITEM):
                    if current_section:
                        current_text.append(element.text)

        # Don't forget last section
        if current_section and current_text:
            sections.append(PaperSection(
                heading=current_section,
                level=current_level,
                text="\n\n".join(current_text).strip(),
                page_start=page_start,
                page_end=page_start,
            ))

        return sections

    def _guess_heading_level(self, text: str) -> int:
        """Guess heading level from text patterns."""
        text = text.strip()
        # Numbered headings: 1., 1.1, 1.1.1, etc.
        import re
        match = re.match(r'^(\d+(?:\.\d+)*)', text)
        if match:
            return match.group(1).count('.') + 1
        # All caps = top level
        if text.isupper() and len(text) > 3:
            return 1
        return 2

    def _parse_with_pdfplumber(self, file_path: Path, paper_id: str, file_hash: str) -> ParsedPaper:
        """Parse using pdfplumber with heuristic structure extraction."""
        full_text_parts = []
        sections = []
        abstract_parts = []
        title = file_path.stem.replace("_", " ").replace("-", " ").title()

        with pdfplumber.open(file_path) as pdf:
            page_count = len(pdf.pages)
            current_section = None
            current_text = []
            current_level = 1
            in_abstract = False
            page_start = 1

            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if not text:
                    continue

                full_text_parts.append(text)
                lines = text.split('\n')

                for line in lines:
                    line_stripped = line.strip()
                    if not line_stripped:
                        continue

                    # Detect title (first large text on first page)
                    if page_num == 1 and not title or title == file_path.stem:
                        if len(line_stripped) > 10 and len(line_stripped) < 200:
                            # Heuristic: title is prominent on first page
                            if line_stripped[0].isupper():
                                title = line_stripped

                    # Detect abstract
                    if page_num <= 3 and "abstract" in line_stripped.lower():
                        if len(line_stripped) < 30:  # Likely a header
                            in_abstract = True
                            continue

                    # Detect section headers (heuristic)
                    is_header = self._is_likely_header(line_stripped)
                    if is_header:
                        # Save previous section
                        if current_section and current_text:
                            sections.append(PaperSection(
                                heading=current_section,
                                level=current_level,
                                text="\n".join(current_text).strip(),
                                page_start=page_start,
                                page_end=page_num,
                            ))
                        # Start new section
                        current_section = line_stripped
                        current_level = self._guess_heading_level_pdfplumber(line_stripped)
                        current_text = []
                        page_start = page_num
                        in_abstract = False
                    elif in_abstract:
                        abstract_parts.append(line_stripped)
                    elif current_section:
                        current_text.append(line_stripped)

            # Save last section
            if current_section and current_text:
                sections.append(PaperSection(
                    heading=current_section,
                    level=current_level,
                    text="\n".join(current_text).strip(),
                    page_start=page_start,
                    page_end=page_count,
                ))

        abstract = "\n\n".join(abstract_parts).strip() if abstract_parts else ""
        full_text = "\n\n".join(full_text_parts)

        # If no abstract found, use first section or first paragraphs
        if not abstract and sections:
            abstract = sections[0].text[:2000]

        return ParsedPaper(
            paper_id=paper_id,
            file_path=str(file_path),
            file_hash=file_hash,
            title=title,
            abstract=abstract,
            sections=sections,
            full_text=full_text,
            page_count=page_count,
            parser_used="pdfplumber",
        )

    def _is_likely_header(self, line: str) -> bool:
        """Heuristic to detect section headers."""
        if len(line) > 150:
            return False
        if len(line) < 3:
            return False

        # Numbered sections: 1., 1.1, 2.3.1, etc.
        import re
        if re.match(r'^\d+(\.\d+)*\s+\w', line):
            return True

        # All caps short lines
        if line.isupper() and 3 < len(line) < 80:
            return True

        # Common section names
        common_sections = {
            'introduction', 'related work', 'background', 'methodology',
            'methods', 'approach', 'experiments', 'evaluation', 'results',
            'discussion', 'conclusion', 'future work', 'references',
            'acknowledgments', 'appendix', 'supplementary'
        }
        first_word = line.split()[0].lower().rstrip(':.')
        if first_word in common_sections:
            return True

        return False

    def _guess_heading_level_pdfplumber(self, text: str) -> int:
        """Guess heading level from text."""
        import re
        match = re.match(r'^(\d+(?:\.\d+)*)', text)
        if match:
            return match.group(1).count('.') + 1
        if text.isupper():
            return 1
        return 2


def parse_paper(file_path: Path) -> ParsedPaper:
    """Convenience function to parse a single paper."""
    parser = PDFParser()
    return parser.parse(file_path)


def parse_all_papers(papers_dir: Optional[Path] = None) -> List[ParsedPaper]:
    """Parse all PDFs in the papers directory."""
    papers_dir = papers_dir or config.paths.papers_dir
    parser = PDFParser()
    papers = []

    for pdf_file in papers_dir.glob("*.pdf"):
        try:
            logger.info(f"Parsing: {pdf_file.name}")
            paper = parser.parse(pdf_file)
            papers.append(paper)
        except Exception as e:
            logger.error(f"Failed to parse {pdf_file}: {e}")

    return papers