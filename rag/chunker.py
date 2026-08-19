"""
Document Chunker module.
Splits knowledge base documents into semantic chunks by section headers and paragraphs.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
from .document_loader import Document
import re


@dataclass
class Chunk:
    """
    Represents a single retrieval chunk.

    Attributes:
        chunk_id: Unique string identifier (e.g. 'refund_policy_chunk_1')
        document_title: Parent document title
        category: 'policies', 'faq', or 'manuals'
        source_file: Name of the original file
        section_title: Header/section title this chunk belongs to
        text: Text content of the chunk
    """
    chunk_id: str
    document_title: str
    category: str
    source_file: str
    section_title: str
    text: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk object to dictionary format for storage/display."""
        return {
            "chunk_id": self.chunk_id,
            "document_title": self.document_title,
            "category": self.category,
            "source_file": self.source_file,
            "section_title": self.section_title,
            "text": self.text
        }


def chunk_documents(documents: List[Document], max_chunk_chars: int = 500) -> List[Chunk]:
    """
    Splits documents into semantic chunks based on markdown section headers (##).

    Args:
        documents: List of loaded Document objects
        max_chunk_chars: Maximum character length for a single chunk

    Returns:
        List of structured Chunk objects
    """
    chunks: List[Chunk] = []

    for doc in documents:
        # Split by markdown headers (e.g. ## Section Name)
        sections = re.split(r'\n(?=##\s+)', doc.content)
        chunk_idx = 1

        for section in sections:
            clean_sec = section.strip()
            if not clean_sec:
                continue

            # Extract section title if present
            lines = clean_sec.split("\n")
            if lines[0].startswith("##"):
                section_title = lines[0].replace("##", "").strip()
                body = "\n".join(lines[1:]).strip()
            else:
                section_title = doc.title
                body = clean_sec

            if not body:
                body = clean_sec

            # Create chunk
            chunk_id = f"{doc.source_file.replace('.', '_')}_{chunk_idx}"
            chunk_text = f"Document: {doc.title}\nSection: {section_title}\n\n{body}"

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_title=doc.title,
                    category=doc.category,
                    source_file=doc.source_file,
                    section_title=section_title,
                    text=chunk_text
                )
            )
            chunk_idx += 1

    return chunks
