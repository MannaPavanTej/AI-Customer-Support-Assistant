"""
Document loader module for the Nexora Technologies knowledge base.
Recursively reads markdown (.md) and text (.txt) files and creates Document objects with metadata.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List
import config


@dataclass
class Document:
    """
    Represents a full document from the company knowledge base.

    Attributes:
        title: Extracted or derived document title
        category: 'policies', 'faq', or 'manuals'
        source_file: Relative or filename path
        content: Full text content of the document
    """
    title: str
    category: str
    source_file: str
    content: str


def load_documents(directory: Path = config.KNOWLEDGE_BASE_DIR) -> List[Document]:
    """
    Loads all markdown and text documents from the knowledge base directory.

    Args:
        directory: Path to knowledge_base folder

    Returns:
        List of Document objects
    """
    documents: List[Document] = []
    
    if not directory.exists():
        return documents

    for file_path in directory.rglob("*"):
        if file_path.suffix.lower() in [".md", ".txt"]:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()

                # Extract title from the first header line if available
                lines = [l.strip() for l in text.split("\n") if l.strip()]
                title = lines[0].replace("#", "").strip() if lines else file_path.stem.replace("_", " ").title()

                # Determine category based on parent folder name
                category = file_path.parent.name.lower()
                if category not in ["policies", "faq", "manuals"]:
                    category = "general"

                documents.append(
                    Document(
                        title=title,
                        category=category,
                        source_file=file_path.name,
                        content=text
                    )
                )
            except Exception as e:
                print(f"Error loading document {file_path}: {e}")

    return documents
