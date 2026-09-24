import os
import re
from typing import List, Dict, Any, Optional

class ResumeChunker:
    """
    Section-aware resume chunker supporting .pdf, .docx, and .txt files.
    Preserves logical resume sections (Summary, Experience, Education, Skills, etc.)
    rather than naive fixed-character splitting.
    """
    
    DEFAULT_SECTIONS = [
        "SUMMARY", "OBJECTIVE", "PROFESSIONAL SUMMARY", "EXECUTIVE SUMMARY",
        "WORK EXPERIENCE", "EXPERIENCE", "EMPLOYMENT HISTORY", "CAREER HISTORY",
        "EDUCATION", "ACADEMIC BACKGROUND",
        "SKILLS", "TECHNICAL SKILLS", "CORE COMPETENCIES", "AREAS OF EXPERTISE",
        "PROJECTS", "PERSONAL PROJECTS", "KEY PROJECTS",
        "CERTIFICATIONS", "LICENSES & CERTIFICATIONS", "HONORS & AWARDS"
    ]
    
    def __init__(self, section_keywords: Optional[List[str]] = None):
        self.section_keywords = section_keywords or self.DEFAULT_SECTIONS
        # Regex to detect section headers (e.g. "=== WORK EXPERIENCE ===", "WORK EXPERIENCE:", "1. Experience", "Work Experience")
        patterns = [
            r"^(?:==+|\#\#+|\*\*)\s*(.*?)\s*(?:==+|\*\*|$)",
            r"^([A-Z\s&/]{3,30}):?\s*$"
        ]
        self.header_regex = re.compile("|".join(patterns), re.IGNORECASE | re.MULTILINE)

    def extract_text_from_file(self, file_path: str) -> str:
        """Extract raw text based on file extension (.pdf, .docx, .txt)."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        elif ext == ".pdf":
            return self._extract_text_from_pdf(file_path)
        elif ext == ".docx":
            return self._extract_text_from_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using PyMuPDF (fitz) or pdfplumber fallback."""
        text = ""
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            for page in doc:
                text += page.get_text() + "\n"
            return text
        except ImportError:
            pass

        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except ImportError:
            pass

        raise RuntimeError("Neither PyMuPDF (fitz) nor pdfplumber is available to parse PDF files.")

    def _extract_text_from_docx(self, docx_path: str) -> str:
        """Extract text from DOCX file using python-docx."""
        try:
            import docx
            doc = docx.Document(docx_path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n".join(paragraphs)
        except ImportError:
            raise RuntimeError("python-docx is not installed to parse .docx files.")

    def chunk_resume(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Ingests a resume file and splits it into section-aware chunks.
        Returns a list of dicts: [{'section_name': str, 'content': str, 'chunk_index': int}]
        """
        raw_text = self.extract_text_from_file(file_path)
        filename = os.path.basename(file_path)
        resume_id = os.path.splitext(filename)[0]

        lines = raw_text.splitlines()
        chunks = []
        current_section = "GENERAL"
        current_lines = []

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            # Check if line matches a known section header
            matched_header = self._detect_section_header(line_str)
            if matched_header:
                if current_lines:
                    content = "\n".join(current_lines).strip()
                    if content:
                        chunks.append({
                            "resume_id": resume_id,
                            "section_name": current_section,
                            "content": content,
                            "chunk_index": len(chunks)
                        })
                    current_lines = []
                current_section = matched_header
            else:
                current_lines.append(line_str)

        # Append last section
        if current_lines:
            content = "\n".join(current_lines).strip()
            if content:
                chunks.append({
                    "resume_id": resume_id,
                    "section_name": current_section,
                    "content": content,
                    "chunk_index": len(chunks)
                })

        # Fallback: If no structured sections were found, chunk by paragraph groups (~300-500 chars)
        if len(chunks) <= 1 and raw_text:
            chunks = self._fallback_paragraph_chunking(raw_text, resume_id)

        return chunks

    def _detect_section_header(self, line: str) -> Optional[str]:
        """Check if a single line represents a section header."""
        cleaned = re.sub(r"^[=\-*#\d.\s]+|[=\-*#\s]+$", "", line).strip().upper()
        for kw in self.section_keywords:
            if cleaned == kw or cleaned.startswith(kw + ":") or cleaned == kw + "S":
                return kw
        return None

    def _fallback_paragraph_chunking(self, raw_text: str, resume_id: str) -> List[Dict[str, Any]]:
        """Fallback chunker for unformatted/headerless text."""
        paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
        chunks = []
        for idx, p in enumerate(paragraphs):
            chunks.append({
                "resume_id": resume_id,
                "section_name": "CONTENT",
                "content": p,
                "chunk_index": idx
            })
        if not chunks:
            chunks.append({
                "resume_id": resume_id,
                "section_name": "CONTENT",
                "content": raw_text.strip(),
                "chunk_index": 0
            })
        return chunks
