"""PDF ingestion and text normalization service.

This module is deliberately isolated from the extraction engine: its only
responsibility is turning raw bytes on disk (or in memory) into clean,
LLM-ready plain text plus lightweight document metadata. Keeping parsing
concerns separate from extraction concerns means we can swap in OCR-based
parsing for scanned documents, or a different PDF backend, without touching
a single line of the extraction/self-healing logic.
"""

from __future__ import annotations

import io
import re
import unicodedata
from dataclasses import dataclass

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from core.config import Settings, get_settings
from core.logger import get_logger

logger = get_logger(__name__)


class DocumentParsingError(Exception):
    """Raised when a source document cannot be parsed into usable text.

    Covers corrupted/encrypted PDFs, page-count limits being exceeded, and
    documents that parse successfully but yield no extractable text (e.g.
    pure image scans with no embedded text layer).
    """


@dataclass(slots=True, frozen=True)
class ParsedDocument:
    """Result of parsing a source document into normalized plain text.

    Attributes:
        text: The cleaned, concatenated plain text of the document, with
            page boundaries marked for LLM context.
        page_count: Number of pages present in the source document.
        character_count: Length of the final cleaned text, in characters.
        title: Document title extracted from PDF metadata, if present.
        author: Document author extracted from PDF metadata, if present.
        was_truncated: ``True`` if the document exceeded configured limits
            and was truncated before returning.
    """

    text: str
    page_count: int
    character_count: int
    title: str | None
    author: str | None
    was_truncated: bool


def _normalize_whitespace(raw_text: str) -> str:
    """Collapses redundant whitespace while preserving paragraph structure.

    PDF text extraction frequently introduces irregular spacing, stray
    control characters, and inconsistent line breaks caused by column
    layouts or kerning artifacts. This normalizes the text into a dense,
    LLM-friendly form without destroying semantically meaningful line
    breaks (e.g. table rows).

    Args:
        raw_text: The raw text extracted from a single PDF page.

    Returns:
        Normalized text with collapsed horizontal whitespace and stripped
        non-printable control characters.
    """
    # Normalize unicode (e.g. ligatures, full-width characters) to a
    # canonical composed form for consistent downstream tokenization.
    text = unicodedata.normalize("NFKC", raw_text)

    # Strip non-printable control characters except newlines/tabs.
    text = "".join(
        char for char in text if char in "\n\t" or not unicodedata.category(char).startswith("C")
    )

    # Collapse runs of horizontal whitespace (but not newlines) to a single space.
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse 3+ consecutive newlines down to a paragraph-sized double break.
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Trim trailing whitespace on each line.
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()


class DocumentParserService:
    """Parses uploaded PDF documents into clean, normalized text.

    This service enforces configured safety limits (max page count, max
    output character count) so that a single pathological or maliciously
    crafted PDF cannot exhaust memory or blow the extraction engine's token
    budget.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Initializes the parser with the given (or default) settings.

        Args:
            settings: Optional explicit settings instance, primarily for
                dependency injection in tests.
        """
        self._settings = settings or get_settings()

    def parse_pdf_bytes(self, file_bytes: bytes, *, source_filename: str) -> ParsedDocument:
        """Parses raw PDF bytes into normalized plain text with metadata.

        Args:
            file_bytes: The raw binary content of the uploaded PDF file.
            source_filename: Original filename, used only for logging
                context.

        Returns:
            A ``ParsedDocument`` containing the cleaned text and metadata.

        Raises:
            DocumentParsingError: If the PDF is corrupted, encrypted without
                a usable password, exceeds the configured page limit, or
                yields no extractable text content.
        """
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
        except PdfReadError as exc:
            logger.error(
                "pdf_parse_failed_corrupted",
                extra={"source_filename": source_filename, "error": str(exc)},
            )
            raise DocumentParsingError(
                f"The file '{source_filename}' could not be read as a valid "
                "PDF. It may be corrupted or not a PDF at all."
            ) from exc

        if reader.is_encrypted:
            try:
                # Attempt a blank-password decrypt, which succeeds for PDFs
                # that only have owner-level (permissions) encryption rather
                # than a true user password.
                decrypt_result = reader.decrypt("")
            except Exception as exc:  # pypdf raises broad exceptions here
                raise DocumentParsingError(
                    f"The file '{source_filename}' is password-protected "
                    "and could not be decrypted."
                ) from exc

            if decrypt_result == 0:
                raise DocumentParsingError(
                    f"The file '{source_filename}' is password-protected "
                    "and could not be decrypted."
                )

        page_count = len(reader.pages)
        if page_count == 0:
            raise DocumentParsingError(
                f"The file '{source_filename}' contains zero pages."
            )
        if page_count > self._settings.max_pdf_pages:
            raise DocumentParsingError(
                f"The file '{source_filename}' has {page_count} pages, "
                f"exceeding the configured limit of "
                f"{self._settings.max_pdf_pages} pages."
            )

        page_texts: list[str] = []
        for page_index, page in enumerate(reader.pages, start=1):
            try:
                raw_page_text = page.extract_text() or ""
            except Exception as exc:  # defensive: pypdf can raise on malformed content streams
                logger.warning(
                    "pdf_page_extraction_failed",
                    extra={
                        "source_filename": source_filename,
                        "page_number": page_index,
                        "error": str(exc),
                    },
                )
                raw_page_text = ""

            cleaned_page_text = _normalize_whitespace(raw_page_text)
            if cleaned_page_text:
                page_texts.append(f"[Page {page_index}]\n{cleaned_page_text}")

        if not page_texts:
            raise DocumentParsingError(
                f"The file '{source_filename}' contains no extractable text. "
                "It may be a scanned image without an OCR text layer, which "
                "this service does not currently support."
            )

        full_text = "\n\n".join(page_texts)
        was_truncated = False
        if len(full_text) > self._settings.max_text_input_chars:
            full_text = full_text[: self._settings.max_text_input_chars]
            was_truncated = True
            logger.warning(
                "pdf_text_truncated",
                extra={
                    "source_filename": source_filename,
                    "limit_chars": self._settings.max_text_input_chars,
                },
            )

        metadata = reader.metadata
        title = str(metadata.title) if metadata and metadata.title else None
        author = str(metadata.author) if metadata and metadata.author else None

        logger.info(
            "pdf_parsed_successfully",
            extra={
                "source_filename": source_filename,
                "page_count": page_count,
                "character_count": len(full_text),
                "was_truncated": was_truncated,
            },
        )

        return ParsedDocument(
            text=full_text,
            page_count=page_count,
            character_count=len(full_text),
            title=title,
            author=author,
            was_truncated=was_truncated,
        )

    def clean_raw_text(self, raw_text: str) -> str:
        """Normalizes plain-text input submitted directly (non-PDF path).

        Args:
            raw_text: Raw text submitted by the caller.

        Returns:
            Normalized text, truncated to the configured character limit if
            necessary.

        Raises:
            DocumentParsingError: If the input is empty after normalization.
        """
        cleaned = _normalize_whitespace(raw_text)
        if not cleaned:
            raise DocumentParsingError("Submitted text is empty after normalization.")
        if len(cleaned) > self._settings.max_text_input_chars:
            cleaned = cleaned[: self._settings.max_text_input_chars]
        return cleaned
