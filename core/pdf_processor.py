"""
PDF Ingestion & Page-to-Image Rendering Engine.
Utilizes PyMuPDF (fitz) for high-fidelity rendering, page analysis, and digital text extraction.
"""

from typing import List, Dict, Any, Union, Optional
import io
import pymupdf
from PIL import Image


class PDFProcessor:
    """Handles PDF parsing, page rendering to high-DPI images, and metadata inspection."""

    def __init__(self, pdf_source: Optional[Union[str, bytes]] = None):
        self.doc: Optional[pymupdf.Document] = None
        self.source = pdf_source
        if pdf_source is not None:
            self.load_pdf(pdf_source)

    def load_pdf(self, pdf_source: Union[str, bytes]) -> None:
        """Loads a PDF from file path or raw bytes."""
        self.close()
        self.source = pdf_source
        if isinstance(pdf_source, str):
            self.doc = pymupdf.open(pdf_source)
        elif isinstance(pdf_source, bytes):
            self.doc = pymupdf.open(stream=pdf_source, filetype="pdf")
        else:
            raise TypeError("pdf_source must be a file path string or bytes.")

    @property
    def page_count(self) -> int:
        """Returns total number of pages."""
        return len(self.doc) if self.doc else 0

    def get_page_info(self, page_number: int) -> Dict[str, Any]:
        """
        Inspects page metadata, bounding rect, embedded image count,
        and determines if the page is primarily scanned or digital text.
        page_number is 1-indexed.
        """
        if not self.doc:
            raise ValueError("No PDF document is currently loaded.")
        if page_number < 1 or page_number > len(self.doc):
            raise IndexError(f"Page number {page_number} out of range (1..{len(self.doc)})")

        page = self.doc[page_number - 1]
        rect = page.rect
        text = page.get_text("text").strip()
        images = page.get_images()

        # If digital text is very sparse (< 30 chars) and page has images or large areas, classify as scanned
        is_scanned = len(text) < 30 and len(images) > 0

        return {
            "page_number": page_number,
            "width": rect.width,
            "height": rect.height,
            "digital_text_len": len(text),
            "image_count": len(images),
            "is_scanned": is_scanned,
            "preview_snippet": text[:100] if text else "",
        }

    def render_page_to_image(self, page_number: int, dpi: int = 200) -> Image.Image:
        """
        Renders specified PDF page (1-indexed) to a high-resolution PIL Image.
        Default 200 DPI gives optimal quality for OCR.
        """
        if not self.doc:
            raise ValueError("No PDF document is currently loaded.")
        if page_number < 1 or page_number > len(self.doc):
            raise IndexError(f"Page number {page_number} out of range (1..{len(self.doc)})")

        page = self.doc[page_number - 1]
        # Standard PDF DPI is 72, zoom factor = dpi / 72
        zoom = dpi / 72.0
        matrix = pymupdf.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img_bytes = pix.tobytes("png")
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")

    def extract_direct_text(self, page_number: int) -> str:
        """Extracts direct digital text from the page (1-indexed)."""
        if not self.doc:
            raise ValueError("No PDF document is currently loaded.")
        if page_number < 1 or page_number > len(self.doc):
            raise IndexError(f"Page number {page_number} out of range (1..{len(self.doc)})")

        page = self.doc[page_number - 1]
        return page.get_text("text")

    def extract_all_direct_text(self) -> str:
        """Extracts and concatenates direct text from all pages."""
        if not self.doc:
            return ""
        return "\n\n".join([page.get_text("text") for page in self.doc])

    def render_all_pages(self, dpi: int = 150) -> List[Image.Image]:
        """Renders all pages into PIL Images."""
        return [self.render_page_to_image(i + 1, dpi=dpi) for i in range(len(self.doc))]

    def close(self) -> None:
        """Closes open document handles."""
        if self.doc:
            try:
                self.doc.close()
            except Exception:
                pass
            self.doc = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
