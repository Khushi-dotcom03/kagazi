"""PDF extraction with OCR fallback for scanned pages."""

from pathlib import Path

import pymupdf


def extract_pages(pdf_path: str | Path, ocr_min_characters: int = 30) -> list[dict]:
    """Return extracted text and its method for every PDF page."""
    document = pymupdf.open(str(pdf_path))
    pages: list[dict] = []

    try:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()
            method = "text"

            if len(text) < ocr_min_characters:
                text = _ocr_page(page)
                method = "ocr"

            pages.append({"page": page_number, "text": text, "method": method})
    finally:
        document.close()

    return pages


def extract_text(pdf_path: str | Path) -> dict:
    """Return a document-level result for older callers."""
    pages = extract_pages(pdf_path)
    text = "\n\n".join(page["text"] for page in pages if page["text"])
    return {
        "text": text,
        "pages": len(pages),
        "characters": len(text),
        "page_data": pages,
    }


def _ocr_page(page: pymupdf.Page) -> str:
    try:
        import pytesseract
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(
            "This scanned PDF needs OCR. Install pytesseract, Pillow, and Tesseract."
        ) from error

    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    return pytesseract.image_to_string(image).strip()

