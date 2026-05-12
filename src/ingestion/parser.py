from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
from PIL import Image
from pypdf import PdfReader


def parse_uploaded_file(uploaded_file) -> str:
    filename = uploaded_file.name
    suffix = Path(filename).suffix.lower()
    data = uploaded_file.getvalue()

    if suffix == ".pdf":
        return _parse_pdf(data)
    if suffix in {".txt", ".md"}:
        return data.decode("utf-8", errors="ignore")
    if suffix in {".csv"}:
        return _parse_csv(data)
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".tiff"}:
        return _parse_image(data)

    raise ValueError(f"Unsupported file type: {suffix}")


def _parse_pdf(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    pages: list[str] = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(f"[Page {index}]\n{text}")
    return "\n\n".join(pages)


def _parse_csv(data: bytes) -> str:
    frame = pd.read_csv(io.BytesIO(data))
    return frame.to_string(index=False)


def _parse_image(data: bytes) -> str:
    try:
        import pytesseract
    except ImportError as exc:
        raise ValueError(
            "Image OCR needs pytesseract installed. Use PDF/TXT/CSV or install pytesseract."
        ) from exc

    image = Image.open(io.BytesIO(data))
    text = pytesseract.image_to_string(image)
    if not text.strip():
        raise ValueError("No readable text found in image.")
    return text

