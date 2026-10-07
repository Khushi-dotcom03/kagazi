def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    """Split text on readable boundaries with modest context overlap."""

    text = " ".join(text.split())

    if not text:
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    
    chunks, start = [], 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = max(text.rfind(". ", start, end), text.rfind(" ", start, end))
            if boundary > start + chunk_size // 2:
                end = boundary + 1
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = end - overlap

    return chunks
