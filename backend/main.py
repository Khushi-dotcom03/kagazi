import re
import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from embeddings.embedder import embed_text
from ingestion.pdf_loader import extract_pages
from preprocessing.chunker import chunk_text
from rag.pipeline import rag_graph
from vectorstore.chroma_store import create_collection, replace_document

app = FastAPI(title="Kagazi API", version="0.1.0")
UPLOAD_FOLDER = Path(__file__).resolve().parent / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)


class QueryRequest(BaseModel):
    question: str = Field(min_length=3)
    user_id: str
    top_k: int = Field(default=4, ge=1, le=10)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/documents", status_code=201)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename or Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    safe_name = re.sub(
        r"[^A-Za-z0-9._-]",
        "_",
        Path(file.filename).name
    )

    document_id = uuid4().hex
    version = 1

    path = UPLOAD_FOLDER / f"{document_id}_{safe_name}"

    with path.open("wb") as destination:
        shutil.copyfileobj(file.file, destination)

    try:
        pages = extract_pages(path)

        chunks = []

        for page in pages:
            page_chunks = chunk_text(page["text"])

            for piece in page_chunks:
                chunk_id = f"{document_id}_chunk_{len(chunks)}"

                chunks.append({
                    "text": piece,
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "source": safe_name,
                    "page": page["page"],
                    "version": version,
                    "method": page["method"],
                    "source_type": "pdf",
                })

        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="No readable text was found in this PDF."
            )

        replace_document(
            create_collection(),
            document_id,
            chunks,
            embed_text([chunk["text"] for chunk in chunks])
        )

    except HTTPException:
        path.unlink(missing_ok=True)
        raise

    except Exception as error:
        path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422,
            detail=str(error)
        ) from error

    return {
        "document_id": document_id,
        "filename": safe_name,
        "version": version,
        "pages": len(pages),
        "chunks": len(chunks)
    }


@app.post("/query")
def query_documents(request: QueryRequest):
    try:
        result = rag_graph.invoke({"question": request.question, "user_id": request.user_id, "top_k": request.top_k})
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {"answer": result["answer"], "sources": [match["citation"] | {"distance": match["distance"]} for match in result["matches"]]}
