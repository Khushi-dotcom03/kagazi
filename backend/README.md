# Kagazi backend

## Run locally

From the project root, activate the virtual environment and start the API:

```powershell
.\venv\Scripts\Activate.ps1
uvicorn main:app --app-dir backend --reload
```

Set `GROQ_API_KEY` in the project-root `.env` before asking questions. The interactive API documentation is available at `http://127.0.0.1:8000/docs`.

## API flow

1. `POST /documents` with a `file` form field containing a PDF. The response returns its `document_id`, page count, and indexed chunk count.
2. `POST /query` with `{ "question": "...", "top_k": 4 }`.
3. The response includes the grounded answer and source metadata (`source`, `page`, `chunk`, and extraction method).

Text PDFs use PyMuPDF. Pages with little selectable text use OCR; scanned PDF support additionally requires the [Tesseract executable](https://github.com/tesseract-ocr/tesseract) installed and available on `PATH`.
