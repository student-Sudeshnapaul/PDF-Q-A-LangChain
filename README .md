# PDF Q&A (free stack)

Upload a PDF, ask questions, get answers grounded in the document — a small
RAG (Retrieval-Augmented Generation) app.

**Cost: $0.** Text extraction and embeddings run locally on your machine.
Only the final answer generation calls an API, and it uses Gemini's free tier.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL Streamlit prints (usually http://localhost:8501).

## Get a free Gemini API key

1. Go to https://aistudio.google.com/apikey
2. Sign in with a Google account
3. Click "Create API key" — no credit card required
4. Paste it into the app's sidebar (or set it as an env var before launching:
   `export GOOGLE_API_KEY=...`)

## How it works

1. **Extract**: `pypdf` pulls text out of each page of the uploaded PDF.
2. **Chunk**: the text is split into overlapping ~800-character pieces so the
   model only ever sees relevant snippets, not the whole document.
3. **Embed**: each chunk is converted into a vector using
   `sentence-transformers` (`all-MiniLM-L6-v2`) — this runs entirely on your
   machine, no API call, no cost. The model (~80MB) downloads automatically
   the first time you run the app.
4. **Retrieve**: when you ask a question, it's embedded the same way, and the
   most similar chunks (by cosine similarity) are pulled out — this is a
   simple numpy dot-product search, no external vector database needed.
5. **Answer**: the retrieved chunks plus your question are sent to Gemini,
   with instructions to answer only from that context (reduces hallucination
   and lets it say "not in the document" when appropriate).

Each answer shows an expandable **Sources** section listing which page(s)
the retrieved context came from, so you can verify the answer.

## Limitations

- Scanned/image-only PDFs won't work — there's no text layer to extract.
  You'd need to add OCR (e.g. `pytesseract`) as a preprocessing step.
- Re-uploading the same file re-indexes it unless the settings and file hash
  match a previous run (handled automatically via `st.session_state`).
- Gemini's free tier has rate limits (requests per minute/day) — fine for
  personal use, not for production traffic.
