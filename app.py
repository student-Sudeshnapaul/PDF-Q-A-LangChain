import os
import time
import tempfile
import hashlib
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_vectorstore(file_bytes, chunk_size, chunk_overlap):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    documents = []
    skipped_pages = 0
    was_repaired = False
    try:
        try:
        
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
        except Exception:

            try:
                import pikepdf

                repaired_path = tmp_path + ".repaired.pdf"
                with pikepdf.open(tmp_path) as pdf:
                    pdf.save(repaired_path)
                was_repaired = True

                try:
                    loader = PyPDFLoader(repaired_path)
                    documents = loader.load()
                except Exception:
                    documents = []
                finally:
                    if os.path.exists(repaired_path):
                        os.unlink(repaired_path)
            except ImportError:

                documents = []
            except Exception:

                documents = []


            if not documents:
                from pypdf import PdfReader
                from langchain_core.documents import Document

                try:
                    reader = PdfReader(tmp_path, strict=False)
                    documents = []
                    for i, page in enumerate(reader.pages):
                        try:
                            text = page.extract_text() or ""
                        except Exception:
                            text = ""
                        if text.strip():
                            documents.append(Document(page_content=text, metadata={"page": i}))
                        else:
                            skipped_pages += 1
                except Exception as e:
                    raise ValueError(
                        "This PDF's internal structure is too damaged to recover automatically "
                        f"(even after a repair attempt). Underlying error: {type(e).__name__}: {e}"
                    ) from e
    finally:
        os.unlink(tmp_path)

    if not documents:
        raise ValueError("No extractable text found in this PDF (it may be scanned/image-only or corrupted).")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)

    embeddings = load_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore, len(documents), len(chunks), skipped_pages, was_repaired

def page_display(doc):
    """Safely convert a doc's 'page' metadata (0-indexed or missing) to a display string."""
    page = doc.metadata.get("page")
    return str(page + 1) if isinstance(page, int) else "?"


def format_docs(docs):
    return "\n\n".join(f"[Page {page_display(d)}]\n{d.page_content}" for d in docs)


def build_chain(vectorstore, api_key, model_name, temperature, top_k):
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    prompt = ChatPromptTemplate.from_template(
        "Answer the question using ONLY the context below, which was extracted "
        "from a PDF. If the answer isn't in the context, say you couldn't find "
        "it in the document — don't make anything up.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=temperature,
    )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def build_transcript_text(history, filename):
    lines = [f"PDF Q&A Transcript — {filename}", "=" * 50, ""]
    for i, turn in enumerate(reversed(history), start=1):
        lines.append(f"Q{i}: {turn['question']}")
        lines.append(f"A{i}: {turn['answer']}")
        if turn.get("latency"):
            lines.append(f"(generated in {turn['latency']:.2f}s)")
        lines.append("")
    return "\n".join(lines)



st.set_page_config(page_title="PDF Q&A · LangChain", page_icon="🔗", layout="wide")

st.markdown("""
<style>
    :root {
        --bg-color: #0b0e14;
        --card-bg: rgba(18, 24, 38, 0.85);
        --card-bg-soft: rgba(18, 24, 38, 0.55);
        --neon-cyan: #00f3ff;
        --neon-purple: #bc13fe;
        --text-color: #f0f6fc;
        --muted-text: #94a3b8;
        --border-soft: rgba(148, 163, 184, 0.15);
    }

    .stApp {
        background-color: var(--bg-color);
        background-image:
            radial-gradient(at 0% 0%, rgba(0, 243, 255, 0.08) 0px, transparent 50%),
            radial-gradient(at 100% 100%, rgba(188, 19, 254, 0.08) 0px, transparent 50%);
        color: var(--text-color);
    }

    #MainMenu, footer { visibility: hidden; }

    /* Header */
    .tech-header {
        text-align: center;
        padding: 22px 24px;
        background: var(--card-bg);
        border: 1px solid rgba(0, 243, 255, 0.3);
        border-radius: 14px;
        box-shadow: 0 0 20px rgba(0, 243, 255, 0.15), inset 0 0 10px rgba(0, 243, 255, 0.05);
        margin-bottom: 20px;
    }
    .tech-title {
        font-family: 'Courier New', monospace;
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: 2px;
        background: linear-gradient(90deg, var(--neon-cyan), var(--neon-purple));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 12px rgba(0, 243, 255, 0.4);
    }
    .tech-subtitle { color: var(--muted-text); margin-top: 6px; font-size: 0.88rem; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(11, 14, 20, 0.97) !important;
        border-right: 1px solid rgba(0, 243, 255, 0.2) !important;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span { color: #f0f6fc !important; font-weight: 500 !important; }
    .sidebar-section-title {
        color: var(--neon-cyan) !important;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 4px 0 8px 0;
        border-bottom: 1px solid var(--border-soft);
        padding-bottom: 6px;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: rgba(18, 24, 38, 0.9) !important;
        border: 1px dashed rgba(0, 243, 255, 0.4) !important;
        border-radius: 10px !important;
        padding: 10px !important;
    }
    [data-testid="stFileUploader"] label {
        color: #f0f6fc !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        opacity: 1 !important;
    }
    [data-testid="stFileUploaderDropzone"],
    [data-testid="stFileUploader"] section {
        background-color: rgba(24, 32, 50, 0.9) !important;
        border-radius: 8px !important;
    }
    [data-testid="stFileUploaderDropzone"] * ,
    [data-testid="stFileUploader"] section * {
        color: #f0f6fc !important;
    }
    /* The "Browse files" button inside the uploader — Streamlit renders it as a plain white button by default.
       Scoped to the uploader only so it doesn't affect other secondary buttons (e.g. Clear chat history). */
    [data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"] {
        background-color: rgba(0, 243, 255, 0.12) !important;
        border: 1px solid rgba(0, 243, 255, 0.5) !important;
        color: var(--neon-cyan) !important;
        font-weight: 700 !important;
    }
    [data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"]:hover {
        background-color: rgba(0, 243, 255, 0.25) !important;
        border-color: var(--neon-cyan) !important;
        color: #ffffff !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] svg { fill: var(--neon-cyan) !important; }

    /* Uploaded file chip (shows filename + size + remove button after a file is selected).
       Streamlit renders this via a shared "FileChip" component. */
    [data-testid="stFileChips"] { background-color: transparent !important; }
    [data-testid="stFileChip"] {
        background-color: rgba(24, 32, 50, 0.95) !important;
        border: 1px solid rgba(0, 243, 255, 0.3) !important;
        border-radius: 8px !important;
    }
    [data-testid="stFileChip"] * {
        color: #f0f6fc !important;
        opacity: 1 !important;
    }
    [data-testid="stFileChipName"] { color: #f0f6fc !important; }
    [data-testid="stFileChip"] svg { fill: var(--neon-cyan) !important; }
    [data-testid="stFileChipDeleteBtn"] {
        background-color: transparent !important;
        color: var(--muted-text) !important;
    }
    [data-testid="stFileChipDeleteBtn"]:hover { color: #ff5c5c !important; }
    [data-testid="stFileChipDeleteBtn"] svg { fill: currentColor !important; }

    /* Inputs */
    .stTextInput input {
        background-color: rgba(18, 24, 38, 0.9) !important;
        border: 1px solid rgba(0, 243, 255, 0.4) !important;
        color: #ffffff !important;
        border-radius: 8px !important;
    }
    .stTextInput input::placeholder { color: #8b98a5 !important; }
    .stTextInput input:focus {
        border-color: var(--neon-cyan) !important;
        box-shadow: 0 0 12px rgba(0, 243, 255, 0.6) !important;
    }

    div[data-baseweb="slider"] * { color: #ffffff !important; }
    div[data-baseweb="select"] > div {
        background-color: rgba(18, 24, 38, 0.9) !important;
        border: 1px solid rgba(0, 243, 255, 0.3) !important;
    }

    /* Buttons */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, rgba(0,243,255,0.25), rgba(188,19,254,0.25)) !important;
        color: var(--neon-cyan) !important;
        border: 1px solid var(--neon-cyan) !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        transition: all 0.2s ease-in-out !important;
        box-shadow: 0 0 12px rgba(0, 243, 255, 0.3) !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, var(--neon-cyan), var(--neon-purple)) !important;
        color: #000000 !important;
        box-shadow: 0 0 20px rgba(0, 243, 255, 0.8), 0 0 30px rgba(188, 19, 254, 0.6) !important;
    }
    div.stButton > button[kind="secondary"] {
        background-color: rgba(148, 163, 184, 0.08) !important;
        color: var(--muted-text) !important;
        border: 1px solid var(--border-soft) !important;
        border-radius: 8px !important;
    }
    div.stButton > button[kind="secondary"]:hover {
        border-color: rgba(188, 19, 254, 0.5) !important;
        color: var(--text-color) !important;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: var(--card-bg-soft);
        border: 1px solid var(--border-soft);
        border-radius: 10px;
        padding: 12px 16px;
    }
    div[data-testid="stMetric"] label { color: var(--muted-text) !important; }

    /* Q&A cards */
    .qa-card {
        background: var(--card-bg);
        border: 1px solid rgba(188, 19, 254, 0.4);
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0 0 15px rgba(188, 19, 254, 0.15);
        margin-bottom: 14px;
    }
    .qa-question {
        color: var(--neon-cyan);
        font-weight: 700;
        font-size: 1.02rem;
        margin-bottom: 8px;
        display: flex;
        gap: 8px;
        align-items: baseline;
    }
    .qa-answer { font-size: 1.02rem; line-height: 1.65; color: var(--text-color); }
    .qa-meta { color: var(--muted-text); font-size: 0.82rem; margin-top: 10px; }

    .status-glow {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 0.88rem;
        background: rgba(0, 243, 255, 0.12);
        border: 1px solid var(--neon-cyan);
        color: var(--neon-cyan);
        box-shadow: 0 0 10px rgba(0, 243, 255, 0.3);
    }

    .empty-state {
        text-align: center;
        padding: 60px 20px;
        color: var(--muted-text);
        border: 1px dashed var(--border-soft);
        border-radius: 14px;
        background: var(--card-bg-soft);
    }
    .empty-state-icon { font-size: 2.4rem; margin-bottom: 10px; }

    .source-chunk {
        background: rgba(255,255,255,0.03);
        border-left: 2px solid var(--neon-purple);
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 0.85rem;
        color: var(--muted-text);
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="tech-header">
    <div class="tech-title">🔗 LANGCHAIN PDF Q&A</div>
    <div class="tech-subtitle">Local embeddings + Gemini free tier · FAISS retrieval-augmented generation</div>
</div>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("<div class='sidebar-section-title'>API Config</div>", unsafe_allow_html=True)
    st.caption("Get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)")
    api_key = st.text_input("API key", type="password", value=os.environ.get("GOOGLE_API_KEY", ""))
    model_name = st.selectbox("Gemini model",["gemini-3.5-flash-lite"])
    if api_key:
        st.markdown("<span class='status-glow'>key set</span>", unsafe_allow_html=True)
    else:
        st.caption("⚠️No API key yet — you can still index a PDF, but asking questions needs one.")

    st.markdown("<div class='sidebar-section-title' style='margin-top:20px;'>🎛️ Indexing</div>", unsafe_allow_html=True)
    chunk_size = st.slider("Chunk size (characters)", 300, 1500, 800, 50)
    chunk_overlap = st.slider("Chunk overlap (characters)", 0, 400, 150, 25)

    st.markdown("<div class='sidebar-section-title' style='margin-top:20px;'>🎯 Retrieval & Generation</div>", unsafe_allow_html=True)
    top_k = st.slider("Chunks retrieved per question", 2, 8, 4)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

    if st.session_state.get("chat_history"):
        st.markdown("<div class='sidebar-section-title' style='margin-top:20px;'>📜 Session</div>", unsafe_allow_html=True)
        avg_latency = sum(t["latency"] for t in st.session_state.chat_history if t.get("latency")) / max(
            1, len([t for t in st.session_state.chat_history if t.get("latency")])
        )
        st.caption(f"{len(st.session_state.chat_history)} question(s) asked · avg {avg_latency:.2f}s/answer")
        if st.button("Clear chat history", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()

    needs_processing = (
        "file_hash" not in st.session_state
        or st.session_state.file_hash != file_hash
        or st.session_state.get("chunk_size") != chunk_size
        or st.session_state.get("chunk_overlap") != chunk_overlap
    )

    if needs_processing:
        with st.spinner("Loading PDF, splitting, and building FAISS index..."):
            try:
                vectorstore, page_count, chunk_count, skipped_pages, was_repaired = build_vectorstore(
                    file_bytes, chunk_size, chunk_overlap
                )
                st.session_state.file_hash = file_hash
                st.session_state.file_name = uploaded_file.name
                st.session_state.chunk_size = chunk_size
                st.session_state.chunk_overlap = chunk_overlap
                st.session_state.vectorstore = vectorstore
                st.session_state.page_count = page_count
                st.session_state.chunk_count = chunk_count
                st.session_state.skipped_pages = skipped_pages
                st.session_state.was_repaired = was_repaired
                st.session_state.chat_history = []
            except Exception as e:
                st.error(f"Couldn't process this PDF: {e}")
                st.stop()

    if st.session_state.get("was_repaired"):
        st.info("🔧 This PDF had a malformed internal structure — it was automatically repaired before indexing.")
    if st.session_state.get("skipped_pages"):
        st.warning(
            f"{st.session_state.skipped_pages} page(s) had no extractable text (scanned image or "
            f"corrupted content) and were skipped. The rest of the document was indexed normally."
        )

    m1, m2, m3 = st.columns(3)
    m1.metric("File", st.session_state.file_name[:24] + ("…" if len(st.session_state.file_name) > 24 else ""))
    m2.metric("Pages", st.session_state.page_count)
    m3.metric("Chunks indexed", st.session_state.chunk_count)

    st.markdown("<div style='height: 6px'></div>", unsafe_allow_html=True)

    col_q, col_btn = st.columns([5, 1])
    with col_q:
        question = st.text_input(
            "Ask a question about the PDF",
            placeholder="e.g., What are the key points in section 2?",
            label_visibility="collapsed",
        )
    with col_btn:
        ask = st.button("Ask ▸", type="primary", use_container_width=True)

    if ask and question.strip():
        if not api_key:
            st.warning("Enter your Gemini API key in the sidebar first.")
        else:
            with st.spinner("Retrieving vector context & generating response..."):
                chain, retriever = build_chain(
                    st.session_state.vectorstore, api_key, model_name, temperature, top_k
                )
                try:
                    start = time.perf_counter()
                    answer = chain.invoke(question)
                    latency = time.perf_counter() - start
                    sources = retriever.invoke(question)
                except Exception as e:
                    msg = str(e)
                    if "429" in msg or "quota" in msg.lower() or "rate" in msg.lower():
                        st.error("Rate limit hit on the Gemini free tier. Wait a bit and try again.")
                    else:
                        st.error(f"Request failed: {msg}")
                    answer, latency, sources = None, None, []

            if answer:
                st.session_state.chat_history.append(
                    {"question": question, "answer": answer, "latency": latency, "sources": sources}
                )
                st.rerun()
    elif ask:
        st.warning("Type a question first.")

    history = st.session_state.get("chat_history", [])

    if history:
        top_row_l, top_row_r = st.columns([4, 1])
        with top_row_l:
            st.markdown(f"#### Conversation ({len(history)})")
        with top_row_r:
            st.download_button(
                "Export",
                data=build_transcript_text(history, st.session_state.file_name),
                file_name=f"{st.session_state.file_name.rsplit('.', 1)[0]}_qa_transcript.txt",
                mime="text/plain",
                use_container_width=True,
            )

        for i, turn in enumerate(reversed(history)):
            st.markdown(f"""
            <div class="qa-card">
                <div class="qa-question"> {turn['question']}</div>
                <div class="qa-answer">{turn['answer']}</div>
                <div class="qa-meta">{f"{turn['latency']:.2f}s" if turn.get('latency') else '—'} · 📚 {len(turn['sources'])} source chunk(s)</div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander(f"View sources for this answer"):
                for doc in turn["sources"]:
                    text = doc.page_content
                    snippet = text[:400] + ("…" if len(text) > 400 else "")
                    st.markdown(
                        f"<div class='source-chunk'><b>Page {page_display(doc)}</b><br>{snippet}</div>",
                        unsafe_allow_html=True,
                    )
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">💬</div>
            No questions asked yet — type one above to get started.
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-state-icon">📄</div>
        <b>Upload a PDF to get started</b><br>
        <span style="font-size: 0.85rem;">It'll be split into chunks and indexed locally with FAISS — no data leaves your machine except the final question sent to Gemini.</span>
    </div>
    """, unsafe_allow_html=True)
