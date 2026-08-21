# 🔗 PDF Q&A — LangChain RAG

> An AI-powered PDF Question Answering application built with **LangChain, FAISS, HuggingFace Embeddings, Google Gemini, and Streamlit**.

Upload a PDF, ask questions about its contents, and receive context-aware answers using a **Retrieval-Augmented Generation (RAG)** pipeline.

---

## 🚀 Overview

**PDF Q&A — LangChain** allows users to interact with PDF documents using natural language.

Instead of sending the entire document to an LLM, the application:

1. Extracts text from the PDF
2. Splits the document into smaller chunks
3. Converts chunks into vector embeddings
4. Stores the embeddings in a FAISS vector database
5. Retrieves the most relevant chunks for a question
6. Sends only the relevant context to Google Gemini
7. Generates a grounded answer based on the retrieved content

This approach improves relevance, reduces unnecessary context, and helps minimize hallucinations.

---

## ✨ Key Features

* 📄 Upload and process PDF documents
* 🤖 AI-powered question answering
* 🔍 Retrieval-Augmented Generation (RAG)
* 🧠 HuggingFace sentence embeddings
* ⚡ FAISS vector similarity search
* 💬 Google Gemini integration
* 📚 Source chunk display for generated answers
* 🎛️ Configurable chunk size and overlap
* 🎯 Configurable Top-K retrieval
* 🌡️ Adjustable LLM temperature
* 📊 PDF page and indexed-chunk statistics
* 🛠️ Automatic handling of malformed PDFs
* 🔄 Fallback PDF extraction for partially corrupted documents
* 💾 Conversation history
* 📥 Export Q&A conversations as TXT
* 🌐 Streamlit-based interactive UI
* 🔐 API key entered securely through the interface

---

## 🏗️ System Architecture

```text
                         ┌──────────────────┐
                         │    PDF Upload    │
                         └─────────┬────────┘
                                   │
                                   ▼
                         ┌──────────────────┐
                         │   PDF Extraction │
                         │    PyPDFLoader   │
                         └─────────┬────────┘
                                   │
                                   ▼
                     ┌──────────────────────────┐
                     │     Text Chunking        │
                     │ RecursiveCharacterText   │
                     │       Splitter           │
                     └────────────┬─────────────┘
                                  │
                                  ▼
                     ┌──────────────────────────┐
                     │    HuggingFace           │
                     │    Embeddings             │
                     │ all-MiniLM-L6-v2         │
                     └────────────┬─────────────┘
                                  │
                                  ▼
                     ┌──────────────────────────┐
                     │      FAISS Vector        │
                     │         Store            │
                     └────────────┬─────────────┘
                                  │
                                  │ Similarity Search
                                  ▼
User Question ───────► ┌────────────────────────┐
                       │      Retriever         │
                       │       Top-K Chunks     │
                       └────────────┬───────────┘
                                    │
                                    ▼
                       ┌────────────────────────┐
                       │   Context + Question   │
                       └────────────┬───────────┘
                                    │
                                    ▼
                       ┌────────────────────────┐
                       │     Google Gemini      │
                       │         LLM            │
                       └────────────┬───────────┘
                                    │
                                    ▼
                       ┌────────────────────────┐
                       │    Grounded Answer     │
                       │    + Source Chunks     │
                       └────────────────────────┘
```

---

## 🧠 RAG Pipeline

The project follows a standard Retrieval-Augmented Generation architecture.

### 1. Document Ingestion

The user uploads a PDF through the Streamlit interface.

```text
PDF → Temporary File → PDF Loader
```

The application uses LangChain's `PyPDFLoader` to extract document content.

---

### 2. Text Splitting

Large documents are divided into smaller chunks using:

```text
RecursiveCharacterTextSplitter
```

The application provides configurable:

* Chunk size
* Chunk overlap

This allows the retrieval pipeline to be tuned for different document types.

---

### 3. Embedding Generation

Each document chunk is converted into a numerical vector using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

These embeddings represent the semantic meaning of the document chunks.

---

### 4. Vector Storage

The generated embeddings are stored in:

```text
FAISS
```

FAISS enables efficient similarity-based retrieval of relevant document chunks.

---

### 5. Retrieval

When the user asks a question, the application searches the FAISS vector store and retrieves the most relevant chunks.

The number of retrieved chunks is configurable through the **Top-K** parameter.

---

### 6. Prompt Construction

The retrieved chunks are combined with the user's question.

The model is instructed to answer using only the retrieved PDF context.

```text
Retrieved Context
        +
User Question
        ↓
Prompt
```

This helps reduce unsupported answers and hallucinations.

---

### 7. Generation

Google Gemini generates the final answer using the retrieved context.

The application also retrieves the source chunks used for the response, allowing users to inspect the supporting document content.

---

## 🛠️ Technology Stack

| Technology                     | Purpose                   |
| ------------------------------ | ------------------------- |
| Python                         | Core programming language |
| Streamlit                      | Web application interface |
| LangChain                      | RAG orchestration         |
| PyPDF                          | PDF processing            |
| HuggingFace                    | Text embeddings           |
| FAISS                          | Vector similarity search  |
| Google Gemini                  | Large Language Model      |
| RecursiveCharacterTextSplitter | Document chunking         |

---

## 📂 Project Structure

```text
PDF-Q-A-LangChain/
│
├── app.py
├── README.md
├── Requirements.txt
├── .gitignore
│
└── .venv/                 # Local only — not committed
```

### `app.py`

Main Streamlit application containing:

* PDF processing
* Chunking
* Embedding generation
* FAISS indexing
* Retrieval
* Gemini integration
* Q&A interface
* Conversation history
* Source display
* Transcript export

### `Requirements.txt`

Contains the Python dependencies required to run the application.

### `.gitignore`

Prevents local environments, secrets, temporary files, and unnecessary files from being uploaded to GitHub.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/student-Sudeshnapaul/PDF-Q-A-LangChain.git
```

```bash
cd PDF-Q-A-LangChain
```

---

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r Requirements.txt
```

---

## 🔑 Gemini API Key

The application requires a Google Gemini API key for generating answers.

You can obtain an API key through Google AI Studio.

The key should **never be committed to GitHub**.

You can enter the key through the application's sidebar when running the application.

---

## ▶️ Run the Application

Start Streamlit with:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 💻 Usage

### Step 1 — Upload PDF

Upload a PDF using the file uploader.

### Step 2 — Configure Retrieval

Adjust:

```text
Chunk Size
Chunk Overlap
Top-K
Temperature
```

### Step 3 — Ask Questions

Example:

```text
What are the main objectives of this document?
```

```text
Explain section 3 in simple terms.
```

```text
What methodology was used?
```

```text
What are the key conclusions?
```

### Step 4 — Inspect Sources

The application displays the source chunks used to generate the answer.

### Step 5 — Export Conversation

The complete Q&A session can be exported as a `.txt` transcript.

---

## 🛡️ Robust PDF Processing

The application includes additional handling for problematic PDF files.

If standard PDF parsing fails, the application attempts to:

```text
Standard PDF Loading
        ↓
Repair malformed PDF structure
        ↓
Retry extraction
        ↓
Fallback to lenient PDF extraction
        ↓
Skip unreadable pages
        ↓
Continue processing readable content
```

This makes the application more resilient to malformed or partially corrupted PDFs.

---

## 🎛️ Configurable Parameters

### Chunk Size

Controls the number of characters contained in each document chunk.

Larger chunks:

* Provide more context
* May reduce retrieval precision

Smaller chunks:

* Improve retrieval granularity
* May lose surrounding context

---

### Chunk Overlap

Controls how much text is shared between consecutive chunks.

Overlap helps preserve contextual continuity between chunks.

---

### Top-K

Controls the number of document chunks retrieved for each question.

```text
Higher K → More context
Lower K  → More focused context
```

---

### Temperature

Controls the randomness of Gemini's generated responses.

```text
Lower temperature → More deterministic
Higher temperature → More creative
```

---

## 📊 Application Flow

```text
Upload PDF
    ↓
Extract Text
    ↓
Split into Chunks
    ↓
Generate Embeddings
    ↓
Create FAISS Index
    ↓
User Question
    ↓
Semantic Similarity Search
    ↓
Retrieve Relevant Chunks
    ↓
Construct Prompt
    ↓
Gemini
    ↓
Generate Answer
    ↓
Display Answer + Sources
```

---

## 🔬 Technical Concepts Demonstrated

This project demonstrates practical implementation of:

* Retrieval-Augmented Generation
* Large Language Models
* Vector embeddings
* Semantic search
* Vector databases
* Prompt engineering
* Document chunking
* Information retrieval
* LLM inference
* LangChain pipelines
* Streamlit application development
* Error handling and fallback strategies
* Conversational state management

---

## 🚀 Future Improvements

Potential improvements include:

* [ ] Multi-PDF question answering
* [ ] Persistent vector databases
* [ ] Conversation-aware retrieval
* [ ] Hybrid keyword + semantic search
* [ ] Re-ranking retrieved chunks
* [ ] OCR support for scanned PDFs
* [ ] Table extraction
* [ ] Citation-aware answers
* [ ] Authentication
* [ ] Cloud deployment
* [ ] Streaming LLM responses
* [ ] Document management dashboard
* [ ] Evaluation using RAG metrics
* [ ] Retrieval precision/recall analysis
* [ ] Support for additional document formats

---

## 🎯 Learning Outcomes

Through this project, I explored how modern AI applications combine:

```text
Traditional Information Retrieval
                +
Vector Embeddings
                +
Vector Search
                +
Large Language Models
                =
Retrieval-Augmented Generation
```

The project helped demonstrate how an LLM can be grounded in external documents rather than relying solely on its pretrained knowledge.

---

## 📌 Project Highlights

### Core AI

* RAG architecture
* Semantic embeddings
* FAISS similarity search
* Gemini LLM integration

### Backend / Processing

* PDF document extraction
* Recursive text chunking
* Vector indexing
* Retrieval pipeline
* Error recovery

### Frontend

* Streamlit
* Interactive PDF upload
* Configurable retrieval parameters
* Conversation interface
* Source visualization
* Transcript export

---

## 👩‍💻 Author

**Sudeshna Paul**

B.Tech — Artificial Intelligence & Machine Learning

---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.

---

## 📄 License

This project is intended for educational and portfolio purposes.
