"""
Research Paper Answer Bot — RAG over Research Papers
=====================================================
A Streamlit application that implements a Retrieval-Augmented Generation (RAG)
system for answering questions about uploaded research papers.

Stretch Goal — Capstone Project
"""

import os
import tempfile
from typing import Generator

import streamlit as st
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except (ImportError, ModuleNotFoundError):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
try:
    from langchain.memory import ConversationBufferWindowMemory
except (ImportError, ModuleNotFoundError):
    from langchain_core.messages import HumanMessage as _HM, AIMessage as _AM
    class ConversationBufferWindowMemory:
        def __init__(self, k=5, memory_key='chat_history', return_messages=True, **kw):
            self.k=k; self.memory_key=memory_key; self.return_messages=return_messages; self.chat_history=[]
        def load_memory_variables(self, inputs=None):
            return {self.memory_key: self.chat_history[-(self.k*2):]}
        def save_context(self, inputs, outputs):
            iv=inputs.get('input', inputs.get('question',''))
            ov=outputs.get('output', outputs.get('answer',''))
            self.chat_history += [_HM(content=iv), _AM(content=ov)]
        def clear(self):
            self.chat_history=[]
from langchain_core.messages import HumanMessage, AIMessage
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_community.vectorstores import Chroma
from rank_bm25 import BM25Okapi

# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Research Paper Answer Bot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    .source-card {
        background-color: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .source-card h4 {
        margin: 0 0 4px 0;
        color: #1a73e8;
        font-size: 0.9rem;
    }
    .source-card .meta {
        font-size: 0.75rem;
        color: #5f6368;
        margin-bottom: 6px;
    }
    .source-card .passage {
        font-size: 0.82rem;
        color: #3c4043;
        line-height: 1.4;
    }
    .status-indicator {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        border-radius: 6px;
        background: #e8f0fe;
        margin-bottom: 8px;
        font-size: 0.85rem;
        color: #1967d2;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────────────────────────────────────


def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "messages": [],
        "chat_memory": ConversationBufferWindowMemory(
            k=5, return_messages=True, memory_key="chat_history"
        ),
        "vectorstore": None,
        "documents": [],
        "paper_metadata": [],
        "bm25_index": None,
        "bm25_corpus": [],
        "bm25_docs": [],
        "processing": False,
        "papers_loaded": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Configuration")

    # --- API Key ---
    st.subheader("🔑 API Key")
    api_key = st.text_input(
        "Google Gemini API Key",
        type="password",
        placeholder="AIza...",
        help="Required for embeddings and chat completion.",
    )
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key

    st.divider()

    # --- Paper Upload ---
    st.subheader("📄 Upload Papers")
    uploaded_files = st.file_uploader(
        "Upload PDF research papers",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more research paper PDFs to build the knowledge base.",
    )

    st.divider()

    # --- Settings ---
    st.subheader("🛠️ Settings")

    chunk_size = st.slider(
        "Chunk Size (tokens)",
        min_value=200,
        max_value=2000,
        value=800,
        step=100,
        help="Size of text chunks for indexing.",
    )

    chunk_overlap = st.slider(
        "Chunk Overlap",
        min_value=0,
        max_value=500,
        value=150,
        step=50,
        help="Overlap between consecutive chunks.",
    )

    retrieval_strategy = st.selectbox(
        "Retrieval Strategy",
        options=["Dense (Embedding)", "Sparse (BM25)", "Hybrid (Dense + BM25)"],
        index=2,
        help="Choose how documents are retrieved for answering.",
    )

    num_results = st.slider(
        "Number of Results (k)",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        help="Number of chunks to retrieve per query.",
    )

    st.divider()

    # --- Process Button ---
    process_btn = st.button(
        "🚀 Process Papers", use_container_width=True, type="primary"
    )

    # --- Status ---
    if st.session_state.papers_loaded:
        st.success(
            f"✅ {len(st.session_state.paper_metadata)} paper(s) indexed "
            f"({len(st.session_state.documents)} chunks)"
        )

# ─────────────────────────────────────────────────────────────────────────────
# Document Processing
# ─────────────────────────────────────────────────────────────────────────────


def process_papers(files, chunk_size: int, chunk_overlap: int):
    """Load PDFs, split into chunks, and index in ChromaDB + BM25."""
    all_docs: list[Document] = []
    metadata_list: list[dict] = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    for file in files:
        # Write to temp file for PyPDFLoader
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

        # Add paper-level metadata
        paper_title = file.name.replace(".pdf", "").replace("_", " ").title()
        metadata_list.append(
            {"title": paper_title, "filename": file.name, "pages": len(pages)}
        )

        for page in pages:
            page.metadata["paper_title"] = paper_title
            page.metadata["source_file"] = file.name

        chunks = text_splitter.split_documents(pages)
        all_docs.extend(chunks)

        # Cleanup temp file
        os.unlink(tmp_path)

    # --- ChromaDB Vector Store ---
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)

    # Use ephemeral client (in-memory)
    chroma_client = chromadb.EphemeralClient(
        settings=ChromaSettings(anonymized_telemetry=False)
    )

    # Create empty collection, then add documents in throttled batches.
    # Free-tier Gemini embeddings allow ~100 requests/minute, so we embed in
    # small batches with a pause + exponential-backoff retry on 429/503 errors.
    import time
    vectorstore = Chroma(
        embedding_function=embeddings,
        client=chroma_client,
        collection_name="research_papers",
    )

    BATCH_SIZE = 50          # stay well under the 100/min free-tier cap
    PAUSE_SECONDS = 30       # wait between batches so the per-minute quota resets

    total = len(all_docs)
    progress = st.progress(0.0, text=f"Embedding 0/{total} chunks...")
    for start in range(0, total, BATCH_SIZE):
        batch = all_docs[start:start + BATCH_SIZE]
        for attempt in range(6):  # retry with exponential backoff
            try:
                vectorstore.add_documents(batch)
                break
            except Exception as e:
                msg = str(e)
                if ("429" in msg or "RESOURCE_EXHAUSTED" in msg
                        or "503" in msg or "UNAVAILABLE" in msg):
                    wait = min(60, 5 * (2 ** attempt))
                    progress.progress(
                        start / total,
                        text=f"Rate limit hit - waiting {wait}s then retrying "
                             f"(batch {start//BATCH_SIZE + 1})...",
                    )
                    time.sleep(wait)
                    continue
                raise  # a real error - surface it
        done = min(start + BATCH_SIZE, total)
        progress.progress(done / total, text=f"Embedded {done}/{total} chunks...")
        if done < total:
            time.sleep(PAUSE_SECONDS)  # respect the per-minute quota
    progress.empty()

    # --- BM25 Index ---
    tokenized_corpus = [doc.page_content.lower().split() for doc in all_docs]
    bm25_index = BM25Okapi(tokenized_corpus)

    return vectorstore, all_docs, metadata_list, bm25_index, tokenized_corpus


# Process papers when button is clicked
if process_btn and uploaded_files:
    if not api_key:
        st.sidebar.error("⚠️ Please enter your Google Gemini API key first.")
    else:
        with st.sidebar:
            with st.spinner("Processing papers..."):
                try:
                    vectorstore, docs, meta, bm25_idx, corpus = process_papers(
                        uploaded_files, chunk_size, chunk_overlap
                    )
                    st.session_state.vectorstore = vectorstore
                    st.session_state.documents = docs
                    st.session_state.paper_metadata = meta
                    st.session_state.bm25_index = bm25_idx
                    st.session_state.bm25_corpus = corpus
                    st.session_state.bm25_docs = docs
                    st.session_state.papers_loaded = True
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error processing papers: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Retrieval Functions
# ─────────────────────────────────────────────────────────────────────────────


def retrieve_dense(query: str, k: int) -> list[Document]:
    """Retrieve using dense embeddings via ChromaDB."""
    if st.session_state.vectorstore is None:
        return []
    results = st.session_state.vectorstore.similarity_search(query, k=k)
    return results


def retrieve_sparse(query: str, k: int) -> list[Document]:
    """Retrieve using BM25 sparse retrieval."""
    if st.session_state.bm25_index is None:
        return []
    tokenized_query = query.lower().split()
    scores = st.session_state.bm25_index.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [st.session_state.bm25_docs[i] for i in top_indices]


def retrieve_hybrid(query: str, k: int) -> list[Document]:
    """Hybrid retrieval combining dense and sparse results with RRF."""
    dense_results = retrieve_dense(query, k=k)
    sparse_results = retrieve_sparse(query, k=k)

    # Reciprocal Rank Fusion (RRF)
    doc_scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}
    rrf_k = 60  # Standard RRF constant

    for rank, doc in enumerate(dense_results):
        key = doc.page_content[:100]
        doc_scores[key] = doc_scores.get(key, 0) + 1.0 / (rrf_k + rank + 1)
        doc_map[key] = doc

    for rank, doc in enumerate(sparse_results):
        key = doc.page_content[:100]
        doc_scores[key] = doc_scores.get(key, 0) + 1.0 / (rrf_k + rank + 1)
        doc_map[key] = doc

    sorted_keys = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
    return [doc_map[key] for key in sorted_keys[:k]]


def retrieve(query: str, k: int, strategy: str) -> list[Document]:
    """Route to the selected retrieval strategy."""
    if "Dense" in strategy:
        return retrieve_dense(query, k)
    elif "Sparse" in strategy:
        return retrieve_sparse(query, k)
    else:
        return retrieve_hybrid(query, k)


# ─────────────────────────────────────────────────────────────────────────────
# RAG Chain (LangChain LCEL)
# ─────────────────────────────────────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """You are a research assistant specializing in answering questions about academic papers. \
Use the provided context from research papers to answer the user's question accurately and thoroughly.

Guidelines:
- Ground your answers in the provided context. If the context doesn't contain enough information, say so.
- Cite specific papers when referencing findings.
- Be precise with technical terminology.
- If multiple papers discuss a topic, synthesize the information.
- Maintain academic rigor in your responses.

Context from research papers:
{context}
"""

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", RAG_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)


def format_docs(docs: list[Document]) -> str:
    """Format retrieved documents into context string."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        paper = doc.metadata.get("paper_title", "Unknown")
        page = doc.metadata.get("page", "?")
        formatted.append(
            f"[Source {i}] Paper: {paper} | Page: {page}\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted)


def get_rag_chain(api_key: str):
    """Build the LCEL RAG chain."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0.1,
        streaming=True,
    )

    chain = RAG_PROMPT | llm | StrOutputParser()
    return chain


def stream_response(query: str, api_key: str, k: int, strategy: str) -> Generator:
    """Stream the RAG response and capture sources."""
    # Retrieve relevant documents
    sources = retrieve(query, k, strategy)
    st.session_state["_current_sources"] = sources

    # Get chat history from memory
    memory_vars = st.session_state.chat_memory.load_memory_variables({})
    chat_history = memory_vars.get("chat_history", [])

    # Build context
    context = format_docs(sources)

    # Get chain and stream
    chain = get_rag_chain(api_key)

    for chunk in chain.stream(
        {"context": context, "question": query, "chat_history": chat_history}
    ):
        yield chunk


# ─────────────────────────────────────────────────────────────────────────────
# Main Chat Interface
# ─────────────────────────────────────────────────────────────────────────────

st.title("📚 Research Paper Answer Bot")
st.caption("Ask questions about your uploaded research papers — powered by RAG")

# Two-column layout: chat + sources
chat_col, source_col = st.columns([3, 1])

with chat_col:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input(
        "Ask a question about your research papers...",
        disabled=not st.session_state.papers_loaded,
    ):
        if not api_key:
            st.error("⚠️ Please enter your Google Gemini API key in the sidebar.")
            st.stop()

        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            # Status indicators
            status_placeholder = st.empty()
            status_placeholder.markdown(
                '<div class="status-indicator">🔍 Searching papers...</div>',
                unsafe_allow_html=True,
            )

            # Stream the response
            try:
                status_placeholder.markdown(
                    '<div class="status-indicator">🧠 Thinking...</div>',
                    unsafe_allow_html=True,
                )
                response = st.write_stream(
                    stream_response(prompt, api_key, num_results, retrieval_strategy)
                )
                status_placeholder.empty()
            except Exception as e:
                status_placeholder.empty()
                response = f"❌ Error generating response: {e}"
                st.error(response)

        # Save assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Update conversational memory
        st.session_state.chat_memory.save_context(
            {"input": prompt}, {"output": response}
        )

        # Trigger rerun to update sources panel
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Source Citations Panel
# ─────────────────────────────────────────────────────────────────────────────

with source_col:
    st.subheader("📎 Sources")

    if "_current_sources" in st.session_state and st.session_state["_current_sources"]:
        sources = st.session_state["_current_sources"][:3]  # Top-3 sources
        for i, doc in enumerate(sources, 1):
            paper_title = doc.metadata.get("paper_title", "Unknown Paper")
            page_num = doc.metadata.get("page", "N/A")
            passage = doc.page_content[:250] + (
                "..." if len(doc.page_content) > 250 else ""
            )

            st.markdown(
                f"""
                <div class="source-card">
                    <h4>📄 {paper_title}</h4>
                    <div class="meta">Page {page_num} · Source {i}</div>
                    <div class="passage">{passage}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("Sources will appear here after you ask a question.")

    st.divider()

    # Paper index info
    if st.session_state.paper_metadata:
        st.subheader("📚 Indexed Papers")
        for paper in st.session_state.paper_metadata:
            st.markdown(f"**{paper['title']}**  \n{paper['pages']} pages")

# ─────────────────────────────────────────────────────────────────────────────
# Empty State
# ─────────────────────────────────────────────────────────────────────────────

if not st.session_state.papers_loaded:
    st.info(
        "👈 Upload research papers in the sidebar and click **Process Papers** to get started."
    )
