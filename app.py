import os
import uuid
import time
import streamlit as st

from dotenv import load_dotenv

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =========================================================
# 1. PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="DocMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. CUSTOM CSS -> LIGHT UI
# =========================================================

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #f7f8fc 0%, #eef1fb 100%);
            color: #1f2233;
        }

        section[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #e3e6f5;
        }

        .hero {
            padding: 22px 28px;
            border-radius: 18px;
            background: linear-gradient(135deg, #6d5bf7 0%, #8f7bff 45%, #b8a6ff 100%);
            box-shadow: 0 8px 24px rgba(109, 91, 247, 0.18);
            margin-bottom: 18px;
        }
        .hero h1 { margin: 0; font-size: 28px; color: white; }
        .hero p  { margin: 4px 0 0 0; color: #f3f0ff; font-size: 14px; }

        .status-pill {
            display: inline-block;
            padding: 6px 14px;
            border-radius: 999px;
            font-size: 13px;
            font-weight: 600;
        }
        .pill-ready   { background: #e5f8ee; color: #1f9d55; border: 1px solid #a9e6c4; }
        .pill-waiting { background: #fff4e2; color: #b5741a; border: 1px solid #f2ce9a; }

        div[data-testid="stChatMessage"] {
            background: #ffffff;
            border: 1px solid #e3e6f5;
            border-radius: 14px;
            padding: 6px 10px;
            box-shadow: 0 2px 6px rgba(31, 34, 51, 0.04);
        }

        .stButton>button {
            border-radius: 10px;
            border: none;
            background: linear-gradient(135deg, #6d5bf7, #8f7bff);
            color: white;
            font-weight: 600;
        }
        .stButton>button:hover {
            background: linear-gradient(135deg, #7c6bff, #a292ff);
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 3. LOAD ENVIRONMENT
# =========================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("❌ GEMINI_API_KEY not found in .env file")
    st.stop()


# =========================================================
# 4. CONSTANTS
# =========================================================

EMBEDDING_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "gemini-3.6-flash"          # current stable Gemini Flash model
UPLOAD_FOLDER = "./uploaded_pdfs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# =========================================================
# 5. HERO HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
        <h1>🧠 DocMind AI</h1>
        <p>Upload your PDF and get accurate answers based only on that document.</p>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 6. LOAD EMBEDDINGS (cached once)
# =========================================================

@st.cache_resource
def load_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key
    )


embeddings = load_embeddings()


# =========================================================
# 7. LOAD LLM (cached once)
# =========================================================

@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0,
        google_api_key=api_key
    )


llm = load_llm()


# =========================================================
# 7b. RATE-LIMITED BATCH EMBEDDING
#
# Google's free-tier embedding API allows only ~100 requests/minute.
# Sending all PDF chunks at once can exceed this and cause a
# RESOURCE_EXHAUSTED (429) error. This helper sends chunks in small
# batches, pauses between batches to stay under the limit, and
# automatically retries with backoff if a 429 still occurs.
# =========================================================

def add_documents_rate_limited(vector_store, chunks, batch_size=25, pause_seconds=20, progress_bar=None):
    total = len(chunks)

    for start in range(0, total, batch_size):
        batch = chunks[start:start + batch_size]

        attempt = 0
        while True:
            try:
                vector_store.add_documents(documents=batch)
                break
            except Exception as e:
                error_text = str(e)
                if "RESOURCE_EXHAUSTED" in error_text or "429" in error_text:
                    attempt += 1
                    if attempt > 4:
                        raise
                    time.sleep(30)  # back off and retry this batch
                else:
                    raise

        if progress_bar is not None:
            done = min(start + batch_size, total)
            progress_bar.progress(done / total, text=f"Embedding chunks... {done}/{total}")

        # Pause between batches to stay under the free-tier rate limit
        if start + batch_size < total:
            time.sleep(pause_seconds)


# =========================================================
# 8. FRESH VECTOR STORE PER PDF
#
# A new, empty in-memory collection is created every time a PDF
# is processed, so answers always come from the current PDF only
# (no mixing with chunks from a previously uploaded PDF).
# =========================================================

if "collection_name" not in st.session_state:
    st.session_state.collection_name = f"rag_{uuid.uuid4().hex}"

if "vector_store" not in st.session_state:
    st.session_state.vector_store = Chroma(
        collection_name=st.session_state.collection_name,
        embedding_function=embeddings
        # no persist_directory -> fresh in-memory store, resets every new PDF
    )

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# 9. SIDEBAR - PDF UPLOAD
# =========================================================

with st.sidebar:

    st.header("📄 Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose any PDF (resume, notes, report...)",
        type=["pdf"],
        help="Answers will be based only on the PDF you upload."
    )

    if uploaded_file is not None:

        st.info(f"📄 {uploaded_file.name}")

        if st.button("🚀 Process PDF", use_container_width=True):

            with st.spinner("📚 Processing PDF..."):

                try:
                    # Save PDF
                    pdf_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
                    with open(pdf_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Load PDF
                    loader = PyPDFLoader(pdf_path)
                    documents = loader.load()

                    # Check if any real text was extracted at all
                    total_text = "".join(doc.page_content.strip() for doc in documents)

                    if not total_text:
                        st.error(
                            "❌ No readable text found in this PDF.\n\n"
                            "This usually means the PDF is scanned or image-based "
                            "(no selectable text layer). Try a text-based PDF, or run OCR "
                            "on it first and upload the OCR'd version."
                        )
                        st.stop()

                    # Split PDF into chunks
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1500,
                        chunk_overlap=150
                    )
                    chunks = text_splitter.split_documents(documents)

                    # Drop any chunks that ended up empty/whitespace-only
                    chunks = [c for c in chunks if c.page_content.strip()]

                    if not chunks:
                        st.error("❌ Could not extract any usable text chunks from this PDF.")
                        st.stop()

                    # Start a brand-new empty collection for this PDF
                    st.session_state.collection_name = f"rag_{uuid.uuid4().hex}"
                    st.session_state.vector_store = Chroma(
                        collection_name=st.session_state.collection_name,
                        embedding_function=embeddings
                    )

                    # Store ONLY current PDF's chunks (rate-limited to avoid 429 errors)
                    progress_bar = st.progress(0, text="Embedding chunks...")
                    add_documents_rate_limited(
                        st.session_state.vector_store,
                        chunks,
                        batch_size=25,
                        pause_seconds=20,
                        progress_bar=progress_bar
                    )
                    progress_bar.empty()

                    st.session_state.pdf_processed = True
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.messages = []  # clear old chat for the new PDF

                    st.success(f"✅ Ready!\n\nPages: {len(documents)}\nChunks: {len(chunks)}")

                except Exception as e:
                    st.error(f"❌ PDF processing failed:\n\n{str(e)}")

    st.divider()

    if st.session_state.pdf_processed:
        st.markdown(
            f'<span class="status-pill pill-ready">✅ Active: {st.session_state.pdf_name}</span>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<span class="status-pill pill-waiting">⏳ No PDF processed yet</span>',
            unsafe_allow_html=True
        )

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# =========================================================
# 10. PROMPT
# =========================================================

prompt = ChatPromptTemplate.from_template(
    """
Answer the question using ONLY the PDF context below.

If the answer is not present in the context, say:
"I could not find the answer in the provided PDF."

Be concise and accurate. Do not use outside knowledge.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
)


# =========================================================
# 11. RAG FUNCTION
# =========================================================

def rag_chain(question: str):

    retriever = st.session_state.vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    attempt = 0
    while True:
        try:
            docs = retriever.invoke(question)
            break
        except Exception as e:
            error_text = str(e)
            if ("RESOURCE_EXHAUSTED" in error_text or "429" in error_text) and attempt < 3:
                attempt += 1
                time.sleep(15)
            else:
                raise

    if not docs:
        return "I could not find the answer in the provided PDF."

    context = "\n\n".join(doc.page_content for doc in docs)

    formatted_prompt = prompt.format(context=context, question=question)

    response = llm.invoke(formatted_prompt)
    answer = response.content

    if isinstance(answer, list):
        answer = "\n".join(
            item.get("text", "") for item in answer if isinstance(item, dict)
        )

    return str(answer).strip()


# =========================================================
# 12. CHAT HISTORY
# =========================================================

if not st.session_state.pdf_processed:
    st.warning("👈 First upload a PDF from the sidebar and click 'Process PDF'.")

for message in st.session_state.messages:
    avatar = "🧑" if message["role"] == "user" else "🧠"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


# =========================================================
# 13. CHAT INPUT
# =========================================================

user_input = st.chat_input(
    "Ask anything about your PDF...",
    disabled=not st.session_state.pdf_processed
)


# =========================================================
# 14. PROCESS QUESTION  (plain answer only, no chunk/source display)
# =========================================================

if user_input:

    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant", avatar="🧠"):

        with st.spinner("🔍 Finding the answer..."):

            try:
                answer = rag_chain(user_input)
                st.markdown(answer)

            except Exception as e:
                answer = f"❌ RAG Error\n\n`{str(e)}`"
                st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})