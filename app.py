import os
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
    page_title="RAG AI Assistant",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# 2. LOAD ENVIRONMENT
# =========================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("❌ GEMINI_API_KEY not found in .env file")
    st.stop()


# =========================================================
# 3. CONSTANTS
# =========================================================

CHROMA_PATH = "./chroma_db"
EMBEDDING_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "gemini-3.6-flash"
UPLOAD_FOLDER = "./uploaded_pdfs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# =========================================================
# 4. HEADER
# =========================================================

st.title("🤖 RAG AI Assistant")
st.caption("Upload a PDF and ask questions from it")

st.divider()


# =========================================================
# 5. LOAD EMBEDDINGS
# =========================================================

@st.cache_resource
def load_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key
    )


embeddings = load_embeddings()


# =========================================================
# 6. LOAD CHROMA
# =========================================================

@st.cache_resource
def load_vector_store():
    return Chroma(
        collection_name="rag_documents",
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )


vector_store = load_vector_store()


# =========================================================
# 7. SIDEBAR - PDF UPLOAD
# =========================================================

with st.sidebar:

    st.header("📄 PDF Upload")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload the PDF you want to ask questions about."
    )

    if uploaded_file is not None:

        st.info(f"📄 {uploaded_file.name}")

        if st.button(
            "🚀 Process PDF",
            use_container_width=True
        ):

            with st.spinner("📚 Processing PDF..."):

                try:
                    # Save PDF
                    pdf_path = os.path.join(
                        UPLOAD_FOLDER,
                        uploaded_file.name
                    )

                    with open(pdf_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Load PDF
                    loader = PyPDFLoader(pdf_path)
                    documents = loader.load()

                    # Split PDF into chunks
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=150
                    )

                    chunks = text_splitter.split_documents(documents)

                    # Store chunks in ChromaDB
                    vector_store.add_documents(
                        documents=chunks
                    )

                    # Remember processed PDF
                    st.session_state["pdf_processed"] = True
                    st.session_state["pdf_name"] = uploaded_file.name

                    st.success(
                        f"✅ PDF ready!\n\n"
                        f"Pages: {len(documents)}\n"
                        f"Chunks: {len(chunks)}"
                    )

                except Exception as e:
                    st.error(
                        f"❌ PDF processing failed:\n\n{str(e)}"
                    )

    st.divider()

    if st.session_state.get("pdf_processed"):
        st.success(
            f"✅ Ready: {st.session_state.get('pdf_name', 'PDF')}"
        )
    else:
        st.warning("⚠️ Upload and process a PDF first.")

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.rerun()


# =========================================================
# 8. RETRIEVER - FAST
# =========================================================

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 2
    }
)


# =========================================================
# 9. LOAD GEMINI LLM
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
# 10. PROMPT - SHORT AND FAST
# =========================================================

prompt = ChatPromptTemplate.from_template(
    """
Answer the question using ONLY the PDF context below.

If the answer is not present in the context, say:
"I could not find the answer in the provided PDF."

Be concise. Do not use outside knowledge.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
)


# =========================================================
# 11. FAST RAG FUNCTION
# =========================================================

def rag_chain(question):

    # Retrieve only the 2 most relevant chunks
    docs = retriever.invoke(question)

    if not docs:
        return (
            "I could not find the answer in the provided PDF.",
            []
        )

    # Keep the context small for faster Gemini response
    context = "\n\n".join(
        doc.page_content[:1500]
        for doc in docs
    )

    # Build prompt
    formatted_prompt = prompt.format(
        context=context,
        question=question
    )

    # Ask Gemini
    response = llm.invoke(formatted_prompt)

    answer = response.content

    # Gemini can sometimes return a list
    if isinstance(answer, list):
        answer = "\n".join(
            item.get("text", "")
            for item in answer
            if isinstance(item, dict)
        )

    return str(answer).strip(), docs


# =========================================================
# 12. SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# 13. CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# =========================================================
# 14. CHAT INPUT
# =========================================================

user_input = st.chat_input(
    "Ask something about your PDF..."
)


# =========================================================
# 15. PROCESS QUESTION
# =========================================================

if user_input:

    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    # Generate answer
    with st.chat_message("assistant"):

        with st.spinner("🔍 Finding answer..."):

            try:

                answer, docs = rag_chain(user_input)

                # Show answer
                st.markdown(answer)

                # Retrieved chunks
                if docs:

                    with st.expander(
                        "📚 Retrieved PDF Chunks"
                    ):

                        for i, doc in enumerate(
                            docs,
                            start=1
                        ):

                            st.markdown(
                                f"### Chunk {i}"
                            )

                            metadata = doc.metadata

                            page = metadata.get(
                                "page_label",
                                metadata.get(
                                    "page",
                                    "Unknown"
                                )
                            )

                            source = metadata.get(
                                "source",
                                "Unknown"
                            )

                            st.write(
                                f"📄 **Page:** {page}"
                            )

                            st.write(
                                f"📁 **Source:** {source}"
                            )

                            st.write(
                                "#### Retrieved Text"
                            )

                            st.write(
                                doc.page_content
                            )

                            st.divider()

            except Exception as e:

                answer = (
                    "❌ RAG Error\n\n"
                    f"`{str(e)}`"
                )

                st.error(answer)

    # Save assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
