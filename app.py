import os
import streamlit as st

from dotenv import load_dotenv

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate


# =========================================================
# 1. PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="RAG AI Assistant",
    page_icon="🤖",
    layout="centered"
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
# 3. HEADER
# =========================================================

st.title("🤖 RAG AI Assistant")
st.caption("Ask questions from your PDF")

st.divider()


# =========================================================
# 4. CONSTANTS
# =========================================================

CHROMA_PATH = "./chroma_db"

EMBEDDING_MODEL = "models/gemini-embedding-001"

LLM_MODEL = "gemini-3.6-flash"


# =========================================================
# 5. CHECK CHROMA DATABASE
# =========================================================

if not os.path.exists(CHROMA_PATH):

    st.error(
        "❌ ChromaDB not found.\n\n"
        "Make sure the `chroma_db` folder exists "
        "in the project directory."
    )

    st.stop()


# =========================================================
# 6. CREATE EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embeddings():

    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key
    )


embeddings = load_embeddings()


# =========================================================
# 7. LOAD CHROMA DATABASE
# =========================================================

@st.cache_resource
def load_vector_store():

    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )


vector_store = load_vector_store()


# =========================================================
# 8. CREATE RETRIEVER
# =========================================================

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 4
    }
)


# =========================================================
# 9. LOAD GEMINI
# =========================================================

@st.cache_resource
def load_llm():

    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0.2,
        google_api_key=api_key
    )


llm = load_llm()


# =========================================================
# 10. RAG PROMPT
# =========================================================

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful PDF question-answering assistant.

Your task is to answer the user's question using ONLY
the information present in the provided context.

IMPORTANT RULES:

1. Use only the provided context.
2. Do not use outside knowledge.
3. If the context contains the answer, answer clearly.
4. If the context does not contain the answer, say exactly:

"I could not find the answer in the provided PDF."

5. Do not make up information.
6. Keep the answer simple and easy to understand.

---------------- CONTEXT ----------------

{context}

---------------- QUESTION ----------------

{question}

---------------- ANSWER ----------------
"""
)


# =========================================================
# 11. EXTRACT RESPONSE TEXT
# =========================================================

def extract_response_text(response):

    content = response.content

    # Gemini may return list content
    if isinstance(content, list):

        text_parts = []

        for item in content:

            if isinstance(item, dict):

                if item.get("type") == "text":

                    text_parts.append(
                        item.get("text", "")
                    )

            elif isinstance(item, str):

                text_parts.append(item)

        return "\n".join(text_parts).strip()

    return str(content).strip()


# =========================================================
# 12. RAG FUNCTION
# =========================================================

def rag_chain(question):

    # -----------------------------------------------------
    # STEP 1: Retrieve documents
    # -----------------------------------------------------

    relevant_docs = retriever.invoke(question)

    # -----------------------------------------------------
    # STEP 2: Check retrieval
    # -----------------------------------------------------

    if not relevant_docs:

        return (
            "I could not find the answer in the provided PDF.",
            []
        )

    # -----------------------------------------------------
    # STEP 3: Combine document chunks
    # -----------------------------------------------------

    context = "\n\n".join(
        doc.page_content
        for doc in relevant_docs
    )

    # -----------------------------------------------------
    # STEP 4: Create prompt
    # -----------------------------------------------------

    formatted_prompt = prompt.format(
        context=context,
        question=question
    )

    # -----------------------------------------------------
    # STEP 5: Send to Gemini
    # -----------------------------------------------------

    response = llm.invoke(
        formatted_prompt
    )

    # -----------------------------------------------------
    # STEP 6: Extract answer
    # -----------------------------------------------------

    answer = extract_response_text(response)

    return answer, relevant_docs


# =========================================================
# 13. SESSION STATE
# =========================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# =========================================================
# 14. SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ RAG Settings")

    st.success("RAG System Ready ✅")

    st.write("### Embedding")

    st.code(
        EMBEDDING_MODEL
    )

    st.write("### LLM")

    st.code(
        LLM_MODEL
    )

    st.write("### Retriever")

    st.code(
        "Similarity Search"
    )

    st.write(
        "Top K = 4"
    )

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# =========================================================
# 15. DISPLAY CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# =========================================================
# 16. CHAT INPUT
# =========================================================

user_input = st.chat_input(
    "Ask something about your PDF..."
)


# =========================================================
# 17. PROCESS USER QUESTION
# =========================================================

if user_input:

    # -----------------------------------------------------
    # Show user message
    # -----------------------------------------------------

    with st.chat_message("user"):

        st.markdown(
            user_input
        )

    # -----------------------------------------------------
    # Save user message
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    # -----------------------------------------------------
    # Generate AI answer
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "🔍 Searching PDF..."
        ):

            try:

                answer, relevant_docs = rag_chain(
                    user_input
                )

                # -------------------------------------------------
                # Display answer
                # -------------------------------------------------

                st.markdown(answer)

                # -------------------------------------------------
                # Display retrieved sources
                # -------------------------------------------------

                if relevant_docs:

                    with st.expander(
                        "📚 Retrieved PDF Chunks"
                    ):

                        for i, doc in enumerate(
                            relevant_docs,
                            start=1
                        ):

                            st.markdown(
                                f"### Chunk {i}"
                            )

                            # Metadata
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


    # -----------------------------------------------------
    # Save assistant response
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )