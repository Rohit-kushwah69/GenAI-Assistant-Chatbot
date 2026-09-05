# 🤖 GenAI Assistant

An intelligent **PDF-based AI Assistant** built using **Retrieval-Augmented Generation (RAG)**, **LangChain**, **ChromaDB**, **Google Gemini**, and **Streamlit**.

The application allows users to ask questions about information contained in a PDF. The system retrieves the most relevant content from the document and uses Google Gemini to generate an answer based on the retrieved context.

## 🌐 Live Demo

🚀 **Try the GenAI Assistant:**  
[YOUR_STREAMLIT_APP_URL](https://genai-assistant018.streamlit.app/)

---

## 📌 Project Overview

The **GenAI Assistant** combines document retrieval with Generative AI.

Instead of asking the AI model to answer a question using only its general knowledge, the system first searches the provided PDF for relevant information and then sends that information to the Gemini LLM.

This allows the application to generate answers based on the information available in the user's document.

---

## 🚀 Features

- 📄 PDF document processing
- ✂️ Text splitting and chunking
- 🧠 Google Gemini embeddings
- 🗄️ ChromaDB vector database
- 🔍 Similarity-based document retrieval
- 📚 Relevant PDF context retrieval
- 🤖 Google Gemini LLM
- 🧩 Retrieval-Augmented Generation (RAG)
- 💬 Interactive AI Assistant
- 🗨️ Chat history
- 🌐 Streamlit Web UI
- 🔐 Environment variable based API key management

---

## 🏗️ RAG Architecture

```text
                         PDF
                          ↓
                   Load Documents
                          ↓
                       Chunking
                          ↓
                     Embeddings
                          ↓
                      ChromaDB
                          ↓
                      Retriever
                          ↓
                  Relevant Chunks
                          ↓
                       Context
                          ↓
                    Gemini LLM
                          ↓
                     RAG Chain
                          ↓
                   GenAI Assistant
                          ↓
                    Streamlit UI