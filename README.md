# 🤖 RAG AI Assistant

An intelligent **PDF-based AI Chatbot** built using **Retrieval-Augmented Generation (RAG)**, **LangChain**, **ChromaDB**, **Google Gemini**, and **Streamlit**.

The application allows users to ask questions about information contained in a PDF. The system retrieves the most relevant content from the document and uses Google Gemini to generate an answer based on that retrieved context.

---

## 📌 Project Overview

The **RAG AI Assistant** combines document retrieval with Generative AI.

Instead of asking the AI model to answer a question using only its general knowledge, the system first searches the provided PDF for relevant information and then sends that information to the Gemini LLM.

This makes the chatbot capable of answering questions based on the user's document.

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
- 💬 AI Chatbot
- 🗨️ Interactive chat interface
- 🧠 Chat history
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
                AI Chatbot
                     ↓
               Streamlit UI
```

---

## 🔄 How the RAG System Works

When the user asks a question, the following process takes place:

### 1. PDF Loading

The PDF document is loaded into the application.

### 2. Chunking

The document is divided into smaller chunks so that relevant information can be efficiently retrieved.

### 3. Embeddings

Each text chunk is converted into a numerical vector using the Google Gemini embedding model.

### 4. Vector Database

The generated embeddings are stored in **ChromaDB**.

### 5. Retriever

When the user asks a question, the retriever searches ChromaDB for the most relevant chunks.

### 6. Context Creation

The retrieved chunks are combined and provided as context to the Gemini model.

### 7. Gemini LLM

Google Gemini receives the context and the user's question.

### 8. Final Answer

Gemini generates the final answer based on the retrieved PDF content.

---

## 💬 AI Chatbot

The project includes an interactive AI chatbot built with Streamlit.

Users can enter questions such as:

```text
What is Python?

What is programming?

What are the features of Python?

What is machine learning?

Explain Python in simple language.
```

The chatbot retrieves relevant information from the PDF and generates an answer using Gemini.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| LangChain | RAG Application Framework |
| Google Gemini | Large Language Model |
| Gemini Embeddings | Text Embeddings |
| ChromaDB | Vector Database |
| Streamlit | Web UI / Chatbot Interface |
| PyPDF | PDF Processing |
| python-dotenv | Environment Variable Management |

---

## 📂 Project Structure

```text
GenAI-Assistant/
│
├── app.py
├── RAG_AI.ipynb
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── Python_Notes.pdf
│
└── chroma_db/
```

> `chroma_db/` and `.env` should not be uploaded to GitHub if they contain local/generated data or secrets.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the Project Folder

```bash
cd GenAI-Assistant
```

### 3. Create Virtual Environment

```bash
python -m venv venv
```

### 4. Activate Virtual Environment

For Windows:

```bash
venv\Scripts\activate
```

---

## 📦 Install Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

---

## 🔑 API Key Configuration

Create a `.env` file in the project root directory:

```text
GEMINI_API_KEY=your_gemini_api_key
```

The application loads the API key using `python-dotenv`.

### ⚠️ Important

Never upload your `.env` file or Gemini API key to GitHub.

Add this to `.gitignore`:

```text
.env
```

---

## ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

After running the command, Streamlit will provide a local URL:

```text
http://localhost:8501
```

Open the URL in your browser to use the AI chatbot.

---

## 🧪 Jupyter Notebook

The project also contains a Jupyter Notebook:

```text
RAG_AI.ipynb
```

The notebook demonstrates the RAG pipeline step by step:

```text
PDF
 ↓
Load Documents
 ↓
Chunking
 ↓
Embeddings
 ↓
Vector Database
 ↓
Retriever
 ↓
Relevant Chunks
 ↓
Gemini
 ↓
RAG Chain
```

The notebook was used to develop and test the RAG pipeline before integrating it with the Streamlit chatbot interface.

---

## 🔍 Retriever Configuration

The application uses similarity search to retrieve relevant chunks.

Example:

```python
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
```

Here:

```text
k = 3
```

means the system retrieves the top 3 most relevant chunks from the vector database.

---

## 🧠 RAG Prompt

The AI is instructed to answer using the retrieved PDF context.

Conceptually:

```text
Context:
Relevant PDF chunks

Question:
User's question

        ↓

Gemini

        ↓

Final Answer
```

If the requested information is not available in the provided context, the chatbot can respond:

```text
I could not find the answer in the provided PDF.
```

---

## 💡 Why RAG?

Traditional LLM:

```text
User Question
      ↓
     LLM
      ↓
   Answer
```

RAG:

```text
User Question
      ↓
   Retriever
      ↓
Relevant Documents
      ↓
    Context
      ↓
      LLM
      ↓
    Answer
```

RAG allows the application to provide document-specific information to the LLM.

---

## 📊 Example Workflow

Suppose the PDF contains information about Python.

The user asks:

```text
What is Python?
```

The system performs:

```text
Question
   ↓
Embedding
   ↓
ChromaDB Search
   ↓
Top 3 Relevant Chunks
   ↓
Context
   ↓
Gemini
   ↓
AI Answer
```

The user receives the final answer through the Streamlit chat interface.

---

## 🔮 Future Improvements

The project can be extended with:

- [ ] Upload PDF directly from Streamlit
- [ ] Support multiple PDF files
- [ ] Support DOCX and TXT documents
- [ ] Source document/page citations
- [ ] Display retrieved chunks
- [ ] Streaming AI responses
- [ ] Improved chat memory
- [ ] Better chatbot UI
- [ ] Authentication
- [ ] Cloud deployment
- [ ] Conversation export
- [ ] Multiple users
- [ ] Document management system

---

## 🎯 Learning Objectives

This project demonstrates practical implementation of:

- Generative AI
- Large Language Models
- Prompt Engineering
- Embeddings
- Vector Databases
- Semantic Search
- Information Retrieval
- Retrieval-Augmented Generation
- LangChain
- Gemini API
- Streamlit
- AI Chatbot Development

---

## 📸 Application

The application provides a simple chat interface where users can ask questions related to their PDF.

```text
┌─────────────────────────────────────┐
│       🤖 RAG AI Assistant           │
│                                     │
│  Ask questions from your PDF        │
│                                     │
│  User: What is Python?              │
│                                     │
│  AI: Python is a high-level...      │
│                                     │
│                                     │
│  Ask something about your PDF...    │
└─────────────────────────────────────┘
```

---

## 🔐 Security

API keys and secrets should always be stored in environment variables.

Do not commit:

```text
.env
API keys
Passwords
Secret tokens
```

Use `.gitignore`:

```text
.env
__pycache__/
venv/
.venv/
chroma_db/
```

---

## 📜 License

This project is created for learning and educational purposes.

---

## 👨‍💻 Author

**Rohit Singh**

---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.

---

## 🚀 Project Summary

**RAG AI Assistant** is an end-to-end Generative AI project that combines:

```text
PDF Documents
      +
Embeddings
      +
Vector Database
      +
Retriever
      +
Google Gemini
      +
RAG
      +
Streamlit
      ↓
AI Chatbot
```

It demonstrates how a document-based AI assistant can be built using modern Generative AI and Retrieval-Augmented Generation techniques.