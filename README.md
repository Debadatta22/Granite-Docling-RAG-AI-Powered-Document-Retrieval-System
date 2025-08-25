# 🚀 Granite-Docling-RAG-AI-Powered-Document-Retrieval-System  

<p align="center">
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/Python-3.10+-blue.svg?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  </a>
  <a href="https://github.com/Debadatta22/Granite-Docling-RAG-AI-Powered-Document-Retrieval-System/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/Debadatta22/Granite-Docling-RAG-AI-Powered-Document-Retrieval-System?style=for-the-badge" alt="License"/>
  </a>
  <a href="https://github.com/Debadatta22/Granite-Docling-RAG-AI-Powered-Document-Retrieval-System/stargazers">
    <img src="https://img.shields.io/github/stars/Debadatta22/Granite-Docling-RAG-AI-Powered-Document-Retrieval-System?style=for-the-badge&logo=github" alt="Stars"/>
  </a>
  <a href="https://github.com/Debadatta22/Granite-Docling-RAG-AI-Powered-Document-Retrieval-System/network/members">
    <img src="https://img.shields.io/github/forks/Debadatta22/Granite-Docling-RAG-AI-Powered-Document-Retrieval-System?style=for-the-badge&logo=github" alt="Forks"/>
  </a>
  <a href="https://github.com/Debadatta22">
    <img src="https://img.shields.io/badge/Author-Debadatta22-2D9BF0?style=for-the-badge&logo=github&logoColor=white" alt="Author"/>
  </a>
</p>

---

## 📜 License  

<p align="center">
  <a href="https://github.com/Debadatta22/Granite-Docling-RAG-AI-Powered-Document-Retrieval-System/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge" alt="MIT License"/>
  </a>
</p>

This project is licensed under the **MIT License** – see the [LICENSE](https://github.com/Debadatta22/Granite-Docling-RAG-AI-Powered-Document-Retrieval-System/blob/main/LICENSE) file for details.
 

---

## 🔗 Quick Access  

<p align="center">
  <a href="https://colab.research.google.com/drive/1Ul0W_HS8lO5jvTqrpy3maTMpm8Xd3myv?usp=sharing" target="_blank">
    <img src="https://img.shields.io/badge/Open%20in%20Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Open In Colab"/>
  </a>
</p>

---

## 📑 Table of Contents  

1. [Problem Statement](#1-problem-statement)  
2. [Solution Overview](#2-solution-overview)  
3. [Architecture & Workflow](#3-architecture--workflow)  
4. [Tech Stack](#4-tech-stack)  
5. [Features](#5-features)  
6. [Getting Started](#6-getting-started)  
7. [Configuration](#7-configuration)  
8. [How to Run](#8-how-to-run)  
9. [Algorithms & Concepts](#9-algorithms--concepts)  
10. [Example: Swapping Sources & Questions](#10-example-swapping-sources--questions)  
11. [Extensibility](#11-extensibility)  
12. [Evaluation & Quality](#12-evaluation--quality)  
13. [Real-World Considerations](#13-real-world-considerations)  
14. [Project Structure](#14-project-structure-suggested)  
15. [License](#15-license)  
16. [Certification](#16-certification)  
17. [Acknowledgments](#17-acknowledgments)  
18. [Conclusion](#18-conclusion)  

---

## 1) Problem Statement  
Teams drown in PDFs, web pages, and manuals. Keyword search misses context; LLMs hallucinate when they don’t “know” your documents. You need a system that understands your content, finds the right passages fast, and answers with citations—reliably and at scale.  

**Goal:** Build a Retrieval-Augmented Generation (RAG) pipeline that ingests heterogeneous documents (PDF/HTML), converts them to high-quality text, indexes them as embeddings in a vector store, and uses an LLM to answer questions grounded in the retrieved chunks.  

---

## 2) Solution Overview  
This project implements an end-to-end RAG stack using:  
- **Docling** for robust document conversion and chunking.  
- **IBM Granite**:  
  - Granite Embeddings to vectorize text for similarity search.  
  - Granite Instruct as the LLM for grounded answers.  
- **LangChain** to orchestrate embedding, storage, retrieval, and prompting.  
- **Milvus (Lite)** as the local vector database for fast semantic search.  

The reference implementation wires these components together and demonstrates grounding on a small corpus (an HTML article + a PDF), then answering questions with citations.  

---

## 3) Architecture & Workflow  

        ┌──────────────────────────┐
        │  Source Documents (PDF,  │
        │  HTML, etc.)             │
        └───────────┬──────────────┘
                    │ parse + segment
                    ▼
           ┌──────────────────┐
           │  Docling         │
           │  (convert+chunk) │
           └─────────┬────────┘
                     │ embeddings
                     ▼
       ┌──────────────────────────┐
       │ Granite Embeddings       │
       │ (HuggingFace interface)  │
       └───────────┬──────────────┘
                   │ upsert
                   ▼
       ┌──────────────────────────┐
       │ Milvus (Lite) Vector DB  │
       └───────────┬──────────────┘
                   │ retrieve top-k
                   ▼
       ┌──────────────────────────┐
       │ LangChain RAG Chain      │
       │ (stuff retrieved chunks) │
       └───────────┬──────────────┘
                   │ prompt
                   ▼
       ┌──────────────────────────┐
       │ IBM Granite Instruct LLM │
       └───────────┬──────────────┘
                   │ grounded answer
                   ▼
            Final Answer + Context



**Key Ideas:**  
- **Convert & Chunk:** Docling normalizes messy inputs to semantically clean text and chunks them for retrieval efficiency.  
- **Embed:** Text chunks embedded with `ibm-granite/granite-embedding-30m-english`.  
- **Store & Retrieve:** Stored in Milvus Lite for fast similarity search.  
- **Generate:** Granite 3.3 8B Instruct answers using retrieved context.  

---

## 4) Tech Stack  
- **Language/Runtime:** Python 3.10+  
- **Core Libraries:** `docling`, `langchain`, `langchain_community`, `langchain_huggingface`, `langchain_milvus`, `transformers`, `replicate`  
- **Models:**  
  - Embeddings → `ibm-granite/granite-embedding-30m-english`  
  - LLM → `ibm-granite/granite-3.3-8b-instruct` (via Replicate)  
- **Vector DB:** Milvus Lite  

---

## 5) Features  
✔ Ingests **web pages & PDFs** (extendable).  
✔ **Hybrid chunking** with metadata for precise retrieval.  
✔ **Local, serverless vector store** with Milvus Lite.  
✔ Configurable **LLM/embedding models** and retrieval parameters.  

---

## 6) Getting Started  

### 🔧 Prerequisites  
- Python 3.10+  
- A Replicate account + API token  
- Basic knowledge of virtual environments  

### 📦 Installation  

```bash
# 1) Create & activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install transformers langchain_community 'langchain_huggingface[full]' \
            langchain_milvus docling replicate

```

### (Optional):

```
pip install "git+https://github.com/ibm-granite-community/utils.git"
```

### 🌍 Environment Setup
```
export REPLICATE_API_TOKEN=YOUR_TOKEN_HERE
# Windows (PowerShell)
$env:REPLICATE_API_TOKEN="YOUR_TOKEN_HERE"
```

## 7) Configuration

- Embeddings: ibm-granite/granite-embedding-30m-english

- LLM: ibm-granite/granite-3.3-8b-instruct

- Vector DB: Milvus Lite (local file by default).

- Sources: Replace sample docs with your own PDFs/HTML.

## 8) How to Run

```
python granite_docling_rag.py
```

### Workflow:

1. Docling converts sources → chunks.
2. Granite embeds chunks → stored in Milvus.
3. Retriever pulls top-k chunks.
4. Granite LLM generates grounded answers with context.


## 9) Algorithms & Concepts

Docling Conversion & Chunking – token-aware hybrid chunking.

Text Embeddings – compact, high-quality vectors.

Vector Search (Milvus) – cosine similarity for retrieval.

RAG Prompting (Stuff Pattern) – chunks stuffed into prompt context.

## 10) Example: Swapping Sources & Questions

Update sources = [...] with internal docs.

Change query = "..." with your domain questions.

## 11) Extensibility

Use different LLMs or embeddings.

Try semantic chunking.

Replace Milvus with FAISS/Chroma/Zilliz Cloud.

Add citations with clickable provenance.

## 12) Evaluation & Quality

✅ Groundedness checks

⚡ Latency optimization

🔒 Safety filters

## 13) Real-World Considerations

📜 Compliance & IP – respect TOS/robots.txt

🔄 Updates – re-embed only changed files

🔑 Security – protect API tokens

## 14) Project Structure (Suggested)
```
.
├── granite_docling_rag.py          # RAG pipeline
├── data/                           # PDFs/HTML
├── requirements.txt                # Dependencies
├── README.md                       # This file
└── LICENSE                         # License

```

## 15) License

MIT License  

Copyright (c) 2025 Debadatta Rout  

Permission is hereby granted, free of charge, to any person obtaining a copy...

## 16) Certification

<img width="1332" height="840" alt="Certification" src="https://github.com/user-attachments/assets/8cab1974-399c-4f16-b5f7-44d55bb9da53" /> <p align="center"> <a href="https://skills.yourlearning.ibm.com/certificate/share/d47842dfb2ewogICJvYmplY3RUeXBlIiA6ICJBQ1RJVklUWSIsCiAgImxlYXJuZXJDTlVNIiA6ICI1MTExOTM4UkVHIiwKICAib2JqZWN0SWQiIDogIkFMTS1DT1VSU0VfMzk0NjQ3MyIKfQd3c4bd8af9-10" target="_blank"> <img src="https://img.shields.io/badge/View%20Certificate-2D9BF0?style=for-the-badge&logo=ibm&logoColor=white" alt="IBM Certificate"/> </a> </p>

## 17) Acknowledgments

IBM Granite community for RAG recipes.

Docling project for parsing + chunking.

Milvus community for vector search.

## 18) Conclusion

This project demonstrates how to ground LLM answers in your own documents using a modular RAG stack. With Docling + Granite + Milvus + LangChain, you get accurate, explainable, and scalable document-aware AI answers.

