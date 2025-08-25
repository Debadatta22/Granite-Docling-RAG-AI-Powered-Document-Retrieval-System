# Granite-Docling-RAG-AI-Powered-Document-Retrieval-System
A production-ready README you can drop into your repo. It explains the problem, the solution, the workflow, and exactly how to run and extend the lab you just completed.

--------------

-------------

1) Problem Statement

Teams drown in PDFs, web pages, and manuals. Keyword search misses context; LLMs hallucinate when they don’t “know” your documents. You need a system that understands your content, finds the right passages fast, and answers with citations—reliably and at scale.

Goal: Build a Retrieval-Augmented Generation (RAG) pipeline that ingests heterogeneous documents (PDF/HTML), converts them to high-quality text, indexes them as embeddings in a vector store, and uses an LLM to answer questions grounded in the retrieved chunks.

2) Solution Overview

This project implements an end-to-end RAG stack using:

Docling for robust document conversion and chunking.

IBM Granite:

Granite Embeddings to vectorize text for similarity search.

Granite Instruct as the LLM for grounded answers.

LangChain to orchestrate embedding, storage, retrieval, and prompting.

Milvus (Lite) as the local vector database for fast semantic search.
The reference implementation wires these components together and demonstrates grounding on a small corpus (an HTML article + a PDF), then answering questions with citations.

3) Architecture & Workflow

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


Key ideas

Convert & Chunk: Docling normalizes messy inputs to semantically clean text and chunks them for retrieval efficiency. 

Embed: Text chunks are embedded using ibm-granite/granite-embedding-30m-english for compact, high-quality vectors. 

Store & Retrieve: Vectors are stored in Milvus (file-backed “Lite” mode) for fast similarity search—no external server needed. 

Generate: A Granite 3.3 8B Instruct model answers using retrieved context via LangChain’s “stuff documents” pattern. 

4) Tech Stack

Language/Runtime: Python 3.10+

Core libs: docling, langchain, langchain_community, langchain_huggingface, langchain_milvus, transformers, replicate

Models:

Embeddings: ibm-granite/granite-embedding-30m-english

LLM: ibm-granite/granite-3.3-8b-instruct (via Replicate) 

Vector DB: Milvus (Lite via langchain_milvus) with local file URI. 

5) Features

Ingests web pages and PDFs (extendable to other formats supported by Docling). 

Hybrid chunking with metadata for precise retrieval. 

Local, serverless vector store (Milvus Lite) for easy dev. 

Configurable LLM/embedding models and retrieval parameters.

6) Getting Started
Prerequisites

Python 3.10+

A Replicate account + API token (for Granite via Replicate).

Basic understanding of virtual environments.

Installation

```
# 1) Create & activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install transformers langchain_community 'langchain_huggingface[full]' \
            langchain_milvus docling replicate
# (Optional) utils from IBM Granite community if you use their helpers
# pip install "git+https://github.com/ibm-granite-community/utils.git"
```

Environment

Set your Replicate token so the LLM can run:
```
export REPLICATE_API_TOKEN=YOUR_TOKEN_HERE
# Windows (PowerShell): $env:REPLICATE_API_TOKEN="YOUR_TOKEN_HERE"
```

7) Configuration

The reference script uses:

Embeddings: ibm-granite/granite-embedding-30m-english

LLM: ibm-granite/granite-3.3-8b-instruct
You can change these in the code where HuggingFaceEmbeddings and Replicate are initialized. 

Vector DB: A Milvus Lite database file is created automatically via connection_args={"uri": <tempfile>}. Replace with a persistent path in production. 

Sources: The sample uses two public sources (an event article and a rules PDF) to demonstrate grounding. Replace sources = [...] with your own URLs/paths. 

8) How to Run
```
python granite_docling_rag.py

```

What happens:

Docling converts your sources and HybridChunker segments text into retrieval-friendly chunks with metadata. 

Chunks are embedded using Granite Embeddings and upserted into Milvus. 

A LangChain retriever pulls top-k chunks for the query. 

A Granite Instruct model receives a prompt built with the retrieved context and returns a grounded answer. 

9) Algorithms & Concepts

Document Conversion & Chunking:
Docling normalizes layout, extracts text, and HybridChunker uses a token-aware heuristic to produce chunks that balance coherence and retrieval quality. Metadata preserves source and chunk doc_id for traceability. 

Text Embeddings:
granite-embedding-30m-english maps chunks to dense vectors. Similarity (typically cosine/inner-product) powers semantic search. 

Vector Search (Milvus):
Chunks are stored with an AUTOINDEX; top-k nearest neighbors are retrieved for a query vector to build context windows. 

RAG Prompting (“Stuff” pattern):
Retrieved texts are concatenated into a TokenizerChatPromptTemplate and sent to Granite Instruct. LangChain’s create_retrieval_chain orchestrates retrieval + generation. 

10) Example: Swapping Sources & Questions

Update sources = [...] with internal docs (policies, manuals, knowledge bases).

Change query = "..." to reflect your domain (“What is our return policy for enterprise customers?”).
The same pipeline will return grounded answers with the right chunks attached. 

11) Extensibility

Different Models: Plug in other Granite sizes or providers (Hugging Face TGI, watsonx, etc.) by swapping the LLM and embeddings initializers. 

Chunking Strategies: Try semantic chunking or overlap tuning for better recall/precision. 

Vector Stores: Replace Milvus with FAISS, Chroma, pgvector, or Zilliz Cloud via LangChain adapters. 

Citations: Echo document.metadata['source'] in final answers for clickable provenance.

12) Evaluation & Quality

Groundedness: Manually verify that the answer’s statements appear in retrieved chunks.

Latency: Cache embeddings; persist the Milvus DB; pre-warm the LLM.

Safety: Add content filters/governance before user-facing deployment.

13) Real-World Considerations

Compliance & IP: Make sure you have the right to crawl/index sources; honor website robots/TOS.

Updates: Re-embed only changed files; keep a background indexer.

Security: Protect API keys via environment and secret managers.

14) Project Structure (suggested)
```
.
├── granite_docling_rag.py          # Reference script (RAG pipeline)
├── data/                           # Local PDFs/HTML (optional)
├── requirements.txt                # Pinned deps for reproducibility
├── README.md                       # This file
└── LICENSE                         # Your chosen license
```

16) License

MIT License

Copyright (c) 2025 Debadatta Rout

Permission is hereby granted, free of charge, to any person obtaining a copy
...

## Certification

<img width="1332" height="840" alt="Screenshot 2025-08-25 213229" src="https://github.com/user-attachments/assets/8cab1974-399c-4f16-b5f7-44d55bb9da53" />


17) Acknowledgments

IBM Granite community examples and recipes for RAG pipelines and LangChain components. 

Docling project for high-fidelity document conversion and chunking. 

Milvus community for scalable vector search and a convenient Lite mode. 

18) Conclusion

This project shows how to ground LLM answers in your own documents using a clean, modular RAG stack. With Docling for robust parsing, Granite for strong embeddings and instruction-following, Milvus for lightning-fast search, and LangChain for orchestration, you get accurate, explainable answers that scale from notebooks to production services. Swap in your documents, tune chunking and retrieval, and you’re ready to deploy.


