# AI-Powered-Document-Assistant

### Retrieval-Augmented Generation (RAG) System with Google Drive Integration

## Project Overview

This project implements a **Retrieval-Augmented Generation (RAG) system** that allows users to ask natural language questions about documents stored in their **Google Drive**. Users connect a Google account and select a folder that acts as their **personal knowledge base**. The system retrieves relevant information from the user's documents and uses a locally hosted large language model (LLM) to generate answers grounded in those documents.

The goal of this project is to demonstrate how modern generative AI systems can be combined with **document retrieval, embeddings, and language models** to build a practical AI application that improves access to personal information.

---

# Problem Statement

Modern users store large amounts of information in cloud storage platforms like Google Drive. Finding specific information within these documents often requires manually searching through multiple files.

Traditional search tools rely on keyword matching and cannot:

* answer complex questions
* synthesize information across documents
* understand context or semantics

This project addresses this problem by building an AI assistant capable of answering questions using the user's own documents as context.

---

# Project Goals

The primary goals of this project are to:

1. Build a multi-user AI system that connects to Google Drive folders.
2. Implement a Retrieval-Augmented Generation (RAG) pipeline.
3. Allow users to ask questions about their personal document collections.
4. Ensure answers are grounded in retrieved document content.
5. Evaluate system performance and accuracy under different configurations.

---

# Key Features

## Google Drive Integration

Users authenticate with Google and grant permission to access their Google Drive.

The system can:

* access a specified folder
* retrieve documents from the folder
* process supported file types

Supported document formats include:

* Google Docs
* PDF
* TXT
* Markdown

---

## Automatic Knowledge Base Creation

Documents retrieved from Google Drive are automatically processed:

1. Documents are downloaded.
2. Text is extracted.
3. Documents are split into smaller chunks.
4. Chunks are converted into vector embeddings.
5. Embeddings are stored in a vector database.

This allows efficient semantic search over the document collection.

---

## Retrieval-Augmented Generation (RAG)

When a user asks a question:

1. The question is converted into an embedding.
2. The system retrieves the most relevant document chunks.
3. Retrieved content is passed to a language model.
4. The language model generates an answer grounded in the retrieved text.

This approach reduces hallucination and ensures responses are based on real documents.

---

## Source Citations

Responses include references to the source documents that contributed to the answer. This allows users to verify information and increases system transparency.

Example output:

```
Answer:
The project deadline is May 2.

Sources:
Project_Syllabus.pdf (Page 3)
Meeting_Notes.docx (Paragraph 2)
```

---

## Multi-User System

The system supports multiple users.

Each user has:

* their own Google Drive connection
* their own document knowledge base
* isolated vector embeddings

This ensures user data remains separate.

---

# System Architecture

```
User
 ↓
Web Interface
 ↓
Authentication (Google OAuth)
 ↓
Backend API
 ↓
Google Drive API
 ↓
Document Processing Pipeline
 ↓
Text Chunking
 ↓
Embedding Generation
 ↓
Vector Database
 ↓
Retrieval-Augmented Generation
 ↓
LLM Response
```

---

# Technology Stack

## Backend

* Python
* FastAPI (API framework)

Responsibilities:

* authentication
* document processing
* RAG pipeline
* API endpoints

---

## Frontend

* React or lightweight web UI

Responsibilities:

* user login
* folder selection
* question input
* displaying answers

---

## Language Model

The system runs a local open-source LLM rather than relying on paid APIs (TBD)

Advantages:

* reduced cost
* improved privacy
* full control over the model

---

## Embeddings

Documents are converted into vector embeddings that capture semantic meaning. These embeddings allow the system to retrieve the most relevant text passages for a given query.

---

## Vector Database

A vector search system stores embeddings and enables fast similarity search across document chunks.

---

# Retrieval-Augmented Generation Pipeline

The RAG pipeline consists of several stages.

## 1. Document Retrieval

Documents are retrieved from the user's Google Drive folder.

## 2. Text Extraction

Text content is extracted from each document.

## 3. Document Chunking

Documents are divided into smaller segments (e.g., 300–500 tokens).

Chunking improves retrieval accuracy.

## 4. Embedding Generation

Each chunk is converted into a numerical vector representing its semantic meaning.

## 5. Vector Storage

Embeddings are stored in a vector database.

## 6. Query Processing

When a question is asked:

* the query is embedded
* similar document chunks are retrieved

## 7. Response Generation

The language model generates a response using the retrieved context.

---

# Evaluation Plan

To evaluate the system, several experiments will be conducted.

## Experiment 1: Chunk Size Impact

Test different chunk sizes to measure how retrieval performance changes.

Examples:

* 200 tokens
* 500 tokens
* 1000 tokens

Metrics:

* response accuracy
* retrieval relevance

---

## Experiment 2: RAG vs Non-RAG

Compare answers generated:

1. With document retrieval
2. Without document retrieval

Goal:

Measure how retrieval improves factual accuracy and reduces hallucinations.

---

## Experiment 3: Retrieval Quality

Evaluate whether the retrieved documents actually contain the correct information needed to answer the question.

Metrics:

* retrieval precision
* retrieval recall

---

# Security and Privacy Considerations

User data must be handled responsibly.

Key considerations:

* OAuth authentication ensures secure login.
* Access tokens are stored securely.
* Users can only access their own documents.
* Document data is not shared between users.

---

# Potential Extensions

Future improvements could include:

* automatic folder synchronization
* document change detection
* conversation memory
* improved ranking algorithms
* support for additional file formats

---

# Expected Outcomes

The final system will demonstrate:

* a functional AI application
* integration of generative AI with external data sources
* a scalable architecture for personal knowledge retrieval

Users will be able to interact with their personal documents through natural language queries.

---

# Deliverables

The final project will include:

1. Source code
2. System documentation
3. Evaluation results
4. Demonstration of the working application

