# 🧠 LangChain Models Component

This project explores the different types of models supported by [LangChain](https://www.langchain.com/) and how to use them effectively for language-based tasks such as generation, embeddings, and semantic search.

Whether you're working with OpenAI, HuggingFace, or your own local models — this repo demonstrates how to integrate them with LangChain in a clean and modular way.

---

## 📚 What Are "Models" in LangChain?

LangChain uses **models** for:

- **Language generation** (e.g., ChatGPT, Claude)
- **Embeddings** (e.g., for similarity search, clustering, RAG pipelines)

Two main categories:
1. **Language Models (LLMs & Chat Models)**
2. **Embedding Models**

---

## 🧩 Types of Models in LangChain

```mermaid
graph TD
    A[Models] --> B[Language Models]
    A --> C[Embedding Models]
    B --> B1[LLMs]
    B --> B2[Chat Models]
