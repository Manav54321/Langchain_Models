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
````

---

## 🗣️ Language Models

### 🔍 LLMs vs Chat Models

| Feature         | LLMs                 | Chat Models                   |
| --------------- | -------------------- | ----------------------------- |
| Input Format    | Single prompt string | Chat-style (list of messages) |
| Usage           | Completions          | Multi-turn conversations      |
| Example         | `text-davinci-003`   | `gpt-3.5-turbo`, `claude-v1`  |
| LangChain Class | `LLM`                | `ChatModel`                   |

---

### 🛑 Closed-source Language Models

| Provider  | Models                             |
| --------- | ---------------------------------- |
| OpenAI    | `gpt-3.5-turbo`, `gpt-4`           |
| Anthropic | `claude-1`, `claude-2`, `claude-3` |
| Google    | `gemini-pro`, `gemini-1.5-pro`     |

Accessed via API keys.

---

### 🌐 Open-source Language Models

| Source      | Examples                                                          |
| ----------- | ----------------------------------------------------------------- |
| HuggingFace | `tiiuae/falcon-7b-instruct`, `mistralai/Mistral-7B-Instruct-v0.1` |
| Others      | LLama 2, Zephyr, Deepseek, OpenChat                               |

**Usage:**

* HuggingFace Inference API
* Run locally using `transformers`, `AutoModel`, or `text-generation-webui`

Example:

```python
from transformers import pipeline
pipe = pipeline("text-generation", model="tiiuae/falcon-7b-instruct")
```

---

## 🧬 Embedding Models

Used to convert text into numerical vectors for:

* Semantic search
* Question answering
* Text similarity
* RAG pipelines

---

### 🛑 Closed-source Embedding Models

| Provider | Model Name               |
| -------- | ------------------------ |
| OpenAI   | `text-embedding-ada-002` |
| Gemini   | Gemini Embed             |
| Cohere   | Embed-v3                 |

---

### 🌐 Open-source Embedding Models

| Source      | Examples                                                           |
| ----------- | ------------------------------------------------------------------ |
| HuggingFace | `sentence-transformers/all-MiniLM-L6-v2`, `BAAI/bge-small-en-v1.5` |
| Usage       | `HuggingFaceEmbeddings` class in LangChain                         |

Example:

```python
from langchain.embeddings import HuggingFaceEmbeddings

model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
```

---

## 🌡️ Temperature (in Language Models)

> Controls randomness and creativity in generation.

| Temperature | Behavior         |
| ----------- | ---------------- |
| 0           | Deterministic    |
| 0.7         | Balanced         |
| 1.0         | More creative    |
| >1.0        | Often incoherent |

Higher temp → more randomness.
Lower temp → more focused and repetitive.

---

## 🛠️ Mini Project: Semantic Search

A simple retrieval-based pipeline using LangChain.

### 🔍 Objective

Given a user query, retrieve the most relevant document chunks using vector embeddings.

### 🧰 Stack Used

* `langchain`
* `sentence-transformers`
* `FAISS` vector DB

### ✅ Pipeline

1. Load and split text documents
2. Convert them to embeddings
3. Store in FAISS
4. Accept user query → embed → compare via cosine similarity → retrieve top matches

### 🚀 Run the Demo

```bash
python semantic_search_demo.py
```

---

## 📁 Project Structure

```
langchain_models/
│
├── language_models/
│   ├── openai_chat.py              # OpenAI Chat Models
│   ├── huggingface_local.py        # Local HF Model via transformers
│
├── embedding_models/
│   ├── openai_embedding.py
│   ├── huggingface_embedding.py
│
├── semantic_search_demo.py         # End-to-end semantic search pipeline
│
└── README.md
```

---

## 💡 Future Additions

* Performance benchmark across models
* Add Gemini + Claude integration
* LangSmith Tracing support
* RAG pipeline (retrieval-augmented generation)

---

## 🤝 Contributing

Pull requests are welcome. If you find issues or want to improve code/docs, feel free to submit a PR or open an issue.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙌 Author

Made with ❤️ by **[Manav Desai](https://github.com/manavdesai)**
AI Engineer | NLP & Agentic AI Enthusiast
