# Document Similarity Search

This project demonstrates how vector embeddings and cosine similarity can be used to perform semantic document retrieval. It uses HuggingFace sentence transformers to convert both queries and documents into dense vector representations and then compares them using cosine similarity to find the most semantically relevant result.

---

## Overview

Traditional search methods rely on keyword matching, which often fails to understand the meaning behind a query. Document similarity search addresses this by embedding both documents and queries into high-dimensional vector space. These embeddings capture the semantic meaning of the text, allowing the system to find the most contextually relevant match.

---

## How It Works

1. **Embeddings**
   All documents and the input query are passed through a pre-trained transformer model to generate dense vector embeddings. These vectors numerically represent the semantic content of the text.

2. **Similarity Calculation**
   Using cosine similarity, the system measures how close the query vector is to each document vector. This results in a similarity score for each document.

3. **Best Match**
   The document with the highest similarity score is selected and returned as the most relevant response to the user's query.

---

## Example Query

Given a query like:
`"Who's Ana de Armas?"`

The system returns:
`"Ana de Armas exudes natural elegance and allure — her features, accent, and poise make her captivating on and off screen."`

With a similarity score (e.g., 0.86).

---

## Use Cases

* AI chat interfaces
* Smart document Q\&A systems
* Semantic product or content recommendation
* Legal or biomedical document retrieval
* Educational assistants for large textbooks or articles

---

## Setup

Clone the repo and install dependencies:

```bash
git clone https://github.com/your-username/document-similarity-search
cd document-similarity-search
pip install -r requirements.txt
```

Then run the app:

```bash
streamlit run app.py
```

---
