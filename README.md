# Document Similarity Search with Sentence Transformers

This repository presents a minimal and effective implementation of semantic document retrieval using vector embeddings and cosine similarity. The system is designed to identify the document that best matches the meaning of a user's input query.

---

## Overview

Traditional search methods rely on keyword overlap. This method improves upon that by leveraging **pretrained transformer-based sentence embeddings**, allowing retrieval based on *semantic similarity* rather than lexical similarity.

The process involves:

* Embedding each document and query into a high-dimensional space.
* Computing cosine similarity between the query vector and document vectors.
* Returning the document with the highest semantic alignment.

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

## Tech Stack

* `langchain_huggingface` for transformer-based embeddings
* `scikit-learn` for cosine similarity
* `streamlit` for the user interface
* `matplotlib` for basic vector visualization

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

## File Structure

```
.
├── app.py                # Main Streamlit application
├── README.md             # Project description and documentation
├── requirements.txt      # Python dependencies
└── assets/               # (Optional) images or static files
```

---