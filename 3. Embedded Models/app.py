from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import streamlit as st

embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

documents = [
    "Margot Robbie blends elegance with charisma — her Barbie-core look and magnetic screen presence make her a standout star.",
    "Megan Fox is iconic for her bold persona, confident attitude, and striking looks that continue to captivate fans worldwide.",
    "Kylie Jenner commands attention with her signature style, glamorous beauty, and trendsetting influence on social media.",
    "Kendall Jenner redefines runway appeal — tall, poised, and effortlessly stylish in every major fashion event.",
    "Scarlett Johansson is timeless — known for her powerful performances, red carpet presence, and unforgettable allure.",
    "Sydney Sweeney brings a unique blend of charm and intensity — her expressive eyes and breakout roles have made her a modern icon.",
    "Ana de Armas exudes natural elegance and allure — her soft features, accent, and poise make her captivating on and off screen.",
    "Emily Ratajkowski owns the fashion space with her strong confidence, sculpted features, and empowering presence.",
    "Shraddha Kapoor radiates warmth and grace — her expressive eyes and sweet personality make her universally adored.",
    "Kriti Sanon brings height, glam, and versatility — seamlessly balancing bold fashion and strong on-screen presence.",
    "Disha Patani is a fitness and fashion powerhouse — known for her toned physique, beach looks, and confident vibe.",
    "Kiara Advani is elegant yet bold — with a calm demeanor and powerful screen moments that leave a lasting impression.",
    "Janhvi Kapoor blends youthful energy with bold choices — growing into a confident and dynamic presence in Bollywood.",
    "Tara Sutaria combines a soft, melodic voice with striking style — a rising name in both music and cinema.",
    "Ananya Panday shows fresh charm with evolving fashion — bringing youthful appeal and growing confidence to every appearance.",
    "Rashmika Mandanna is beloved for her natural smile and girl-next-door charm — effortlessly winning hearts across industries.",
    "Samantha Ruth Prabhu is a powerhouse of strength, beauty, and elegance — her career glow-up inspires millions."
]

# ------------------------------------------------------------------------------
st.title("Document Similarity Search")

# Streamlit part i
st.markdown("""
### What is Document Similarity Search?

Think of it as your AI-powered index finder. Instead of skimming a whole document or book, it narrows down meaning — fast.

Behind the scenes, both the documents and your query get transformed into **vector embeddings** — basically, high-dimensional numerical fingerprints of their meaning.

We then measure how close those fingerprints are using **cosine similarity**. The closer the vectors, the closer the meaning.

Simple idea. Powerful results. From PDF Q&A to smarter chatbots — it's how machines find what *really* matters.
""")

# Streamlit part ii
import matplotlib.pyplot as plt

# Sample 3D vectors
doc_vectors = np.array([
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
    [0.7, 0.8, 0.5],
    [0.3, 0.9, 0.2],
    [0.9, 0.1, 0.4]
])
query_vector = np.array([0.5, 0.5, 0.5])

# Plot just the x and y dimensions
fig, ax = plt.subplots(figsize=(7, 5))

# Documents
ax.scatter(doc_vectors[:, 0], doc_vectors[:, 1], color='blue', s=100, label='Documents')

# Query
ax.scatter(query_vector[0], query_vector[1], color='red', s=150, label='Query', marker='o')

# Annotations
for i, vec in enumerate(doc_vectors):
    ax.text(vec[0] + 0.01, vec[1], f'Doc {i}', fontsize=9)
ax.text(query_vector[0] + 0.01, query_vector[1], 'Query', fontsize=10, color='red')

# Style
ax.set_xlabel("Dim 1 (X)")
ax.set_ylabel("Dim 2 (Y)")
ax.set_title("2D Projection of 3D Embedding Space")
ax.grid(True)
ax.legend()

st.pyplot(fig)

# streamlit part iii
st.markdown("""
Document similarity search powers everything from AI chatbots and smarter search engines to recommendation systems and document Q&A tools. Instead of relying on keywords, it matches based on meaning — helping systems find the most relevant content. Whether you're digging through legal docs, exploring product matches, or chatting with a PDF, this technique makes it fast and context-aware.
""")

# ---------------------------------------------------------------------------------
query = st.text_input("Ask a query based on actresses like 'who's Ana de Armas?'")

doc_embeddings = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

sorted_scores = list(enumerate(scores))

document_index, score = sorted(sorted_scores, key=lambda x: x[1])[-1]

# ---------------------------------------------------------------------------------
if st.button("Search"):
    st.write(f"Query: {query}")
    st.write(f"Output: {documents[document_index]}")
    st.write(f"Score: {score}")

