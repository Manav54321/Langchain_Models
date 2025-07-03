from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import gradio as gr

# Embedding model setup
embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

# Documents
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

# Core function to return top matching document
def find_similar_document(query):
    if not query.strip():
        return "Please enter a query.", "", ""
    
    doc_embeddings = embeddings.embed_documents(documents)
    query_embedding = embeddings.embed_query(query)

    scores = cosine_similarity([query_embedding], doc_embeddings)[0]
    sorted_scores = list(enumerate(scores))
    document_index, score = sorted(sorted_scores, key=lambda x: x[1])[-1]

    return (
        f"query: {query}",
        f"best_match: {documents[document_index]}",
        f"similarity_score: {round(score, 4)}"
    )

# Gradio Interface
demo = gr.Interface(
    fn=find_similar_document,
    inputs=gr.Textbox(label="Ask a query (e.g., who's Kylie?)", placeholder="Enter your query here..."),
    outputs=[
        gr.Textbox(label="your_query"),
        gr.Textbox(label="your_best_match"),
        gr.Textbox(label="your_similarity_score")
    ],
    title="Document Similarity Search",
    description="Document similarity search is a natural language processing technique that retrieves the most semantically relevant content from a collection of texts. Instead of matching exact words, it relies on vector embeddings — dense numerical representations of text that capture contextual meaning. By converting both user queries and documents into this high-dimensional space, we can compute similarity using cosine distance to find content that best aligns with the user's intent. This approach powers intelligent retrieval systems behind modern AI assistants, search engines, and document understanding tools."
)

# Launch the app
demo.launch()