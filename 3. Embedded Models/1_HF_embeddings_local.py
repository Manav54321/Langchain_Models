from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

text = "Delhi is the capital of India."

vector = embeddings.embed_query(text)

print(str(vector))

# for documents:

# documents = ["Delhi is the capital of India.", "Paris is the capital of France.", "Berlin is the capital of Germany."]

# vectors = embeddings.embed_documents(documents)