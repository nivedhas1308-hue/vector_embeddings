import os
import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create ChromaDB client
client = chromadb.PersistentClient(path="chroma_db")

# Create collection
collection = client.get_or_create_collection(name="ai_documents")

# Read documents
documents = []
ids = []
metadatas = []

folder = "dataset/documents"

for filename in os.listdir(folder):
    if filename.endswith(".txt"):
        with open(os.path.join(folder, filename), "r", encoding="utf-8") as file:
            text = file.read()

        documents.append(text)
        ids.append(filename.replace(".txt", ""))

        metadatas.append({
            "title": filename,
            "category": "Artificial Intelligence",
            "source": "AI Text Documents"
        })

# Generate embeddings
embeddings = model.encode(documents).tolist()

# Store in ChromaDB
collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas
)

# Save embeddings
import numpy as np
np.save("dataset/embeddings.npy", embeddings)

print("Embeddings generated successfully!")
print("Documents stored in ChromaDB:", collection.count())

# Similarity Search
query = "How do computers learn?"

query_embedding = model.encode([query]).tolist()

results = collection.query(
    query_embeddings=query_embedding,
    n_results=5
)

print("\nTop 5 Similar Documents:")

for i, document in enumerate(results["documents"][0]):
    print(f"\n{i+1}. {document}")

    # Keyword-Based Search
query = "Artificial Intelligence"

print("\nKeyword-Based Search Results:")

for filename in os.listdir("dataset/documents"):
    if filename.endswith(".txt"):
        with open(os.path.join("dataset/documents", filename), "r", encoding="utf-8") as file:
            content = file.read()

        if query.lower() in content.lower():
            print(filename, ":", content)

# Save retrieved results
results_data = []

for i, document in enumerate(results["documents"][0]):
    results_data.append({
        "Rank": i + 1,
        "Query": "How do computers learn?",
        "Retrieved_Document": document
    })

df = pd.DataFrame(results_data)

df.to_csv("dataset/retrieved_results.csv", index=False)

print("\nRetrieved results saved successfully!")

np.save("dataset/embeddings.npy", embeddings)