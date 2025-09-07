import os
import pickle
from sentence_transformers import SentenceTransformer
from src.ingestion import load_docs

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDINGS_PATH = "data/embeddings/embeddings.pkl"

def generate_embeddings(docs: list, save_path: str = EMBEDDINGS_PATH):
    if not docs:
        print("❌ No documents found. Exiting without saving embeddings.")
        return

    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [doc["text"] for doc in docs]
    print(f"📊 Generating embeddings for {len(texts)} docs...")
    embeddings = model.encode(texts, show_progress_bar=True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump({"texts": texts, "titles": [doc["title"] for doc in docs], "embeddings": embeddings}, f)

    print(f"✅ Saved embeddings to {save_path}")

if __name__ == "__main__":
    docs = load_docs("data/docs")
    generate_embeddings(docs)
