import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class VectorDB:
    def __init__(self, embeddings_path="data/embeddings/embeddings.pkl"):
        with open(embeddings_path, "rb") as f:
            data = pickle.load(f)
        self.texts = data["texts"]
        self.titles = data["titles"]
        self.embeddings = np.array(data["embeddings"]).astype("float32")

        # Create FAISS index
        dim = self.embeddings.shape[1]  # embedding dimension
        self.index = faiss.IndexFlatIP(dim)  # inner product for cosine
        faiss.normalize_L2(self.embeddings)  # normalize for cosine similarity
        self.index.add(self.embeddings)

        # Keep a reference model if needed
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def query(self, query_text: str, top_k: int = 3, model=None):
        """
        Return top_k most similar documents for a query using FAISS.
        """
        if model is None:
            model = self.model

        q_emb = model.encode([query_text], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_emb)

        # Search in FAISS
        scores, idxs = self.index.search(q_emb, top_k)
        idxs = idxs[0]
        scores = scores[0]

        results = []
        for i, score in zip(idxs, scores):
            results.append({
                "title": self.titles[i],
                "text": self.texts[i],
                "score": float(score)
            })
        return results
