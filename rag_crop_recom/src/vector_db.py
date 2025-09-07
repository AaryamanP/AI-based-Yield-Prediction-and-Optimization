import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class VectorDB:
    def __init__(self, embeddings_path="data/embeddings/embeddings.pkl"):
        with open(embeddings_path, "rb") as f:
            data = pickle.load(f)
        self.texts = data["texts"]
        self.titles = data["titles"]
        self.embeddings = np.array(data["embeddings"])

    def query(self, query_text: str, top_k: int = 3, model=None):
        """
        Return top_k most similar documents for a query.
        """
        if model is None:
            model = SentenceTransformer("all-MiniLM-L6-v2")
        q_emb = model.encode([query_text])
        sims = cosine_similarity(q_emb, self.embeddings)[0]
        top_idx = sims.argsort()[-top_k:][::-1]
        return [{"title": self.titles[i], "text": self.texts[i], "score": float(sims[i])} for i in top_idx]
