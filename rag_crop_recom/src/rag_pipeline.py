from src.vector_db import VectorDB
from src.llm_model import LLMModel
from typing import Dict
from src.utils import safe_parse_json

CHUNK_MAX_CHARS = 400   # roughly 300 tokens

PROMPT_TEMPLATE = """
You are an expert agricultural advisor.
Based on the following information, give a clear recommendation with two parts:
1. Best Action (most effective improvement for yield in this district with given soil & fertilizer)
2. Conservative Action (low-risk, resource-saving alternative)

Crop: {crop}
District: {district}
Predicted Yield: {pred_yield} kg/ha
Soil Composition: {soil}
Fertilizer Use: {fertilizer}

Relevant Documents:
{docs}

Important:
- Only base the answer on the documents and crop information above.
- Do not copy random sentences.
- Think in terms of practical, agronomic advice (irrigation, fertilizer adjustment, crop management).
- Output strictly in JSON format like this:
{{
  "best_action": "...",
  "conservative_action": "..."
}}
"""



class RAGCropRecommender:
    def __init__(self, embeddings_path="data/embeddings/embeddings.pkl",
                 model_name="google/flan-t5-base", device="cpu"):
        self.vdb = VectorDB(embeddings_path=embeddings_path)
        self.llm = LLMModel(model_name=model_name)

    def _make_excerpt(self, doc_text: str, max_chars: int = CHUNK_MAX_CHARS) -> str:
        snippet = doc_text.strip().replace("\n", " ")
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars].rsplit(" ", 1)[0] + "..."
        return snippet

    def get_recommendation(self, crop_name: str, district_name: str,
                           soil_dict: Dict[str, float], fert_dict: Dict[str, float],
                           predicted_yield: float, top_k: int = 3):

        # 1) Build a retrieval query string (simpler, not the full prompt!)
        retrieval_query = f"Crop: {crop_name}, District: {district_name}, Yield: {predicted_yield}"

        # 2) Retrieve top-k docs
        retrieved = self.vdb.query(retrieval_query, top_k=top_k)

        # 3) Create context from excerpts
        excerpts_list = []
        for d in retrieved:
            snippet = self._make_excerpt(d["text"], max_chars=CHUNK_MAX_CHARS)
            excerpts_list.append(f"Title: {d['title']}\nSnippet: {snippet}")
        context = "\n\n".join(excerpts_list) if excerpts_list else "No document excerpts found."

        # 4) Fill the actual prompt template
        prompt = PROMPT_TEMPLATE.format(
            crop=crop_name,
            district=district_name,
            pred_yield=predicted_yield,
            soil=soil_dict,
            fertilizer=fert_dict,
            docs=context
        )

        # 5) Generate with LLM
        raw_output = self.llm.generate(prompt, max_new_tokens=200)
        parsed_output = safe_parse_json(raw_output)

        # 6) Return structured result
        return {
            "query": prompt.strip(),
            "retrieved_docs": [d["title"] for d in retrieved],
            "recommendation": parsed_output
        }
