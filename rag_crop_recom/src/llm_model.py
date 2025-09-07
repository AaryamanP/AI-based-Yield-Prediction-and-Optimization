import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class LLMModel:
    def __init__(self, model_name="google/flan-t5-base", device="cpu"):
        print(f"🔹 Loading LLM: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Always run on CPU to avoid MPS OOM
        self.device = torch.device(device)
        self.model.to(self.device)

    def generate(self, prompt: str, max_new_tokens=150):
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
