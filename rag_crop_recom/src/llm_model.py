import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM , BitsAndBytesConfig

class LLMModel:
    def __init__(self, model_name="google/flan-t5-base"):
        print(f"🔹 Loading LLM: {model_name} with 4-bit quantization")

        # Configure 4-bit quantization
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16",   # computation precision
            bnb_4bit_use_double_quant=True,     # double quantization (saves more memory)
            bnb_4bit_quant_type="nf4"           # NormalFloat4, best for LLMs
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model in 4-bit mode
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
        )


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
