# ingestion.py (debug version)
import os
from pathlib import Path
from typing import List

def load_docs(data_dir: str = "data/docs") -> List[dict]:
    docs = []
    for path in Path(data_dir).glob("*"):
        print(f"📄 Checking file: {path}")  # DEBUG
        if path.suffix.lower() == ".pdf":
            try:
                import pdfplumber
                text = ""
                with pdfplumber.open(str(path)) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text() or ""
                        text += page_text
                print(f"✅ Extracted {len(text)} chars from {path.name}")
                if text.strip():
                    docs.append({"title": path.name, "text": text})
                else:
                    print(f"⚠️ No text extracted from {path}, likely scanned PDF")
            except Exception as e:
                print(f"❌ Failed to read {path}: {e}")
        elif path.suffix.lower() == ".txt":
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                print(f"✅ Loaded TXT {path.name} with {len(text)} chars")
                if text.strip():
                    docs.append({"title": path.name, "text": text})
            except Exception as e:
                print(f"❌ Failed to read {path}: {e}")
    print(f"📊 Loaded {len(docs)} documents in total")
    return docs
