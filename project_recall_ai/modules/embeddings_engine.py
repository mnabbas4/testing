import os
import json
import numpy as np
from pathlib import Path
from openai import OpenAI

def _make_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment.")
    return OpenAI(api_key=api_key)

class EmbeddingsEngine:
    def __init__(self):
        try:
            self.client = _make_client()
        except Exception as e:
            print(f"⚠️ Failed to initialize OpenAI client: {e}")
            self.client = None

    def embed_texts(self, texts):
        if not self.client:
            print("⚠️ Using dummy embeddings (no API key found).")
            return [np.random.rand(1536).tolist() for _ in texts]
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"⚠️ Embedding generation failed: {e}")
            return [np.random.rand(1536).tolist() for _ in texts]

    def _text_for_row(self, row):
        # Combine text columns for embeddings
        return " ".join(str(v) for v in row.values if isinstance(v, str))

    def index_dataframe(self, memory_path, df, id_prefix="mem"):
        texts = [self._text_for_row(r) for _, r in df.iterrows()]
        embeddings = self.embed_texts(texts)
        out_path = Path(memory_path).with_suffix("").parent / f"{id_prefix}_embeddings.json"
        out_path.write_text(json.dumps(embeddings))
        print(f"✅ Saved {len(embeddings)} embeddings to {out_path}")
