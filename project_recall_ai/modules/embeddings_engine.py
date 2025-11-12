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
        return [self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=text
                ).data[0].embedding for text in texts]

    def index_dataframe(self, memory_path, df, id_prefix="mem"):
        texts = [self._text_for_row(r) for _, r in df.iterrows()]
        embeddings = self.embed_texts(texts)
        out_path = Path(memory_path).with_suffix("").parent / f"{id_prefix}_embeddings.json"
        out_path.write_text(json.dumps(embeddings))
