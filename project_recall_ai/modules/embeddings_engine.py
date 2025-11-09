# modules/embeddings_engine.py
import os
import json
import pandas as pd
from pathlib import Path
from openai import OpenAI



def get_openai_client():
    """
    Safe lazy initialization of the OpenAI client.
    Prevents Streamlit Cloud import errors and proxy issues.
    """
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    try:
        return OpenAI(api_key=key)
    except Exception:
        return None


class EmbeddingsEngine:
    def __init__(self):
        self.client = get_openai_client()

    def _text_for_row(self, row):
        """
        Convert row dict to a unified text string representation.
        Used for embedding generation.
        """
        cols = ['Project Category', 'Project Reference', 'Phase',
                'Problems Encountered', 'Solutions Adopted']
        return " | ".join([str(row.get(c, "")) for c in cols if row.get(c, "")])

    def embed_texts(self, texts):
        """
        Generate embeddings for a list of texts using OpenAI Embeddings API.
        Falls back gracefully if no API key is configured.
        """
        if not self.client:
            raise RuntimeError("OpenAI client not configured. Set OPENAI_API_KEY in Streamlit secrets.")

        try:
            resp = self.client.embeddings.create(
                model="text-embedding-3-large",
                input=texts
            )
            return [d.embedding for d in resp.data]
        except Exception as e:
            raise RuntimeError(f"Embedding generation failed: {e}")

    def index_dataframe(self, memory_path, df, id_prefix='mem'):
        """
        Embed and store the full dataframe as a list of embeddings JSON file.
        """
        texts = [self._text_for_row(r) for _, r in df.iterrows()]
        embeddings = self.embed_texts(texts)

        out_path = Path(memory_path).with_suffix('').parent / f"{id_prefix}_embeddings.json"
        out_path.write_text(json.dumps(embeddings))
        return str(out_path)


