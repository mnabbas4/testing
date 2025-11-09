import os
import json
from pathlib import Path
import numpy as np
import streamlit as st
from openai import OpenAI
OPENAI_API_KEY = "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"
# --- Safe client init ---
def get_openai_client():
    key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    if not key:
        st.error("❌ No OpenAI API key found in Streamlit secrets.")
        return None
    try:
        # Compatible with openai<1.50 (no proxy args)
        return OpenAI(api_key=key)
    except Exception as e:
        st.error(f"⚠️ OpenAI client init failed: {e}")
        return None



class EmbeddingsEngine:
    def __init__(self):
        

    def _text_for_row(self, row):
        """
        Convert row dict to a unified text string representation.
        Used for embedding generation.
        """
        cols = [
            'Project Category',
            'Project Reference',
            'Phase',
            'Problems Encountered',
            'Solutions Adopted'
        ]
        return " | ".join([str(row.get(c, "")) for c in cols if row.get(c, "")])

    def embed_texts(self, texts):
        """
        Generate embeddings for a list of texts using OpenAI Embeddings API.
        """
        if not self.client:
            raise RuntimeError("❌ OpenAI client not configured. Set OPENAI_API_KEY in Streamlit secrets.")

        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-large",
                input=texts
            )
            return [item.embedding for item in response.data]
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



