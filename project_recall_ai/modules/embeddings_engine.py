import os
import json
import streamlit as st
from openai import OpenAI
from pathlib import Path
import numpy as np

# --- Load API key from Streamlit secrets or environment ---
def get_api_key():
    if "OPENAI_API_KEY" in st.secrets:
        return st.secrets["OPENAI_API_KEY"]
    elif os.getenv("OPENAI_API_KEY"):
        return os.getenv("OPENAI_API_KEY")
    return None


def get_openai_client():
    """
    Safe lazy initialization of the OpenAI client.
    Works in both local and Streamlit Cloud environments.
    """
    key = get_api_key()
    if not key:
        return None
    try:
        return OpenAI(api_key=key)
    except Exception as e:
        st.warning(f"⚠️ Failed to initialize OpenAI client: {e}")
        return None


class EmbeddingsEngine:
    def __init__(self):
        self.client = get_openai_client()

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
