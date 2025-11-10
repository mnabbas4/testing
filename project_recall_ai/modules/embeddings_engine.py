import os
import streamlit as st
from openai import OpenAI
from pathlib import Path
import json
import numpy as np

# --- Get API key (works both locally and on Streamlit Cloud) ---
def _get_openai_key():
    # Try Streamlit secrets first
    try:
        if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    # Fallback to environment variable
    return os.getenv("OPENAI_API_KEY")

# --- Safe client creation ---
def _make_client():
    key = _get_openai_key()
    if not key:
        st.warning("⚠️ No OpenAI API key found. Please add it in Streamlit → Settings → Secrets.")
        return None
    try:
        # Explicitly disable proxies (fixes 'unexpected keyword argument proxies')
        return OpenAI(api_key=key, max_retries=2, timeout=60)
    except TypeError:
        # Some Streamlit environments inject proxy params — handle gracefully
        try:
            return OpenAI(api_key=key)
        except Exception as e:
            st.error(f"Failed to initialize OpenAI client: {e}")
            return None

client = _make_client()

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





