import os
import streamlit as st
from openai import OpenAI
from pathlib import Path
import json
import numpy as np

# --- Get API key (works both locally and on Streamlit Cloud) ---
def _get_openai_key():
    try:
        if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY")

# --- Safe client creation ---
def _make_client():
    key = _get_openai_key()
    if not key:
        st.warning("⚠️ No OpenAI API key found. Please add it in Streamlit → Settings → Secrets.")
        return None
    try:
        return OpenAI(api_key=key, max_retries=2, timeout=60)
    except TypeError:
        try:
            return OpenAI(api_key=key)
        except Exception as e:
            st.error(f"Failed to initialize OpenAI client: {e}")
            return None

class EmbeddingsEngine:
    def __init__(self):
        self.client = _make_client()

    def _text_for_row(self, row):
        cols = [
            'Project Category',
            'Project Reference',
            'Phase',
            'Problems Encountered',
            'Solutions Adopted'
        ]
        return " | ".join([str(row.get(c, "")) for c in cols if row.get(c, "")])

    def embed_texts(self, texts):
        if not self.client:
            st.warning("⚠️ Using dummy embeddings (no API key found).")
            return [np.random.rand(1536).tolist() for _ in texts]

        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-large",
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            st.error(f"Embedding generation failed, falling back to dummy embeddings: {e}")
            return [np.random.rand(1536).tolist() for _ in texts]

    def index_dataframe(self, memory_path, df, id_prefix='mem'):
        texts = [self._text_for_row(r) for _, r in df.iterrows()]
        embeddings = self.embed_texts(texts)
        out_path = Path(memory_path).with_suffix('').parent / f"{id_prefix}_embeddings.json"
        out_path.write_text(json.dumps(embeddings))
        return str(out_path)
