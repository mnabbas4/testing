import os
import json
import numpy as np
from pathlib import Path
import os
import streamlit as st


# ✅ Load key into environment BEFORE importing anything else
try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
        #st.sidebar.success("✅ OpenAI key loaded.")
    else:
        st.warning("⚠️ OPENAI_API_KEY not found in Streamlit secrets.")
except Exception as e:
    st.error(f"Error loading secrets: {e}")
# Streamlit is optional — imported safely
try:
    import streamlit as st
except ImportError:
    st = None

from openai import OpenAI


def _get_openai_key():
    """
    Tries every possible source for the API key:
    1. Streamlit secrets
    2. Environment variable
    3. Cached global (if set later in runtime)
    """
    key = None
    try:
        if st and hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
            key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass

    # fallback: environment
    if not key:
        key = os.getenv("OPENAI_API_KEY")

    # final fallback: config cache file (optional safety net)
    if not key and Path(".openai_key_cache").exists():
        key = Path(".openai_key_cache").read_text().strip()

    if key and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = key
    return key


#def _make_client():

#    Creates OpenAI client safely with all fallback handling.
    
#    key = _get_openai_key()
#    if not key:
#        if st:
#            st.warning("⚠️ No OpenAI API key found in Streamlit secrets or environment.")
#        return None

#    try:
#        client = OpenAI(api_key=key)
#        return client
#    except Exception as e:
#        if st:
#            st.error(f"Failed to initialize OpenAI client: {e}")
#        return None


class EmbeddingsEngine:
    def __init__(self):
        self.client = _make_client()

    def _text_for_row(self, row):
        cols = [
            "Project Category",
            "Project Reference",
            "Phase",
            "Problems Encountered",
            "Solutions Adopted",
        ]
        return " | ".join([str(row.get(c, "")) for c in cols if row.get(c, "")])

    def embed_texts(self, texts):
        if not self.client:
            if st:
                st.warning("⚠️ Using dummy embeddings (no API key found).")
            return [np.random.rand(1536).tolist() for _ in texts]

        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-large", input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            if st:
                st.error(f"Embedding generation failed: {e}")
            return [np.random.rand(1536).tolist() for _ in texts]

    def index_dataframe(self, memory_path, df, id_prefix="mem"):
        texts = [self._text_for_row(r) for _, r in df.iterrows()]
        embeddings = self.embed_texts(texts)
        out_path = Path(memory_path).with_suffix("").parent / f"{id_prefix}_embeddings.json"
        out_path.write_text(json.dumps(embeddings))
        return str(out_path)




