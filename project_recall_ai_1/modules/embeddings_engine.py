# modules/embeddings_engine.py
import os
import json
from pathlib import Path
from openai import OpenAI
OPENAI_API_KEY = "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"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    client = None

class EmbeddingsEngine:
    def __init__(self):
        if client is None:
            raise ValueError("OPENAI_API_KEY is not set. Please set it to compute embeddings.")
        self.client = client

    def _text_for_row(self, row):
        cols = ['Project Category','Project Reference','Phase','Problems Encountered','Solutions Adopted']
        return " || ".join([str(row.get(c,'')) for c in cols if row.get(c,'')])

    def embed_texts(self, texts):
        # returns list of embeddings (lists of floats)
        # client.embeddings.create -> response.data[*].embedding
        resp = self.client.embeddings.create(model="text-embedding-3-large", input=texts)
        return [item.embedding for item in resp.data]

    def index_dataframe(self, memory_path, df, id_prefix='mem'):
        """
        memory_path: path to saved dataframe file (e.g., data/memories/memory_1.parquet)
        Creates and saves embeddings JSON as: data/memories/{mem_id}_embeddings.json
        """
        texts = [self._text_for_row(r) for _, r in df.iterrows()]
        embeddings = self.embed_texts(texts)

        mem_path = Path(memory_path)
        mem_id = mem_path.stem  # e.g., memory_1
        out = mem_path.with_suffix('').parent / f"{mem_id}_embeddings.json"
        out.write_text(json.dumps(embeddings))
        return str(out)
