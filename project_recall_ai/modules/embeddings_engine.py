# modules/embeddings_engine.py
import os
import numpy as np
from pathlib import Path
from openai import OpenAI
import json




class EmbeddingsEngine:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = None
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        else:
            raise ValueError("OPENAI_API_KEY environment variable not found.")

    def embed_texts(self, texts):
        if not self.client:
            raise ValueError("OpenAI client is not initialized.")
        response = self.client.embeddings.create(
            model="text-embedding-3-large",
            input=texts
        )
        return [d.embedding for d in response.data]

    def _text_for_row(self, row):
        cols = ['Project Category','Project Reference','Phase','Problems Encountered','Solutions Adopted']
        return " || ".join([str(row.get(c,'')) for c in cols if row.get(c,'')])



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


