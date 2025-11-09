# modules/recall_engine.py
import os
import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from textblob import TextBlob
from rapidfuzz import fuzz
from openai import OpenAI

# Initialize OpenAI client (read from env)
_OPENAI_KEY = os.getenv("OPENAI_API_KEY")
_openai_client = OpenAI(api_key=_OPENAI_KEY) if _OPENAI_KEY else None


def _cosine_similarity(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return 0.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class RecallEngine:
    def __init__(self, emb_engine, mem_manager,
                 phase_match_threshold=75,
                 category_match_threshold=75,
                 reference_match_threshold=75):
        self.emb_engine = emb_engine
        self.mem_manager = mem_manager
        self.phase_threshold = phase_match_threshold
        self.category_threshold = category_match_threshold
        self.reference_threshold = reference_match_threshold

    # ------------------------------- Spell correction -------------------------
    def _correct_spelling(self, text):
        try:
            return str(TextBlob(text).correct())
        except Exception:
            return text

    # ------------------------------- Load embeddings --------------------------
    def _load_embeddings(self, mem_id):
        path = Path(self.mem_manager.base) / "memories" / f"{mem_id}_embeddings.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    # ------------------------------- Simple NLP extraction ---------------------
    def _extract_context_hints(self, query, df):
        """Detect likely phase, category, or reference words in query."""
        q = query.lower()
        matched_phase = matched_category = matched_reference = None

        # unique values from df
        phases = [str(x) for x in pd.unique(df["Phase"].dropna()) if str(x).strip()]
        categories = [str(x) for x in pd.unique(df["Project Category"].dropna()) if str(x).strip()]
        references = [str(x) for x in pd.unique(df["Project Reference"].dropna()) if str(x).strip()]

        # regex helpers
        regex_words = re.findall(r"\b[a-zA-Z0-9\-]+\b", q)

        # fuzzy matchers
        def fuzzy_detect(options, threshold):
            best = None
            best_score = 0
            for opt in options:
                for token in regex_words:
                    score = fuzz.partial_ratio(opt.lower(), token)
                    if score > best_score and score >= threshold:
                        best, best_score = opt, score
            return best

        matched_phase = fuzzy_detect(phases, self.phase_threshold)
        matched_category = fuzzy_detect(categories, self.category_threshold)
        matched_reference = fuzzy_detect(references, self.reference_threshold)

        return matched_phase, matched_category, matched_reference

    # ------------------------------- Main query -------------------------------
    def query_memory(self, mem_id, query,
                     min_score=0.25, spell_correction=True,
                     weight_text=0.70, weight_phase=0.15, weight_category=0.10,
                     fallback_k=3):
        """Semantic recall with contextual awareness (phase/category/reference)."""
        q_text = self._correct_spelling(query) if spell_correction else query

        df = self.mem_manager.load_memory_dataframe(mem_id)
        if df is None or df.empty:
            return pd.DataFrame()

        emb_list = self._load_embeddings(mem_id)
        if emb_list is None:
            raise FileNotFoundError(
                f"Embeddings for memory '{mem_id}' not found. Please index first."
            )

        # Detect contextual hints
        matched_phase, matched_category, matched_reference = self._extract_context_hints(q_text, df)

        # Embed query
        q_emb = self.emb_engine.embed_texts([q_text])[0]

        results = []
        for i, row_emb in enumerate(emb_list):
            row = df.iloc[i]
            text_score = _cosine_similarity(q_emb, row_emb)

            # bonus weighting
            phase_bonus = category_bonus = reference_bonus = 0.0

            if matched_phase:
                ph = str(row.get("Phase", "")).lower()
                phase_bonus = fuzz.partial_ratio(ph, matched_phase.lower()) / 100

            if matched_category:
                cat = str(row.get("Project Category", "")).lower()
                category_bonus = fuzz.partial_ratio(cat, matched_category.lower()) / 100

            if matched_reference:
                ref = str(row.get("Project Reference", "")).lower()
                reference_bonus = fuzz.partial_ratio(ref, matched_reference.lower()) / 100

            # weighted score
            final_score = (
                weight_text * text_score
                + weight_phase * phase_bonus
                + weight_category * category_bonus
                + 0.05 * reference_bonus
            )

            results.append({
                "idx": i,
                "FinalScore": round(final_score, 4),
                "TextScore": round(text_score, 4),
                "PhaseBonus": round(phase_bonus, 4),
                "CategoryBonus": round(category_bonus, 4),
                "ReferenceBonus": round(reference_bonus, 4)
            })

        scored_df = pd.DataFrame(results).sort_values("FinalScore", ascending=False)

        # apply dynamic threshold
        matches = scored_df[scored_df["FinalScore"] >= float(min_score)]
        if matches.empty:
            matches = scored_df.head(fallback_k)

        # map results back to main data
        rows = []
        for _, r in matches.iterrows():
            i = int(r["idx"])
            base = df.iloc[i].to_dict()
            rows.append({
                "Score": r["FinalScore"],
                "Project Category": base.get("Project Category", ""),
                "Project Reference": base.get("Project Reference", ""),
                "Phase": base.get("Phase", ""),
                "Problem": base.get("Problems Encountered", ""),
                "Solution": base.get("Solutions Adopted", ""),
            })

        out = pd.DataFrame(rows)
        out.attrs["matched_phase"] = matched_phase
        out.attrs["matched_category"] = matched_category
        out.attrs["matched_reference"] = matched_reference
        return out

    # ------------------------------- Structured insights -----------------------
    def generate_structured_insights(self, matches_df):
        if matches_df is None or matches_df.empty:
            return {"top_problems": [], "per_phase_summary": {}}

        df = matches_df.copy()
        df["Problem_norm"] = df["Problem"].astype(str).str.strip().str.lower()
        grouped = (
            df.groupby("Problem_norm")
            .agg(
                count=("Problem_norm", "size"),
                avg_score=("Score", "mean"),
                solutions=("Solution", lambda s: list(pd.unique(s)))
            )
            .reset_index()
            .sort_values("count", ascending=False)
        )

        top_problems = []
        for _, row in grouped.iterrows():
            top_problems.append({
                "problem": row["Problem_norm"],
                "count": int(row["count"]),
                "avg_score": round(float(row["avg_score"]), 4),
                "solutions": row["solutions"],
            })

        per_phase = {}
        for phase, grp in df.groupby("Phase"):
            per_phase[phase] = {
                "matches": int(len(grp)),
                "top_problems": grp["Problem"].value_counts().head(5).to_dict(),
            }

        return {"top_problems": top_problems, "per_phase_summary": per_phase}

    # ------------------------------- Narrative generation ----------------------
    def generate_insights_narrative(self, structured_insights, max_tokens=400, temperature=0.2):
        if _openai_client is None:
            return "OpenAI client not configured. Set OPENAI_API_KEY."

        facts = []
        if structured_insights.get("top_problems"):
            facts.append("Observed recurring issues with their counts and solutions:")
            for p in structured_insights["top_problems"]:
                sols = "; ".join(p["solutions"][:3]) if p["solutions"] else "No solutions recorded"
                facts.append(f"- {p['problem']} ({p['count']}×) → Solutions: {sols}")

        if structured_insights.get("per_phase_summary"):
            facts.append("Per-phase issue summary:")
            for ph, info in structured_insights["per_phase_summary"].items():
                top = ", ".join(f"{k}:{v}" for k, v in info["top_problems"].items())
                facts.append(f"- {ph}: {top}")

        prompt = (
            "You are a project memory assistant. Summarize these factual findings into practical "
            "lessons to prevent repeating mistakes. Use clear, numbered bullet points.\n\nFacts:\n"
            + "\n".join(facts)
        )

        try:
            resp = _openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"Failed to generate narrative: {e}"
