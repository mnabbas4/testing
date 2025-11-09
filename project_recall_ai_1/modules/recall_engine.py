# modules/recall_engine.py
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from textblob import TextBlob
from rapidfuzz import fuzz
from openai import OpenAI
OPENAI_API_KEY = "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"

# Initialize OpenAI client if key exists
_OPENAI_KEY = os.getenv("OPENAI_API_KEY")
_openai_client = OpenAI(api_key=_OPENAI_KEY) if _OPENAI_KEY else None


def _cosine_similarity(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return 0.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class RecallEngine:
    def __init__(self, emb_engine, mem_manager, phase_match_threshold=75, category_match_threshold=75):
        """
        emb_engine: object with embed_texts([text]) -> [embedding]
        mem_manager: MemoryManager instance exposing load_memory_dataframe(mem_id) and base path
        """
        self.emb_engine = emb_engine
        self.mem_manager = mem_manager
        self.phase_threshold = phase_match_threshold
        self.category_threshold = category_match_threshold

    def _correct_spelling(self, text):
        try:
            return str(TextBlob(text).correct())
        except Exception:
            return text

    def _load_embeddings(self, mem_id):
        path = Path(self.mem_manager.base) / 'memories' / f"{mem_id}_embeddings.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def _extract_context_hints(self, query, df):
        """
        Try to detect phase/category mentions in query using fuzzy matching.
        Returns (matched_phase or None, matched_category or None)
        """
        q = query.lower()
        matched_phase = None
        matched_category = None

        # unique values from df
        phases = [str(x) for x in pd.unique(df['Phase'].dropna()) if str(x).strip() != ""]
        categories = [str(x) for x in pd.unique(df['Project Category'].dropna()) if str(x).strip() != ""]

        # check phases
        for p in phases:
            score = fuzz.partial_ratio(p.lower(), q)
            if score >= self.phase_threshold:
                matched_phase = p
                break

        # check categories
        for c in categories:
            score = fuzz.partial_ratio(c.lower(), q)
            if score >= self.category_threshold:
                matched_category = c
                break

        return matched_phase, matched_category

    def query_memory(self, mem_id, query, min_score=0.25, spell_correction=True,
                     hard_limit=None, enforce_context=False, weight_text=0.70,
                     weight_phase=0.15, weight_category=0.15, fallback_k=3):
        """
        Core querying method.

        - If spell_correction: corrects query text.
        - If enforce_context=True and a phase/category hint is found, the candidate set
          will be pre-filtered to that phase/category (strict).
        - Scores combine text similarity with phase/category bonuses.
        - Returns pandas.DataFrame of matches sorted by final_score desc.
        - If no match >= min_score, returns top `fallback_k` nearest neighbors (with a warning).
        - If hard_limit (int) > 0 provided, results are capped after filtering and sorting.
        """
        if spell_correction:
            q_text = self._correct_spelling(query)
        else:
            q_text = query

        df = self.mem_manager.load_memory_dataframe(mem_id)
        if df is None or df.empty:
            return pd.DataFrame(columns=[
                'FinalScore', 'TextScore', 'PhaseBonus', 'CategoryBonus',
                'Project Category','Project Reference','Phase','Problems Encountered','Solutions Adopted'
            ])

        emb_list = self._load_embeddings(mem_id)
        if emb_list is None:
            raise FileNotFoundError(f"Embeddings for memory '{mem_id}' not found. Please index first.")

        # detect hints (phase/category)
        matched_phase, matched_category = self._extract_context_hints(q_text, df)

        # Optionally filter strictly by detected context
        candidate_idxs = list(range(len(emb_list)))
        if enforce_context:
            if matched_phase:
                candidate_idxs = [i for i in candidate_idxs if str(df.iloc[i].get('Phase','')).strip().lower() == matched_phase.lower()]
            if matched_category:
                candidate_idxs = [i for i in candidate_idxs if str(df.iloc[i].get('Project Category','')).strip().lower() == matched_category.lower()]

            if len(candidate_idxs) == 0:
                # if strict filtering removes everything, fall back to all candidates
                candidate_idxs = list(range(len(emb_list)))

        # embed the query
        q_emb = self.emb_engine.embed_texts([q_text])[0]

        scored = []
        for idx in candidate_idxs:
            row_emb = emb_list[idx]
            text_score = _cosine_similarity(q_emb, row_emb)
            # phase bonus (fuzzy)
            phase_bonus = 0.0
            if matched_phase:
                p = str(df.iloc[idx].get('Phase','')).lower()
                ph_score = fuzz.partial_ratio(p, matched_phase.lower()) if p else 0
                if ph_score >= 85:
                    phase_bonus = 1.0
                elif ph_score >= 65:
                    phase_bonus = 0.5
            # category bonus (fuzzy)
            category_bonus = 0.0
            if matched_category:
                c = str(df.iloc[idx].get('Project Category','')).lower()
                cat_score = fuzz.partial_ratio(c, matched_category.lower()) if c else 0
                if cat_score >= 85:
                    category_bonus = 1.0
                elif cat_score >= 65:
                    category_bonus = 0.5

            final_score = (weight_text * text_score) + (weight_phase * phase_bonus) + (weight_category * category_bonus)
            scored.append({
                'idx': idx,
                'TextScore': round(text_score, 4),
                'PhaseBonus': round(phase_bonus, 4),
                'CategoryBonus': round(category_bonus, 4),
                'FinalScore': round(final_score, 4)
            })

        scored_df = pd.DataFrame(scored).sort_values('FinalScore', ascending=False)

        # filter by threshold
        matches = scored_df[scored_df['FinalScore'] >= float(min_score)].copy()

        # If none pass threshold, fallback to top-k nearest neighbors (by TextScore)
        used_fallback = False
        if matches.empty:
            used_fallback = True
            # pick top fallback_k by TextScore across all candidates
            fallback_df = pd.DataFrame(scored).sort_values('TextScore', ascending=False).head(fallback_k)
            matches = fallback_df

        # Apply hard cap if set (hard_limit > 0)
        if hard_limit is not None and isinstance(hard_limit, int) and hard_limit > 0:
            matches = matches.head(hard_limit)

        # map back to original rows and build result frame
        out_rows = []
        for _, r in matches.iterrows():
            i = int(r['idx'])
            row = df.iloc[i].to_dict()
            out_rows.append({
                'FinalScore': round(float(r['FinalScore']), 4),
                'TextScore': r['TextScore'],
                'PhaseBonus': r['PhaseBonus'],
                'CategoryBonus': r['CategoryBonus'],
                'Project Category': row.get('Project Category',''),
                'Project Reference': row.get('Project Reference',''),
                'Phase': row.get('Phase',''),
                'Problems Encountered': row.get('Problems Encountered',''),
                'Solutions Adopted': row.get('Solutions Adopted',''),
            })

        out_df = pd.DataFrame(out_rows)
        # attach metadata about whether fallback was used & detected hints
        out_df.attrs['used_fallback'] = used_fallback
        out_df.attrs['matched_phase'] = matched_phase
        out_df.attrs['matched_category'] = matched_category
        return out_df

    def generate_structured_insights(self, matches_df):
        """
        Deterministic, evidence-first aggregation.
        Produces:
          - top_problems: list of dicts {problem, count, avg_score, solutions}
          - per_phase_summary: {phase: {'matches':n, 'top_problems':{problem:count}}}
        """
        if matches_df is None or matches_df.empty:
            return {'top_problems': [], 'per_phase_summary': {}}

        df = matches_df.copy()
        # normalize problem text
        df['Problem_norm'] = df['Problems Encountered'].astype(str).str.strip().str.lower()
        grouped = df.groupby('Problem_norm').agg(
            count=('Problem_norm','size'),
            avg_score=('FinalScore','mean'),
            solutions=('Solutions Adopted', lambda s: list(pd.unique(s)))
        ).reset_index().sort_values('count', ascending=False)

        top_problems = []
        for _, row in grouped.iterrows():
            top_problems.append({
                'problem': row['Problem_norm'],
                'count': int(row['count']),
                'avg_score': round(float(row['avg_score']), 4),
                'solutions': row['solutions']
            })

        per_phase = {}
        for phase, grp in df.groupby('Phase'):
            per_phase[phase] = {
                'matches': int(len(grp)),
                'top_problems': grp['Problems Encountered'].value_counts().head(5).to_dict()
            }

        return {'top_problems': top_problems, 'per_phase_summary': per_phase}

    def generate_insights_narrative(self, structured_insights, max_tokens=400, temperature=0.2):
        """
        Uses OpenAI to convert structured facts into a short, strictly-fact-based narrative.
        Only calls OpenAI if client is available.
        """
        if _openai_client is None:
            return "OpenAI client is not configured. Set OPENAI_API_KEY to enable narrative generation."

        facts = []
        if structured_insights.get('top_problems'):
            facts.append("Top problems (problem; count; avg_score; example solutions):")
            for p in structured_insights['top_problems']:
                example_sols = "; ".join(p['solutions'][:3]) if p['solutions'] else "No solutions recorded"
                facts.append(f"- {p['problem']} ; {p['count']} occurrences ; avg_score={p['avg_score']} ; solutions: {example_sols}")

        if structured_insights.get('per_phase_summary'):
            facts.append("Per-phase summary (phase: matches -> top problems):")
            for ph, info in structured_insights['per_phase_summary'].items():
                top = ", ".join(f"{k}:{v}" for k,v in info['top_problems'].items())
                facts.append(f"- {ph}: {info['matches']} matches -> {top}")

        prompt = (
            "You are an assistant that only produces a concise narrative strictly based on the facts provided. "
            "Do NOT add any facts or speculation. Write 3-6 short actionable bullet points for a project team to avoid repeating these issues.\n\n"
            "Facts:\n" + "\n".join(facts)
        )

        resp = _openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        try:
            return resp.choices[0].message.content
        except Exception:
            return "Failed to generate narrative from OpenAI."
