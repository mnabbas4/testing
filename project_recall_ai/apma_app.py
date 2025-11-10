# apma_app.py
import sys, os
import streamlit as st
from modules.utils import ensure_data_dirs
import pandas as pd


# Add this directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "modules"))

from modules.embeddings_engine import EmbeddingsEngine

from modules.recall_engine import RecallEngine
from modules.file_manager import MemoryManager

from modules.data_handler import DataHandler
import os, streamlit as st

# Ensure secrets propagate to environment variables for submodules
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]



st.set_page_config(page_title="APMA — AI Project Memory Assistant", layout="wide")
ensure_data_dirs()

st.title("AI Project Memory Assistant (APMA)")

# Sidebar
mode = st.sidebar.selectbox("Mode", ["Upload / Update Memory File", "Query Knowledge Base", "Settings"])

mem_manager = MemoryManager(data_dir='data')

# Embeddings engine
emb_engine = None
try:
    emb_engine = EmbeddingsEngine()
except Exception as e:
    emb_engine = None
    emb_msg = str(e)

recall_engine = RecallEngine(emb_engine=emb_engine, mem_manager=mem_manager)

# initialize session state keys
if 'last_results' not in st.session_state:
    st.session_state['last_results'] = None
if 'last_query' not in st.session_state:
    st.session_state['last_query'] = None
if 'last_mem_id' not in st.session_state:
    st.session_state['last_mem_id'] = None
if 'last_insights' not in st.session_state:
    st.session_state['last_insights'] = None
if 'last_narrative' not in st.session_state:
    st.session_state['last_narrative'] = None

if mode == "Upload / Update Memory File":
    st.header("Upload / Update Memory File")
    uploaded = st.file_uploader("Upload CSV or Excel", type=['csv','xlsx'])
    if uploaded is not None:
        df, err = DataHandler.read_and_validate(uploaded)
        if err:
            st.error(err)
        else:
            st.success(f"Validated: {len(df)} rows")
            st.dataframe(df.head(10))
            existing = mem_manager.list_memories()
            choice = st.radio("Save options:", ["Create new memory file", "Append to existing memory", "Edit existing memory"])
            sel = None
            if choice != "Create new memory file":
                sel = st.selectbox("Select memory to modify", ["-- choose --"] + existing)
            mem_name = st.text_input("Memory file name", value=f"memory_{len(existing)+1}")
            if st.button("Save to memory"):
                saved_path = mem_manager.save_upload(uploaded)
                meta = mem_manager.create_or_update_memory(mem_name, df, mode=choice, target_memory=sel)
                st.success(f"Saved dataframe to memory file: {meta['memory_id']}")
                if emb_engine is not None:
                    with st.spinner("Computing embeddings..."):
                        emb_engine.index_dataframe(meta['memory_path'], df, id_prefix=meta['memory_id'])
                    st.success("Embeddings computed and saved.")
                else:
                    st.warning("Embeddings engine not active. Set OPENAI_API_KEY to compute embeddings.")

elif mode == "Query Knowledge Base":
    st.header("Query Knowledge Base")
    memories = mem_manager.list_memories()
    if not memories:
        st.warning("No memories found. Upload a file first.")
    else:
        sel_mem = st.selectbox("Select memory to query", memories)
        st.markdown("---")
        st.write("Result controls")
        top_n = st.slider("Hard cap (set 0 for no hard cap):", 0, 200, 0)  # default 0 = no cap
        similarity_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.25)
        spell = st.checkbox("Enable spell correction", value=True)
        enforce_context = st.checkbox("Enforce detected phase/category (strict filter)", value=False)
        q = st.text_area("Enter your query (natural language)", height=120)

        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Search") and q.strip():
                try:
                    hard_limit = top_n if top_n > 0 else None
                    res_df = recall_engine.query_memory(
                        mem_id=sel_mem,
                        query=q,
                        min_score=similarity_threshold,
                        spell_correction=spell,
                        hard_limit=hard_limit,
                        enforce_context=enforce_context
                    )
                    st.session_state['last_results'] = res_df
                    st.session_state['last_query'] = q
                    st.session_state['last_mem_id'] = sel_mem
                    st.session_state['last_insights'] = None
                    st.session_state['last_narrative'] = None
                except Exception as e:
                    st.error(f"Query failed: {e}")

            if st.button("Reset Query"):
                st.session_state['last_results'] = None
                st.session_state['last_query'] = None
                st.session_state['last_insights'] = None
                st.session_state['last_narrative'] = None

        with col2:
            st.markdown("**Tips:** include phase names (e.g., 'Order Confirmation') or categories to improve precision. Use 'Enforce detected phase/category' to restrict search to exact phase/category matches if needed.")

        st.markdown("---")

        # display last results from session state (persistent)
        if st.session_state['last_results'] is None:
            st.info("No previous results. Run a search to see matches.")
        else:
            results_df = st.session_state['last_results']
            used_fallback = results_df.attrs.get('used_fallback', False)
            matched_phase = results_df.attrs.get('matched_phase', None)
            matched_category = results_df.attrs.get('matched_category', None)

            # header context
            ctx_lines = []
            if matched_phase:
                ctx_lines.append(f"Detected phase hint: **{matched_phase}**")
            if matched_category:
                ctx_lines.append(f"Detected category hint: **{matched_category}**")
            if used_fallback:
                ctx_lines.append("No matches passed the similarity threshold — showing nearest neighbors as fallback.")
            if ctx_lines:
                st.markdown("**Context:** " + " — ".join(ctx_lines))

            # show results table
            display_cols = ['FinalScore','TextScore','PhaseBonus','CategoryBonus','Project Category','Project Reference','Phase','Problems Encountered','Solutions Adopted']
            st.write("### Results")
            st.dataframe(results_df[display_cols].reset_index(drop=True), use_container_width=True)

            # Structured insights (data-first)
            st.markdown("### Structured Insights (data-driven)")
            insights = recall_engine.generate_structured_insights(results_df)
            st.session_state['last_insights'] = insights
            if insights['top_problems']:
                top_table = pd.DataFrame([{
                    'Problem': p['problem'],
                    'Count': p['count'],
                    'AvgScore': p['avg_score'],
                    'ExampleSolutions': " | ".join(p['solutions'][:3])
                } for p in insights['top_problems']])
                st.write("Top problems (aggregated):")
                st.dataframe(top_table, use_container_width=True)
            else:
                st.write("No aggregated problems to show.")

            if insights['per_phase_summary']:
                st.write("Per-phase summary:")
                st.json(insights['per_phase_summary'])

            # LLM narrative generation (uses stored insights, won't refresh the page)
            if st.button("Generate human-readable insights (LLM)"):
                if OPENAI_API_KEY is None:
                    st.error("OpenAI key not configured. Set OPENAI_API_KEY to enable narrative generation.")
                else:
                    with st.spinner("Generating narrative from facts..."):
                        narrative = recall_engine.generate_insights_narrative(insights)
                        st.session_state['last_narrative'] = narrative
            if st.session_state['last_narrative']:
                st.markdown("**Narrative (based strictly on recorded facts):**")
                st.write(st.session_state['last_narrative'])

else:  # Settings
    st.header("Settings")
    st.write("Saved memories:")
    for m in mem_manager.list_memories():
        st.write(f"- {m}")
    if emb_engine is None:
        st.warning("Embeddings engine not active. Set OPENAI_API_KEY to compute embeddings.")
    if st.button("Rebuild embeddings for all memories"):
        if emb_engine is None:
            st.error("Embeddings engine not active.")
        else:
            with st.spinner("Rebuilding embeddings..."):
                mems = mem_manager.list_memories_full()
                for mid, path in mems.items():
                    df = mem_manager.load_memory_dataframe(mid)
                    emb_engine.index_dataframe(path, df, id_prefix=mid)
            st.success("Rebuilt embeddings for all memories.")



