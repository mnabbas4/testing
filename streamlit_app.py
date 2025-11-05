import streamlit as st
import requests
import json
import openai
from pydantic import ValidationError
from datetime import datetime
from ai_project_planner_strict import extract_project_structure, ProjectModel

# ---------------------
# Configuration
# ---------------------

openai.api_key = OPENAI_API_KEY
BACKEND_URL = "https://bea8a2b73844.ngrok-free.app"  # replace this after starting backend

st.set_page_config(page_title="AI Project Planner", layout="wide")
st.title("🤖 AI Project Planner (Remote Mode)")
st.caption("Front-end only: JSON extraction via GPT + sends to backend for Aspose XML generation.")

# ---------------------
# Input Area
# ---------------------
user_input = st.text_area(
    "Describe your project in natural language:",
    height=380,
    placeholder="e.g. Build a mobile app with phases: Design, Development, Testing..."
)

start_override = st.date_input("Override start date (optional)", value=None)

# ---------------------
# Process Button
# ---------------------
if st.button("Generate Project XML"):
    if not user_input.strip():
        st.error("❌ Please enter a project description.")
        st.stop()

    with st.spinner("Extracting structured project plan via GPT..."):
        try:
            raw_json = extract_project_structure(user_input)
        except Exception as e:
            st.error(f"OpenAI extraction failed: {e}")
            st.stop()

    if start_override:
        raw_json["start_date"] = start_override.strftime("%Y-%m-%d")

    try:
        project_model = ProjectModel(**raw_json)
    except ValidationError as ve:
        st.error("❌ JSON validation failed.")
        st.json(json.loads(ve.json()))
        st.stop()

    st.success("✅ JSON validated successfully.")
    st.subheader("Structured Project Data")
    st.json(project_model.model_dump())

    # Send JSON to backend
    with st.spinner("Sending project to backend for Aspose XML generation..."):
        try:
            response = requests.post(
                f"{BACKEND_URL}/generate",
                json=project_model.model_dump(),
                timeout=120
            )

            if response.status_code == 200:
                st.success("✅ Project XML generated successfully!")
                st.download_button(
                    "⬇️ Download MS Project XML",
                    response.content,
                    file_name=f"{project_model.project_name}.xml",
                    mime="application/xml"
                )
            else:
                st.error(f"❌ Backend returned error: {response.text}")

        except Exception as e:
            st.error(f"Failed to reach backend: {e}")


