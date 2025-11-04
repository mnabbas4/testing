# frontend_streamlit.py
import streamlit as st
import openai
import json
import requests
from datetime import datetime
from pydantic import ValidationError
from ai_project_planner_strict import extract_project_structure, ProjectModel

st.set_page_config(page_title="AI Project Planner (Frontend)", layout="wide")
st.title("🌐 Remote AI Project Planner (Frontend)")
st.caption("Uses remote Aspose backend over ngrok to generate MS Project XML")

# 👉 Update this URL with your ngrok public URL
BACKEND_URL = "https://e0b045081384.ngrok-free.app"

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    st.warning("⚠️ Missing OpenAI key. Set it in Streamlit Secrets or environment.")
openai.api_key = OPENAI_API_KEY
OPENAI_MODEL = "gpt-4o-mini"

user_input = st.text_area(
    "Describe your project plan in natural language:",
    height=380,
    placeholder="e.g. Develop AI project planner with research, coding, testing phases..."
)
start_override = st.date_input("Optional Start Date")

if st.button("Generate MS Project XML"):
    if not user_input.strip():
        st.error("Please enter a description.")
        st.stop()

    with st.spinner("Extracting structured project plan..."):
        try:
            raw_json = extract_project_structure(user_input)
        except Exception as e:
            st.error(f"❌ Failed to extract JSON: {e}")
            st.stop()

    # Apply override
    try:
        if start_override:
            raw_json["start_date"] = start_override.strftime("%Y-%m-%d")
        project_model = ProjectModel(**raw_json)
    except ValidationError as ve:
        st.error("❌ JSON validation failed.")
        st.json(json.loads(ve.json()))
        st.stop()

    st.success("✅ JSON validated successfully!")
    st.json(project_model.model_dump())

    filename = f"{project_model.project_name.replace(' ', '_')}.xml"

    with st.spinner("Sending to remote Aspose backend..."):
        try:
            payload = json.dumps(project_model.model_dump())
            resp = requests.post(
                f"{BACKEND_URL}/generate",
                data={"json_data": payload, "filename": filename},
                timeout=180
            )
        except Exception as e:
            st.error(f"❌ Network error: {e}")
            st.stop()

        if not resp.ok:
            st.error(f"❌ Backend returned {resp.status_code}: {resp.text}")
            st.stop()

        result = resp.json()
        if result.get("status") != "ok":
            st.error(f"❌ Backend error: {result}")
            st.stop()

        download_url = f"{BACKEND_URL}{result['download_path']}"
        st.success("✅ File successfully generated!")
        st.markdown(f"[⬇️ Download XML file]({download_url})")
