import streamlit as st
import openai
import requests, json
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Optional

st.set_page_config(page_title="AI Project Planner (Frontend)", layout="wide")
st.title("🧠 AI Project Planner → Remote Aspose XML")

BACKEND_URL = st.secrets["BACKEND_URL"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
openai.api_key = OPENAI_API_KEY
MODEL = "gpt-4o-mini"

class TaskModel(BaseModel):
    name: str
    duration_days: int = Field(..., ge=0)
    dependencies: List[str] = []
    resources: List[str] = []

class PhaseModel(BaseModel):
    name: str
    duration_days: int = Field(..., ge=0)
    tasks: List[TaskModel]

class ProjectModel(BaseModel):
    project_name: str
    start_date: str
    phases: List[PhaseModel]

def extract_project(raw_text: str):
    system_msg = "Convert plain English project description into structured JSON."
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": raw_text}
        ],
        temperature=0
    )
    raw = response["choices"][0]["message"]["content"].strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.splitlines()[1:-1])
    return json.loads(raw)

user_input = st.text_area("Describe your project:", height=300)

if st.button("Generate XML"):
    if not user_input.strip():
        st.warning("Please enter a project description.")
        st.stop()

    with st.spinner("Extracting structure via OpenAI..."):
        try:
            structured = extract_project(user_input)
            project = ProjectModel(**structured)
        except ValidationError as ve:
            st.error("JSON validation failed.")
            st.json(json.loads(ve.json()))
            st.stop()
        except Exception as e:
            st.error(f"OpenAI parsing failed: {e}")
            st.stop()

    st.success("✅ Valid project structure")
    st.json(project.model_dump())

    with st.spinner("Sending to Aspose backend... please wait..."):
        payload = {
            "json_data": json.dumps(project.model_dump()),
            "filename": f"{project.project_name.replace(' ', '_')}.xml"
        }
        try:
            resp = requests.post(f"{BACKEND_URL.rstrip('/')}/generate", data=payload)
        except Exception as e:
            st.error(f"Failed to reach backend: {e}")
            st.stop()

        if not resp.ok:
            st.error(f"Server error: {resp.text}")
            st.stop()

        result = resp.json()
        download_url = f"{BACKEND_URL.rstrip('/')}{result['download_path']}"
        st.success("✅ File generated remotely!")
        st.markdown(f"[⬇️ Download XML File]({download_url})")

