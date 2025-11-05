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
    """
    Efficient project-structure extraction for planners.
    Produces normalized JSON that always passes ProjectModel validation.
    """
    schema = {
        "project_name": "string",
        "start_date": "YYYY-MM-DD",
        "phases": [
            {
                "name": "string",
                "duration_days": 0,
                "tasks": [
                    {
                        "name": "string",
                        "duration_days": 0,
                        "dependencies": ["string"],
                        "resources": ["string"]
                    }
                ]
            }
        ]
    }

    system_msg = (
        "You are an experienced project planner specializing in MS Project scheduling. "
        "Convert the following description into a CLEAN JSON structure strictly following this schema:\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        "Guidelines:\n"
        "- Use working-day durations (not calendar weeks).\n"
        "- Derive approximate durations if not stated (e.g., small tasks → 2 days).\n"
        "- Convert milestones (no work) to duration_days = 0.\n"
        "- Include dependencies logically (e.g., Design → Implementation).\n"
        "- Avoid extra fields, text, or commentary."
    )

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
    data = json.loads(raw)

    # --- Normalization: fix typical GPT variations ---
    if "project" in data:
        data = data["project"]
    if "name" in data and "project_name" not in data:
        data["project_name"] = data.pop("name")
    data.setdefault("start_date", datetime.now().strftime("%Y-%m-%d"))

    # Normalize durations (if strings like "3 weeks")
    for phase in data.get("phases", []):
        if isinstance(phase.get("duration_days"), str):
            if "week" in phase["duration_days"].lower():
                n = int("".join([c for c in phase["duration_days"] if c.isdigit()]) or "1")
                phase["duration_days"] = n * 5
            elif "day" in phase["duration_days"].lower():
                n = int("".join([c for c in phase["duration_days"] if c.isdigit()]) or "1")
                phase["duration_days"] = n
        for task in phase.get("tasks", []):
            if isinstance(task.get("duration_days"), str):
                if "week" in task["duration_days"].lower():
                    n = int("".join([c for c in task["duration_days"] if c.isdigit()]) or "1")
                    task["duration_days"] = n * 5
                elif "day" in task["duration_days"].lower():
                    n = int("".join([c for c in task["duration_days"] if c.isdigit()]) or "1")
                    task["duration_days"] = n
            task.setdefault("dependencies", [])
            task.setdefault("resources", [])

    return data


user_input = st.text_area("Describe your project:", height=300)
def validate_plan(project: ProjectModel):
    issues = []
    for phase in project.phases:
        if phase.duration_days == 0:
            issues.append(f"Phase '{phase.name}' has zero duration.")
        for t in phase.tasks:
            if t.duration_days == 0 and not t.dependencies:
                issues.append(f"Task '{t.name}' may be missing predecessors or milestone tag.")
    return issues

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



