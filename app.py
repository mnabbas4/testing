import streamlit as st
from openai import OpenAI
import xml.etree.ElementTree as ET
from datetime import datetime
import json
import os

# ======================
# 1️⃣ OpenAI API Setup
# ======================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ======================
# 2️⃣ Helper: Clean GPT JSON
# ======================
def clean_gpt_json(raw_text):
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3:
            cleaned = "\n".join(lines[1:-1])
    return cleaned

# ======================
# 3️⃣ NLP → JSON
# ======================
def parse_project_description(user_input):
    prompt = f"""
    You are an expert project planner. 
    Extract the following JSON from this user input:
    {user_input}

    JSON structure:
    {{
        "project_name": "string",
        "start_date": "YYYY-MM-DD",
        "phases": [
            {{"name": "string", "duration_days": int, "tasks": ["task1", "task2"]}}
        ],
        "resources": ["Resource1", "Resource2"]
    }}
    Only return JSON, nothing else.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        text = response.choices[0].message.content
        text = clean_gpt_json(text)

        try:
            return json.loads(text)
        except Exception as e:
            st.error(f"❌ JSON parsing failed: {e}")
            st.text_area("Raw GPT Output", text, height=300)
            return None

    except Exception as e:
        st.error(f"⚠️ OpenAI API Error: {e}")
        return None

# ======================
# 4️⃣ JSON → MS Project XML
# ======================
def generate_ms_project_xml(project_json):
    project = ET.Element("Project")
    ET.SubElement(project, "Name").text = project_json["project_name"]
    ET.SubElement(project, "StartDate").text = project_json.get("start_date", datetime.now().strftime("%Y-%m-%dT08:00:00"))

    tasks_elem = ET.SubElement(project, "Tasks")
    uid_counter = 1
    for phase in project_json["phases"]:
        # Phase task
        phase_task = ET.SubElement(tasks_elem, "Task")
        ET.SubElement(phase_task, "UID").text = str(uid_counter)
        ET.SubElement(phase_task, "ID").text = str(uid_counter)
        ET.SubElement(phase_task, "Name").text = phase["name"]
        ET.SubElement(phase_task, "Duration").text = f"PT{phase['duration_days']*8}H0M0S"
        uid_counter += 1

        # Subtasks
        for task_name in phase.get("tasks", []):
            sub_task = ET.SubElement(tasks_elem, "Task")
            ET.SubElement(sub_task, "UID").text = str(uid_counter)
            ET.SubElement(sub_task, "ID").text = str(uid_counter)
            ET.SubElement(sub_task, "Name").text = task_name
            ET.SubElement(sub_task, "Duration").text = "PT8H0M0S"  # default 1 day
            ET.SubElement(sub_task, "PredecessorLink").text = str(uid_counter - 1)
            uid_counter += 1

    resources_elem = ET.SubElement(project, "Resources")
    for i, res in enumerate(project_json.get("resources", []), start=1):
        resource = ET.SubElement(resources_elem, "Resource")
        ET.SubElement(resource, "UID").text = str(i)
        ET.SubElement(resource, "Name").text = res

    return ET.tostring(project, encoding="utf-8", xml_declaration=True)

# ======================
# 5️⃣ Streamlit UI
# ======================
st.title("🧠 AI Project Planner for MS Project")
st.write("Enter your project description in plain English and get a ready-to-use Microsoft Project XML file.")

user_input = st.text_area(
    "Describe your project",
    "Plan a 3-month e-commerce website project with phases: Planning, Design, Development, Testing. Resources: 5 developers, 1 designer, 1 QA."
)

if st.button("Generate Project XML"):
    if not user_input.strip():
        st.warning("Please enter a project description.")
    else:
        with st.spinner("Processing..."):
            project_json = parse_project_description(user_input)
            if project_json:
                st.success("✅ Project JSON parsed successfully!")
                st.json(project_json)
                xml_bytes = generate_ms_project_xml(project_json)
                st.download_button(
                    label="⬇️ Download Project XML",
                    data=xml_bytes,
                    file_name=f"{project_json['project_name'].replace(' ', '_')}.xml",
                    mime="application/xml"
                )


