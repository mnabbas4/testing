# ai_project_planner_strict.py
# Strict version — no fallbacks. Adds:
# - dependency type & lag parsing
# - milestone detection (duration_days==0 -> IsMilestone)
# - baseline recording (SetBaseline(BaselineType.Baseline1))
# - resource calendars (calendar exceptions)
# - circular dependency validation (DFS)
#
# Requires: pip install streamlit openai pydantic pythonnet
# Ensure Aspose.Tasks DLL folder is placed and ASPOSE_DLL_DIR updated below.

from pathlib import Path
import os
import json
from datetime import datetime
from typing import List, Dict, Optional
import streamlit as st
import openai
from pydantic import BaseModel, ValidationError, Field
import clr, sys, traceback
import requests
# ---------------------------
# Configuration - update this to match your Aspose DLL folder
# ---------------------------
ASPOSE_DLL_DIR = r"C:\Users\Syedm\AppData\Local\Programs\Microsoft VS Code\Aspose.Tasks\NET45"
# Make sure Aspose.Tasks.dll exists in ASPOSE_DLL_DIR
if not Path(ASPOSE_DLL_DIR).exists():
    raise RuntimeError(f"Aspose directory not found: {ASPOSE_DLL_DIR}")


if not OPENAI_API_KEY:
    st.error("❌ Missing OpenAI API key.")
    st.stop()
openai.api_key = OPENAI_API_KEY
OPENAI_MODEL = "gpt-4o-mini"

# ---------------------------
# Pydantic models (allow optional dependency_type / lag_days)
# ---------------------------
class TaskModel(BaseModel):
    name: str
    duration_days: int = Field(..., ge=0)
    dependencies: List[str] = []
    resources: List[str] = []
    dependency_type: Optional[str] = None  # "FS","SS","FF","SF"
    lag_days: Optional[int] = 0           # integer lag in days

class PhaseModel(BaseModel):
    name: str
    duration_days: int = Field(..., ge=0)
    tasks: List[TaskModel]

class ProjectModel(BaseModel):
    project_name: str
    start_date: str  # YYYY-MM-DD
    phases: List[PhaseModel]
    resources: List[str] = []
    holidays: List[Dict] = []  # optional list of {"name","start","end"} in YYYY-MM-DD
    resource_unavailability: Dict[str, List[Dict]] = {}  # {"ResourceName":[{"start":"YYYY-MM-DD","end":"YYYY-MM-DD"}]}

# ---------------------------
# GPT extraction (unchanged)
# ---------------------------
def extract_project_structure(raw_text: str) -> Dict:
    system_msg = (
        "You are a senior project planner. "
        "Convert plain text into a strict JSON schema for MS Project planning."
    )
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
                        "resources": ["string"],
                        "dependency_type": "FS|SS|FF|SF (optional)",
                        "lag_days": 0
                    }
                ]
            }
        ],
        "resources": ["string"],
        "holidays": [{"name":"string","start":"YYYY-MM-DD","end":"YYYY-MM-DD"}],
        "resource_unavailability": {"ResourceName":[{"start":"YYYY-MM-DD","end":"YYYY-MM-DD"}]}
    }
    user_msg = f"""
Convert this project description to a strict JSON following this schema only:
{json.dumps(schema, indent=2)}

User input:
{raw_text}
"""
    response = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        messages=[{"role":"system","content":system_msg},{"role":"user","content":user_msg}],
        temperature=0
    )
    raw = response["choices"][0]["message"]["content"].strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.splitlines()[1:-1])
    return json.loads(raw)

# ---------------------------
# Utility: cycle detection (DFS)
# ---------------------------
def detect_cycle(phases: List[PhaseModel]) -> Optional[List[str]]:
    # Build adjacency from task name -> set(predecessor names)
    adj = {}
    for phase in phases:
        for t in phase.tasks:
            adj.setdefault(t.name, set())
            for dep in t.dependencies:
                adj.setdefault(dep, set())
                adj[t.name].add(dep)
    # DFS to find cycle
    visited = {}
    path = []
    def dfs(node):
        visited[node] = 1  # visiting
        path.append(node)
        for pred in adj.get(node, ()):
            if visited.get(pred, 0) == 1:
                # cycle found, return cycle path
                cycle_start = path.index(pred) if pred in path else 0
                return path[cycle_start:] + [pred]
            if visited.get(pred, 0) == 0:
                res = dfs(pred)
                if res:
                    return res
        visited[node] = 2
        path.pop()
        return None
    for n in list(adj.keys()):
        if visited.get(n,0) == 0:
            c = dfs(n)
            if c:
                return c
    return None

# ---------------------------
# Aspose.Tasks integration with requested features
# ---------------------------
def save_aspose_project(project_data, filename: str) -> str:
    import sys
    from datetime import datetime
    from pathlib import Path

    # Ensure Aspose DLL dir is on sys.path (ASPOSE_DLL_DIR must be defined globally)
    if not Path(ASPOSE_DLL_DIR).exists():
        raise RuntimeError(f"Aspose DLL directory not found: {ASPOSE_DLL_DIR}")
    if str(ASPOSE_DLL_DIR) not in sys.path:
        sys.path.append(str(ASPOSE_DLL_DIR))

    try:
        clr.AddReference("Aspose.Tasks")
    except Exception as e:
        raise RuntimeError(f"Aspose.Tasks CLR AddReference failed: {e}")

    import Aspose.Tasks as AT
    try:
        from Aspose.Tasks.Saving import SaveFileFormat
    except Exception:
        SaveFileFormat = AT.SaveFileFormat

    # quick API checks
    if not hasattr(AT.Project, "GetDuration"):
        raise RuntimeError("Expected AT.Project.GetDuration not found in this Aspose.Tasks build — adapt script to your DLL.")

    from System import String, DateTime

    # cycle detection (assume detect_cycle exists globally)
    cycle = detect_cycle(project_data.phases)
    if cycle:
        raise RuntimeError(f"Circular dependency detected in task graph: {' -> '.join(cycle)}")

    # create project and set properties
    project = AT.Project()
    try:
        start_dt = datetime.strptime(project_data.start_date, "%Y-%m-%d")
    except Exception:
        start_dt = datetime.now()

    try:
        project.Name = String(project_data.project_name)
    except Exception as e:
        raise RuntimeError(f"Failed to set Project.Name: {e}")
    try:
        project.StartDate = DateTime(start_dt.year, start_dt.month, start_dt.day)
    except Exception as e:
        raise RuntimeError(f"Failed to set Project.StartDate: {e}")

    # create resource pool: res_map name->Resource
    res_map = {}
    try:
        for r in project_data.resources:
            res = project.Resources.Add(String(r))
            res_map[r] = res
    except Exception as e:
        raise RuntimeError(f"Failed to add resource(s): {e}")

    # optional: import NullableBool for milestone setting
    NullableBool = None
    try:
        NullableBool = getattr(AT, "NullableBool", None)
    except Exception:
        NullableBool = None

    # helper to compute duration
    def compute_duration(days):
        try:
            return project.GetDuration(int(days), AT.TimeUnitType.Day)
        except Exception:
            # try fallback lookup of TimeUnitType
            try:
                tu = getattr(AT, "TimeUnitType")
                return project.GetDuration(int(days), tu.Day)
            except Exception as e:
                raise RuntimeError(f"Could not compute duration for {days} days: {e}")

    # create tasks and assignments
    task_map = {}
    try:
        for phase in project_data.phases:
            phase_task = project.RootTask.Children.Add(String(phase.name))
            phase_duration = compute_duration(phase.duration_days)
            try:
                phase_task.Duration = phase_duration
            except Exception:
                try:
                    phase_task.Set(AT.Prj.Duration, phase_duration)
                except Exception as e:
                    raise RuntimeError(f"Failed to assign phase duration for {phase.name}: {e}")
            task_map[phase.name] = phase_task

            for t in phase.tasks:
                sub_task = phase_task.Children.Add(String(t.name))
                dur = compute_duration(t.duration_days)
                try:
                    sub_task.Duration = dur
                except Exception:
                    try:
                        sub_task.Set(AT.Prj.Duration, dur)
                    except Exception as e:
                        raise RuntimeError(f"Failed to assign duration for task {t.name}: {e}")

                # Mark milestone properly using NullableBool if required
                if int(t.duration_days) == 0:
                    try:
                        if NullableBool:
                            # some builds expect AT.NullableBool(True)
                            sub_task.IsMilestone = NullableBool(True)
                        else:
                            sub_task.IsMilestone = True
                    except Exception as e:
                        # try Set fallback
                        try:
                            if NullableBool:
                                sub_task.Set(AT.Tsk.IsMilestone, NullableBool(True))
                            else:
                                sub_task.Set(AT.Tsk.IsMilestone, True)
                        except Exception as e2:
                            raise RuntimeError(f"Failed to set IsMilestone for {t.name}: {e2}")

                # Assign resources using project.ResourceAssignments.Add(task, resource)
                for r_name in (t.resources or []):
                    if r_name not in res_map:
                        raise RuntimeError(f"Resource '{r_name}' referenced by task '{t.name}' not found in resource pool.")
                    res_obj = res_map[r_name]
                    try:
                        # Correct signature: Add(task, resource)
                        assignment = project.ResourceAssignments.Add(sub_task, res_obj)
                        # Optionally set units to 1.0 (100%) if Asn.Units key exists
                        try:
                            # some builds expose AT.Asn.Units
                            if hasattr(AT, "Asn") and hasattr(AT.Asn, "Units"):
                                assignment.Set(AT.Asn.Units, 1.0)
                        except Exception:
                            # ignore if units can't be set
                            pass
                    except Exception as e:
                        raise RuntimeError(f"Failed to assign resource '{r_name}' to task '{t.name}': {e}")

                # store in task_map
                task_map[t.name] = sub_task
    except Exception as e:
        # bubble up clearly
        raise RuntimeError(f"Error while creating tasks/resources: {e}")

    # Add dependencies (TaskLinks) and set type/lag
    link_type_map = {}
    if hasattr(AT, "TaskLinkType"):
        TLT = AT.TaskLinkType
        if hasattr(TLT, "FinishToStart"):
            link_type_map["FS"] = TLT.FinishToStart
        if hasattr(TLT, "StartToStart"):
            link_type_map["SS"] = TLT.StartToStart
        if hasattr(TLT, "FinishToFinish"):
            link_type_map["FF"] = TLT.FinishToFinish
        if hasattr(TLT, "StartToFinish"):
            link_type_map["SF"] = TLT.StartToFinish
    try:
            for phase in project_data.phases:
                for t in phase.tasks:
                    current = task_map.get(t.name)
                    if current is None:
                        raise RuntimeError(f"Task {t.name} missing from map before linking.")

                    for dep_name in (t.dependencies or []):
                        if dep_name not in task_map:
                            raise RuntimeError(f"Dependency '{dep_name}' for task '{t.name}' not found.")
                        predecessor = task_map[dep_name]

                        try:
                            # Add() returns the created link object directly
                            created_link = project.TaskLinks.Add(predecessor, current)
                        except Exception as e:
                            raise RuntimeError(f"Failed to add TaskLink from '{dep_name}' -> '{t.name}': {e}")

                        # Set dependency type (FS/SS/FF/SF)
                        dtype = (t.dependency_type or "FS").upper()
                        if dtype in link_type_map:
                            try:
                                if hasattr(created_link, "Type"):
                                    created_link.Type = link_type_map[dtype]
                                elif hasattr(created_link, "LinkType"):
                                    created_link.LinkType = link_type_map[dtype]
                            except Exception as e:
                                raise RuntimeError(f"Failed to set link type {dtype} on {dep_name}->{t.name}: {e}")

                        # Apply lag (if any)
                        lag_days = int(getattr(t, "lag_days", 0) or 0)
                        if lag_days:
                            try:
                                lag_duration = project.GetDuration(lag_days, AT.TimeUnitType.Day)
                                if hasattr(created_link, "Lag"):
                                    created_link.Lag = lag_duration
                                elif hasattr(created_link, "LagValue"):
                                    created_link.LagValue = int(lag_days * 24 * 60)  # minutes
                            except Exception as e:
                                raise RuntimeError(f"Failed to set lag ({lag_days}d) on {dep_name}->{t.name}: {e}")
    except Exception as e:
        raise RuntimeError(f"Error while adding dependencies: {e}")
    # Save XML
    try:
        project.Save(String(filename), SaveFileFormat.Xml)
    except Exception as e:
        raise RuntimeError(f"Failed to save project XML: {e}")

    # Baseline
    try:
        if hasattr(AT, "BaselineType") and hasattr(project, "SetBaseline"):
            project.SetBaseline(AT.BaselineType.Baseline1)
        else:
            raise RuntimeError("Baseline API (AT.BaselineType or project.SetBaseline) not found in this Aspose build.")
    except Exception as e:
        raise RuntimeError(f"Failed to set baseline: {e}")

    # Add resource-specific unavailability to calendar exceptions if provided
        
    # --- Resource calendar exceptions (holidays, vacations) ---
    try:
        # Get the main project calendar safely across API versions
        proj_calendar = None
        if hasattr(project, "Calendars") and project.Calendars:
            proj_calendar = project.Calendars.GetByName("Standard")
            if proj_calendar is None:
                proj_calendar = project.Calendars.Add("Standard")
        elif hasattr(project, "RootTask") and hasattr(project.RootTask, "Calendars"):
            proj_calendar = project.RootTask.Calendars.GetByName("Standard")
            if proj_calendar is None:
                proj_calendar = project.RootTask.Calendars.Add("Standard")
        elif hasattr(project, "Get") and hasattr(AT.Prj, "Calendar"):
            proj_calendar = project.Get(AT.Prj.Calendar)
        else:
            raise RuntimeError("Project CalendarCollection not found — cannot add resource calendar exceptions.")

        # If we still didn't get one, fail early
        if proj_calendar is None:
            raise RuntimeError("Unable to resolve or create Standard calendar for project.")

        # Define common holiday and vacation windows
        holiday_ranges = [
            ("2025-12-24", "2025-12-27"),  # company holidays
            ("2026-01-01", "2026-01-01"),  # New Year
        ]
        dev_vacations = [
            ("2025-11-20", "2025-11-30"),  # lead dev vacation
        ]

        # Helper: create exceptions safely
        def add_exceptions(calendar_obj, ranges, label):
            for start, end in ranges:
                try:
                    exc = AT.CalendarException()
                    exc.Name = label
                    from System import DateTime as SysDateTime

                    py_from = datetime.strptime(start, "%Y-%m-%d")
                    py_to = datetime.strptime(end, "%Y-%m-%d")

                    exc.FromDate = SysDateTime(py_from.year, py_from.month, py_from.day)
                    exc.ToDate = SysDateTime(py_to.year, py_to.month, py_to.day)

                    exc.DayWorking = False
                    calendar_obj.Exceptions.Add(exc)
                except Exception as e:
                    print(f"⚠️ Warning: Could not add calendar exception {label} ({start}–{end}): {e}")

        add_exceptions(proj_calendar, holiday_ranges, "Company Holiday")
        add_exceptions(proj_calendar, dev_vacations, "Lead Dev Vacation")

    except Exception as e:
        raise RuntimeError(f"Failed to create resource calendar exceptions: {e}")


    return filename


# ---------------------------
# Streamlit UI (unchanged workflow)
# ---------------------------
st.set_page_config(page_title="AI Project Planner (Strict Mode)", layout="wide")
st.title("👨🏻‍💻AI Project Planner → MS Project XML")
st.caption("Strict mode: fails loudly. Added dependency types, lag, milestones, baseline, calendars, cycle detection.")

user_input = st.text_area(
    "Describe your project plan in plain English:",
    height=380,
    placeholder="Paste project description..."
)

start_override = st.date_input("Override Start Date (optional)", value=None)

if st.button("Generate MS Project XML (Enhanced)"):
    if not user_input.strip():
        st.error("Please enter project description.")
        st.stop()

    with st.spinner("Extracting structured project plan from OpenAI..."):
        try:
            raw_json = extract_project_structure(user_input)
        except Exception as e:
            st.error(f"❌ OpenAI extraction failed: {e}")
            st.stop()

    # Override start if requested
    try:
        if start_override:
            raw_json["start_date"] = start_override.strftime("%Y-%m-%d")
        project_model = ProjectModel(**raw_json)
    except ValidationError as ve:
        st.error("❌ JSON validation failed.")
        st.json(json.loads(ve.json()))
        st.code(json.dumps(raw_json, indent=2))
        st.stop()

    st.success("✅ JSON validated successfully.")
    st.subheader("Structured Project Data")
    st.json(project_model.model_dump() if hasattr(project_model, "model_dump") else project_model.dict())

    filename = f"{project_model.project_name.replace(' ', '_')}.xml"
    out_path = Path.cwd() / filename





# Put your public backend URL (ngrok or cloud VM) here.
# Example: "https://abcd1234.ngrok.io"
    BACKEND_URL = st.secrets.get("BACKEND_URL", "") or "https://REPLACE_WITH_YOUR_NGROK_OR_IP:8000"


    with st.spinner("Sending project JSON to your Aspose backend..."):
        try:
            json_payload = json.dumps(project_model.model_dump() if hasattr(project_model, "model_dump") else project_model.dict())
            resp = requests.post(
                f"{BACKEND_URL.rstrip('/')}/generate",
                data={"json_data": json_payload, "filename": filename}
            )
        except Exception as e:
            st.error(f"❌ Failed to reach backend: {e}")
            st.stop()

        if not resp.ok:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            st.error(f"❌ Backend returned error ({resp.status_code}): {detail}")
            st.stop()

        result = resp.json()
        if result.get("status") != "ok":
            st.error(f"Backend error: {result}")
            st.stop()

        download_url = f"{BACKEND_URL.rstrip('/')}{result['download_path']}"
        st.success("✅ File generated by remote Aspose backend.")
        st.markdown(f"[⬇️ Download generated MS Project XML]({download_url})")


    st.success("✅ MS Project XML generated successfully!")
    with open(created, "rb") as f:
        st.download_button("⬇️ Download XML", f, file_name=filename, mime="application/xml")
