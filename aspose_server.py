# aspose_server.py
"""
FastAPI wrapper around your local Aspose save function.

Run on your Windows PC where Aspose.Tasks and pythonnet work.
Expects ai_project_planner_strict.save_aspose_project and ProjectModel to be importable.
"""

import json
import os
from pathlib import Path
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import FileResponse
import uvicorn
import traceback

# If ai_project_planner_strict.py is in the same directory or repo root,
# make sure Python can import it. Adjust path if necessary.
HERE = Path(__file__).parent.resolve()
import sys
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    # import the save function and ProjectModel from your file
    from ai_project_planner_strict import save_aspose_project, ProjectModel
except Exception as e:
    raise RuntimeError(f"Failed to import save_aspose_project from ai_project_planner_strict: {e}")

app = FastAPI(title="Aspose Tasks Local Backend")

# simple health check
@app.get("/health")
def health():
    return {"status": "ok"}

# generate endpoint: accepts JSON string (project model) and returns download URL or error
@app.post("/generate")
def generate(json_data: str = Form(...), filename: str = Form(None)):
    try:
        data = json.loads(json_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {e}")

    # validate with Pydantic model
    try:
        project = ProjectModel(**data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Validation error: {e}")

    # choose an output filename if caller didn't send one
    safe_name = (filename or project.project_name.replace(" ", "_")) + ".xml"
    out_path = HERE / "generated" 
    out_path.mkdir(exist_ok=True)
    out_file = str((out_path / safe_name).resolve())

    try:
        # call your existing save function — it will raise on error
        created_filename = save_aspose_project(project, out_file)
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}\n{tb}")

    # return path to download (backend serves file directly)
    if not Path(created_filename).exists():
        raise HTTPException(status_code=500, detail="Created file not found after generation.")

    return {"status": "ok", "filename": Path(created_filename).name, "download_path": f"/download/{Path(created_filename).name}"}

@app.get("/download/{fn}")
def download(fn: str):
    file_path = HERE / "generated" / fn
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(file_path), media_type="application/xml", filename=fn)


if __name__ == "__main__":
    # Default: listen on all interfaces so ngrok or other tunnel can reach it. Use firewall rules to restrict.
    uvicorn.run("aspose_server:app", host="0.0.0.0", port=8000, reload=False)
