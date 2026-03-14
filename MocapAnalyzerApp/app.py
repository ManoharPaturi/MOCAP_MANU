from __future__ import annotations

import os
import json
import sqlite3
import shutil
import sys
import tempfile
import threading
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from mocap_service import (
    WORKFLOW_MAP,
    JobStore,
    create_dataset_from_upload,
    list_artifacts,
    run_job,
)
from visualization import (
    generate_full_nodes_analysis_video,
    generate_motion_plot_video,
    generate_stickman_from_motion_capture_db,
    generate_stickman_video,
)


APP_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = APP_ROOT.parent
EASY_ROOT = (WORKSPACE_ROOT / "EasyMocap").resolve()
V1_RESULTS_ROOT = (WORKSPACE_ROOT / "V1" / "results").resolve()
JOBS_ROOT = (APP_ROOT / "jobs").resolve()
JOBS_INDEX = (JOBS_ROOT / "jobs_index.json").resolve()
ANALYZER_DB = (JOBS_ROOT / "analyzer.db").resolve()
PYTHON_BIN = sys.executable
MAX_UPLOAD_MB = int(os.environ.get("MOCAP_MAX_UPLOAD_MB", "512"))
SMPL_MODEL_REL = Path("data/bodymodels/SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl")
SMPL_MODEL_FALLBACKS = [
    WORKSPACE_ROOT / "basicmodel_m_lbs_10_207_0_v1.0.0.npz",
    WORKSPACE_ROOT / "basicModel_f_lbs_10_207_0_v1.0.0.npz",
]

JOBS_ROOT.mkdir(parents=True, exist_ok=True)


def _init_analyzer_db() -> None:
    conn = sqlite3.connect(str(ANALYZER_DB))
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS external_visualizations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL,
            source_folder TEXT NOT NULL,
            source_table TEXT,
            output_path TEXT,
            status TEXT NOT NULL,
            error TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def _record_external_visualization(
    *,
    job_id: str,
    source_folder: Path,
    source_table: str | None,
    output_path: Path | None,
    status: str,
    error: str | None,
) -> None:
    conn = sqlite3.connect(str(ANALYZER_DB))
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO external_visualizations
        (job_id, source_folder, source_table, output_path, status, error, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            job_id,
            str(source_folder),
            source_table,
            str(output_path) if output_path is not None else None,
            status,
            error,
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    conn.close()


_init_analyzer_db()
V1_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="manu_mocap", version="1.1.0")
app.mount("/static", StaticFiles(directory=str(APP_ROOT / "static")), name="static")

job_store = JobStore(index_file=JOBS_INDEX)


def _safe_artifact_path(dataset_dir: Path, rel_path: str) -> Path:
    candidate = (dataset_dir / rel_path).resolve()
    dataset_real = dataset_dir.resolve()
    if not str(candidate).startswith(str(dataset_real)):
        raise HTTPException(status_code=400, detail="Invalid artifact path")
    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return candidate


def _resolve_smpl_model_path() -> Path | None:
    primary_path = (EASY_ROOT / SMPL_MODEL_REL).resolve()
    if primary_path.exists():
        return primary_path
    for candidate in SMPL_MODEL_FALLBACKS:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    return None


def _copy_to_v1_results(video_path: Path, stem_prefix: str) -> Path:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    target = V1_RESULTS_ROOT / f"{stem_prefix}_{ts}.mp4"
    shutil.copy2(video_path, target)
    return target


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html_path = APP_ROOT / "templates" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/health")
def health() -> JSONResponse:
    smpl_model_path = _resolve_smpl_model_path()
    return JSONResponse(
        {
            "ok": True,
            "easy_mocap_exists": EASY_ROOT.exists(),
            "smpl_model_exists": smpl_model_path is not None,
            "smpl_model_path": str(smpl_model_path) if smpl_model_path is not None else str((EASY_ROOT / SMPL_MODEL_REL).resolve()),
            "python": PYTHON_BIN,
            "max_upload_mb": MAX_UPLOAD_MB,
        }
    )


@app.get("/api/jobs")
def list_jobs() -> JSONResponse:
    return JSONResponse({"jobs": job_store.list()})


@app.post("/api/jobs")
async def create_job(mode: str = Form("balanced"), video: UploadFile = File(...)) -> JSONResponse:
    if mode not in WORKFLOW_MAP:
        raise HTTPException(status_code=400, detail="Invalid mode")

    suffix = Path(video.filename or "input.mp4").suffix.lower()
    if suffix not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        raise HTTPException(status_code=400, detail="Unsupported video format")

    if not EASY_ROOT.exists():
        raise HTTPException(status_code=500, detail="EasyMocap folder not found")

    smpl_model_path = _resolve_smpl_model_path()
    if smpl_model_path is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "Missing SMPL model file. "
                f"Expected: {(EASY_ROOT / SMPL_MODEL_REL).resolve()}. "
                "Or place one of these fallback files in the workspace root: "
                "basicmodel_m_lbs_10_207_0_v1.0.0.npz, basicModel_f_lbs_10_207_0_v1.0.0.npz."
            ),
        )

    content = await video.read()
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Upload too large. Max is {MAX_UPLOAD_MB} MB")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    dataset_dir, input_video, log_file = create_dataset_from_upload(JOBS_ROOT, tmp_path, mode)
    tmp_path.unlink(missing_ok=True)

    workflow = WORKFLOW_MAP[mode]
    output_dir = dataset_dir / "output-smpl-3d"
    job = job_store.create(
        mode=mode,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        input_video=input_video,
        log_file=log_file,
        workflow=workflow,
    )

    thread = threading.Thread(
        target=run_job,
        args=(job_store, job.job_id, PYTHON_BIN, EASY_ROOT),
        daemon=True,
    )
    thread.start()

    return JSONResponse({"job_id": job.job_id, "status": job.status, "dataset_dir": str(dataset_dir)})


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> JSONResponse:
    data = job_store.to_dict(job_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Job not found")

    log_file = Path(data["log_file"])
    if data["status"] == "running" and log_file.exists():
        text = log_file.read_text(encoding="utf-8", errors="ignore")
        stage = data.get("stage", "running")
        progress = data.get("progress", 10)
        if "extract_keypoints.py" in text:
            stage, progress = "detecting keypoints", max(progress, 45)
        elif "apps/fit/fit.py" in text:
            stage, progress = "fitting SMPL", max(progress, 75)
        elif "ffmpeg_wrapper" in text:
            stage, progress = "rendering output", max(progress, 88)
        if stage != data.get("stage") or progress != data.get("progress"):
            job_store.update(job_id, stage=stage, progress=progress)
            data = job_store.to_dict(job_id) or data

    return JSONResponse(data)


@app.get("/api/jobs/{job_id}/log")
def get_job_log(job_id: str) -> JSONResponse:
    data = job_store.to_dict(job_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Job not found")
    log_file = Path(data["log_file"])
    if not log_file.exists():
        return JSONResponse({"log": ""})
    return JSONResponse({"log": log_file.read_text(encoding="utf-8", errors="ignore")[-25000:]})


@app.get("/api/jobs/{job_id}/report")
def get_job_report(job_id: str) -> JSONResponse:
    data = job_store.to_dict(job_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if data.get("report_json") is None:
        raise HTTPException(status_code=404, detail="Report not ready")
    report_path = Path(data["report_json"])
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report file missing")
    import json

    return JSONResponse(json.loads(report_path.read_text(encoding="utf-8")))


@app.get("/api/jobs/{job_id}/report.md")
def get_job_report_md(job_id: str):
    data = job_store.to_dict(job_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if data.get("report_md") is None:
        raise HTTPException(status_code=404, detail="Markdown report not ready")
    md_path = Path(data["report_md"])
    if not md_path.exists():
        raise HTTPException(status_code=404, detail="Markdown report file missing")
    return FileResponse(str(md_path), media_type="text/markdown", filename=f"{job_id}_report.md")


@app.get("/api/jobs/{job_id}/artifacts")
def get_job_artifacts(job_id: str) -> JSONResponse:
    data = job_store.to_dict(job_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Job not found")
    dataset_dir = Path(data["dataset_dir"])
    artifacts = list_artifacts(dataset_dir)
    return JSONResponse({"artifacts": artifacts})


@app.post("/api/jobs/{job_id}/visualize/plots")
def generate_plots_video(job_id: str) -> JSONResponse:
    data = job_store.to_dict(job_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Job not found")

    dataset_dir = Path(data["dataset_dir"])
    output_dir = Path(data["output_dir"])

    try:
        out_path = generate_motion_plot_video(dataset_dir=dataset_dir, output_dir=output_dir, fps=30.0)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to generate plots video: {exc}") from exc

    rel = str(out_path.relative_to(dataset_dir))
    return JSONResponse({"ok": True, "artifact": rel})


@app.post("/api/jobs/{job_id}/visualize/stickman")
def generate_stickman_animation(job_id: str) -> JSONResponse:
    data = job_store.to_dict(job_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Job not found")

    dataset_dir = Path(data["dataset_dir"])
    try:
        out_path = generate_stickman_video(dataset_dir=dataset_dir, fps=30.0, min_conf=0.2)
        v1_path = _copy_to_v1_results(out_path, f"stickman_{job_id[:8]}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to generate stickman animation: {exc}") from exc

    rel = str(out_path.relative_to(dataset_dir))
    return JSONResponse({"ok": True, "artifact": rel, "v1_result": str(v1_path)})


@app.post("/api/jobs/{job_id}/visualize/fullnodes")
def generate_full_nodes_animation(
    job_id: str,
    slow_factor: float = 1.0,
    pose_model_variant: str = "lite",
) -> JSONResponse:
    data = job_store.to_dict(job_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Job not found")

    dataset_dir = Path(data["dataset_dir"])
    input_video = Path(data["input_video"])
    out_path = dataset_dir / "analysis" / "full_nodes_analysis.mp4"

    try:
        out_path = generate_full_nodes_analysis_video(
            input_video=input_video,
            output_path=out_path,
            min_conf=0.2,
            slow_factor=slow_factor,
            pose_model_variant=pose_model_variant,
        )
        v1_path = _copy_to_v1_results(out_path, f"fullnodes_{job_id[:8]}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to generate full-node analysis video: {exc}") from exc

    rel = str(out_path.relative_to(dataset_dir))
    return JSONResponse({"ok": True, "artifact": rel, "v1_result": str(v1_path)})


@app.post("/api/external/motion-capture/stickman")
def generate_external_motion_capture_stickman(folder_path: str, table_name: str | None = None) -> JSONResponse:
    motion_dir = Path(folder_path).resolve()
    if not motion_dir.exists() or not motion_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"Invalid folder path: {folder_path}")

    dataset_dir = JOBS_ROOT / f"external_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "analysis").mkdir(parents=True, exist_ok=True)
    log_file = dataset_dir / "run.log"
    output_dir = dataset_dir / "analysis"
    input_placeholder = dataset_dir / "videos" / "external.mp4"
    input_placeholder.parent.mkdir(parents=True, exist_ok=True)

    job = job_store.create(
        mode="external-mediapipe",
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        input_video=input_placeholder,
        log_file=log_file,
        workflow="mediapipe-stickman",
    )
    job_store.update(job.job_id, status="running", stage="importing external mocap", progress=20)

    meta_path = dataset_dir / "external_source.json"
    meta_path.write_text(
        json.dumps(
            {
                "source_folder": str(motion_dir),
                "source_table": table_name,
                "created_at": datetime.utcnow().isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    try:
        out_path = generate_stickman_from_motion_capture_db(
            motion_capture_dir=motion_dir,
            output_path=dataset_dir / "analysis" / "stickman_animation.mp4",
            table_name=table_name,
            fps=30.0,
            width=1280,
            height=720,
            min_conf=0.2,
            draw_face=True,
            draw_hands=True,
        )
        v1_path = _copy_to_v1_results(out_path, f"external_stickman_{job.job_id[:8]}")
        with log_file.open("a", encoding="utf-8") as fp:
            fp.write(f"External source: {motion_dir}\n")
            fp.write(f"Table: {table_name or 'latest non-empty'}\n")
            fp.write(f"Output: {out_path}\n")
            fp.write(f"V1 copy: {v1_path}\n")

        _record_external_visualization(
            job_id=job.job_id,
            source_folder=motion_dir,
            source_table=table_name,
            output_path=out_path,
            status="completed",
            error=None,
        )
        job_store.update(job.job_id, status="completed", stage="completed", progress=100)
    except Exception as exc:
        with log_file.open("a", encoding="utf-8") as fp:
            fp.write(f"External source: {motion_dir}\n")
            fp.write(f"Table: {table_name or 'latest non-empty'}\n")
            fp.write(f"Error: {exc}\n")

        _record_external_visualization(
            job_id=job.job_id,
            source_folder=motion_dir,
            source_table=table_name,
            output_path=None,
            status="failed",
            error=str(exc),
        )
        job_store.update(job.job_id, status="failed", stage="failed", progress=100, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Failed to generate external stickman: {exc}") from exc

    rel = str(out_path.relative_to(dataset_dir))
    return JSONResponse({"ok": True, "job_id": job.job_id, "artifact": rel, "dataset_dir": str(dataset_dir), "v1_result": str(v1_path)})


@app.get("/api/jobs/{job_id}/artifacts/file/{artifact_path:path}")
def download_artifact(job_id: str, artifact_path: str):
    data = job_store.to_dict(job_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Job not found")
    dataset_dir = Path(data["dataset_dir"])
    file_path = _safe_artifact_path(dataset_dir, artifact_path)
    return FileResponse(str(file_path), filename=file_path.name)


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> JSONResponse:
    data = job_store.to_dict(job_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Job not found")
    dataset_dir = Path(data["dataset_dir"])
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    job_store.remove(job_id)
    return JSONResponse({"deleted": True, "job_id": job_id})
