from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from reporting import generate_report, write_reports
from visualization import generate_motion_plot_video


WORKFLOW_MAP = {
    "fast": "internet",
    "balanced": "internet",
    "accurate": "internet-rotate",
}

SMPL_MODEL_REL = Path("data/bodymodels/SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl")
SMPL_MODEL_FALLBACKS = [
    Path("/Users/manoharpaturi/Desktop/mocap_manu/basicmodel_m_lbs_10_207_0_v1.0.0.npz"),
    Path("/Users/manoharpaturi/Desktop/mocap_manu/basicModel_f_lbs_10_207_0_v1.0.0.npz"),
]
WORKFLOW_OVERRIDE_REL = Path("config/mocap_workflow_local_runtime.yml")
FIT_OVERRIDE_REL = Path("config/fit/1v1p_local_runtime.yml")

STATUS_PROGRESS = {
    "queued": 2,
    "running": 10,
    "completed": 100,
    "failed": 100,
}


@dataclass
class Job:
    job_id: str
    status: str
    created_at: str
    updated_at: str
    mode: str
    workflow: str
    dataset_dir: str
    output_dir: str
    input_video: str
    log_file: str
    progress: int = 0
    stage: str = "queued"
    error: str | None = None
    report_json: str | None = None
    report_md: str | None = None


class JobStore:
    def __init__(self, index_file: Path) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._index_file = index_file
        self._index_file.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        if not self._index_file.exists():
            return
        try:
            raw = json.loads(self._index_file.read_text(encoding="utf-8"))
        except Exception:
            return
        jobs = raw if isinstance(raw, list) else []
        for item in jobs:
            try:
                job = Job(**item)
            except TypeError:
                continue
            self._jobs[job.job_id] = job

    def _save(self) -> None:
        payload = [asdict(job) for job in sorted(self._jobs.values(), key=lambda x: x.created_at, reverse=True)]
        self._index_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def create(self, mode: str, dataset_dir: Path, output_dir: Path, input_video: Path, log_file: Path, workflow: str) -> Job:
        now = datetime.utcnow().isoformat()
        job = Job(
            job_id=str(uuid.uuid4()),
            status="queued",
            created_at=now,
            updated_at=now,
            mode=mode,
            workflow=workflow,
            dataset_dir=str(dataset_dir),
            output_dir=str(output_dir),
            input_video=str(input_video),
            log_file=str(log_file),
            progress=STATUS_PROGRESS["queued"],
            stage="queued",
        )
        with self._lock:
            self._jobs[job.job_id] = job
            self._save()
        return job

    def update(self, job_id: str, **kwargs: Any) -> None:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in kwargs.items():
                setattr(job, key, value)
            if "status" in kwargs and "progress" not in kwargs:
                job.progress = STATUS_PROGRESS.get(job.status, job.progress)
            job.updated_at = datetime.utcnow().isoformat()
            self._save()

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda x: x.created_at, reverse=True)
            return [asdict(job) for job in jobs]

    def remove(self, job_id: str) -> bool:
        with self._lock:
            if job_id not in self._jobs:
                return False
            del self._jobs[job_id]
            self._save()
            return True

    def to_dict(self, job_id: str) -> dict[str, Any] | None:
        job = self.get(job_id)
        if job is None:
            return None
        return asdict(job)


def save_upload_to_dataset(upload_path: Path, dataset_dir: Path) -> Path:
    videos_dir = dataset_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    target = videos_dir / "input.mp4"
    shutil.copy2(upload_path, target)
    return target


def list_artifacts(dataset_dir: Path) -> list[str]:
    allowed_roots = [
        dataset_dir / "analysis",
        dataset_dir / "output-smpl-3d",
    ]
    suffixes = {".mp4", ".json", ".md", ".log", ".yml", ".yaml"}
    artifacts: list[str] = []

    for root in allowed_roots:
        if not root.exists():
            continue
        for file_path in root.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in suffixes:
                continue
            artifacts.append(str(file_path.relative_to(dataset_dir)))
    return sorted(set(artifacts))


def _infer_stage_from_log(log_text: str) -> tuple[str, int]:
    stage = "running"
    progress = 20

    if "extract_image.py" in log_text:
        stage, progress = "extracting frames", 25
    if "extract_keypoints.py" in log_text or "[Skip] detection" in log_text:
        stage, progress = "detecting keypoints", 45
    if "apps/fit/fit.py" in log_text:
        stage, progress = "fitting SMPL", 75
    if "ffmpeg_wrapper" in log_text:
        stage, progress = "rendering output", 88
    if "Report generated." in log_text:
        stage, progress = "finalizing report", 96

    return stage, progress


def _run_command(cmd: list[str], cwd: Path, log_file_path: Path) -> None:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    repo_root = str(cwd)
    env["PYTHONPATH"] = f"{repo_root}:{existing_pythonpath}" if existing_pythonpath else repo_root

    with log_file_path.open("a", encoding="utf-8") as log_fp:
        log_fp.write("$ " + " ".join(cmd) + "\n")
        log_fp.write(f"PYTHONPATH={env['PYTHONPATH']}\n")
        log_fp.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return_code = proc.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}")


def _resolve_smpl_model_path(easy_root: Path) -> Path | None:
    primary_path = (easy_root / SMPL_MODEL_REL).resolve()
    if primary_path.exists():
        return primary_path
    for candidate in SMPL_MODEL_FALLBACKS:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    return None


def _write_workflow_override(easy_root: Path, smpl_model_path: Path) -> None:
        fit_override_rel = str(FIT_OVERRIDE_REL)
        override_path = easy_root / WORKFLOW_OVERRIDE_REL
        override_text = f'''internet:
    calibration: "python3 apps/calibration/create_blank_camera.py ${{data}}"
    extract_image: none
    extract_keypoints: "python3 apps/preprocess/extract_keypoints.py ${{data}} --mode yolo-hrnet"
    fit:
        data: config/data/multivideo.yml
        model: config/model/smpl.yml
        exp: {fit_override_rel}
        opt_model: [args.model_path, "{smpl_model_path}"]
    output: output-smpl-3d
internet-rotate:
    calibration: "python3 apps/calibration/create_blank_camera.py ${{data}}"
    extract_image: none
    extract_keypoints: "python3 apps/preprocess/extract_keypoints.py ${{data}} --mode yolo-hrnet"
    fit:
        data: config/data/multivideo.yml
        model: config/model/smpl.yml
        exp: {fit_override_rel}
        opt_model: [args.model_path, "{smpl_model_path}"]
    output: output-smpl-3d
'''
        fit_override_path = easy_root / FIT_OVERRIDE_REL
        fit_override_path.parent.mkdir(parents=True, exist_ok=True)
        fit_override_text = '''module: easymocap.multistage.base.MultiStage
args:
    batch_size: 1
    _parents_:
        - config/fit/lbfgs.yml
    initialize:
        init_pose:
            module: easymocap.multistage.init_pose.SmoothPoses
            args:
                window_size: 2
    stages:
        camera:
            optimize: [Th, Rh]
            repeat: 1
            before_after: {}
            loss:
                k2d:
                    weight: 100.0
                    module: easymocap.multistage.lossbase.Keypoints2D
                    infos: [keypoints2d, K, Rc, Tc]
                    args:
                        index_est: [2, 5, 9, 12]
                        index_gt: [2, 5, 9, 12]
                        norm: l2
                        unproj: True
                smoothTh:
                    weight: 100.
                    module: easymocap.multistage.lossbase.AnySmooth
                    args:
                        key: Th
                        weight: [0.5, 0.3, 0.1, 0.1]
                        norm: l2
                        order: 2
                smoothDepth:
                    weight: 1000.
                    module: easymocap.multistage.lossbase.AnySmooth
                    args:
                        key: Th
                        index: [2]
                        weight: [0.5, 0.3, 0.1, 0.1]
                        norm: l2
                        order: 2
                smoothRh:
                    weight: 100.
                    module: easymocap.multistage.lossbase.SmoothRot
                    args:
                        key: Rh
                        weight: [0.5, 0.3, 0.1, 0.1]
                        norm: l2
                        order: 2
        joints:
            optimize: [poses, Th, Rh]
            repeat: 2
            loss:
                k2d:
                    weight: 1000.
                    module: easymocap.multistage.lossbase.Keypoints2D
                    infos: [keypoints2d, K, Rc, Tc]
                    args:
                        norm: l2
                        norm_info: 0.02
                        unproj: True
                init:
                    weight: 1.
                    module: easymocap.multistage.lossbase.AnyReg
                    infos: [init_poses]
                    args:
                        key: poses
                        norm: l2
                pf-smoothtorso:
                    weight: 100.
                    module: easymocap.multistage.lossbase.AnySmooth
                    args:
                        key: poses_full
                        index: [0,1,2,9,10,11,12,13,14,18,19,20,39,40,41,42,43,44]
                        weight: [0.5, 0.3, 0.1, 0.1]
                        norm: l2
                        order: 2
                smoothposes:
                    weight: 100.
                    module: easymocap.multistage.lossbase.AnySmooth
                    args:
                        key: poses
                        weight: [0.5, 0.3, 0.1, 0.1]
                        norm: l2
                        order: 2
                smoothTh:
                    weight: 100.
                    module: easymocap.multistage.lossbase.AnySmooth
                    args:
                        key: Th
                        weight: [0.5, 0.3, 0.1, 0.1]
                        norm: l2
                        order: 2
                smoothDepth:
                    weight: 1000.
                    module: easymocap.multistage.lossbase.AnySmooth
                    args:
                        key: Th
                        index: [2]
                        weight: [0.5, 0.3, 0.1, 0.1]
                        norm: l2
                        order: 2
                smoothRh:
                    weight: 100.
                    module: easymocap.multistage.lossbase.SmoothRot
                    args:
                        key: Rh
                        weight: [0.5, 0.3, 0.1, 0.1]
                        norm: l2
                        order: 2
'''
        fit_override_path.write_text(fit_override_text, encoding="utf-8")
        override_path.write_text(override_text, encoding="utf-8")


def run_job(job_store: JobStore, job_id: str, python_bin: str, easy_root: Path) -> None:
    job = job_store.get(job_id)
    if job is None:
        return

    dataset_dir = Path(job.dataset_dir)
    output_dir = Path(job.output_dir)
    input_video = Path(job.input_video)
    log_file = Path(job.log_file)

    job_store.update(job_id, status="running", stage="starting workflow", progress=10)
    try:
        smpl_model_path = _resolve_smpl_model_path(easy_root)
        if smpl_model_path is None:
            with log_file.open("a", encoding="utf-8") as log_fp:
                log_fp.write(
                    "Missing SMPL model file.\n"
                    f"Expected: {(easy_root / SMPL_MODEL_REL).resolve()}\n"
                    "Fallbacks checked: /Users/manoharpaturi/Desktop/mocap_manu/basicmodel_m_lbs_10_207_0_v1.0.0.npz, "
                    "/Users/manoharpaturi/Desktop/mocap_manu/basicModel_f_lbs_10_207_0_v1.0.0.npz\n"
                )
                log_fp.flush()
            raise RuntimeError(f"Missing SMPL model file: {(easy_root / SMPL_MODEL_REL).resolve()}")

        with log_file.open("a", encoding="utf-8") as log_fp:
            log_fp.write(f"Start: {datetime.utcnow().isoformat()}\n")
            log_fp.write(f"Mode: {job.mode}, Workflow: {job.workflow}\n")
            log_fp.write(f"SMPL model: {smpl_model_path}\n")
            log_fp.flush()

        _write_workflow_override(easy_root, smpl_model_path)

        cmd = [
            python_bin,
            "apps/demo/mocap.py",
            "--work",
            job.workflow,
            str(dataset_dir),
            "--fps",
            "30",
        ]
        _run_command(cmd, cwd=easy_root, log_file_path=log_file)

        if log_file.exists():
            stage, progress = _infer_stage_from_log(log_file.read_text(encoding="utf-8", errors="ignore"))
            job_store.update(job_id, stage=stage, progress=progress)

        report = generate_report(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            input_video=input_video,
            workflow=job.workflow,
        )
        report_json, report_md = write_reports(dataset_dir, report)

        try:
            generate_motion_plot_video(dataset_dir=dataset_dir, output_dir=output_dir, fps=30.0)
        except Exception as vis_exc:
            with log_file.open("a", encoding="utf-8") as log_fp:
                log_fp.write(f"Plot visualization generation skipped: {vis_exc}\n")
                log_fp.flush()

        with log_file.open("a", encoding="utf-8") as log_fp:
            log_fp.write("Report generated.\n")
            log_fp.flush()

        job_store.update(
            job_id,
            status="completed",
            progress=100,
            stage="completed",
            report_json=str(report_json),
            report_md=str(report_md),
        )
    except Exception as exc:
        job_store.update(job_id, status="failed", stage="failed", progress=100, error=str(exc))


def create_dataset_from_upload(jobs_root: Path, uploaded_file: Path, mode: str) -> tuple[Path, Path, Path]:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    dataset_dir = jobs_root / f"job_{timestamp}_{uuid.uuid4().hex[:8]}"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    input_video = save_upload_to_dataset(uploaded_file, dataset_dir)
    log_file = dataset_dir / "run.log"

    metadata = {
        "mode": mode,
        "uploaded_at": datetime.utcnow().isoformat(),
        "input_video": str(input_video),
    }
    (dataset_dir / "job_meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return dataset_dir, input_video, log_file
