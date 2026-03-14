# Mocap Analyzer App

A complete upload-to-analysis application built on top of EasyMocap monocular workflow methods.

## Features

- Upload one video from browser.
- Builds EasyMocap dataset layout automatically.
- Runs EasyMocap workflow:
  - `internet` for `fast` and `balanced`
  - `internet-rotate` for `accurate`
- Persistent job history (`jobs/jobs_index.json`).
- Background processing with stage and progress updates.
- Full report generation:
  - video metadata,
  - frame extraction coverage,
  - annotation and detection counts,
  - SMPL tracking and motion statistics,
  - generated output artifacts.
- Artifact listing and download endpoints.

## Why this matches EasyMocap research workflow

EasyMocap includes workflow definitions in `config/mocap_workflow.yml` and fit profiles in `config/mocap_index.yml`:

- `internet` workflow: monocular processing for web videos.
- `mono-smpl-robust` fit: robust monocular SMPL optimization.
- `internet-rotate` workflow: better rotation handling for dynamic motion.

This app directly calls:

```bash
python apps/demo/mocap.py --work internet <dataset>
```

or:

```bash
python apps/demo/mocap.py --work internet-rotate <dataset>
```

## Setup

From workspace root:

```bash
source /Users/manoharpaturi/manu/bin/activate
cd /Users/manoharpaturi/Desktop/mocap_manu/MocapAnalyzerApp
pip install -r requirements.txt
```

## Run

```bash
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

Then open:

- http://localhost:8080

## Config

Environment variables:

- `MOCAP_MAX_UPLOAD_MB` (default `512`): max upload size in MB.

## API Summary

- `GET /api/health`
- `GET /api/jobs`
- `POST /api/jobs`
- `GET /api/jobs/{job_id}`
- `GET /api/jobs/{job_id}/log`
- `GET /api/jobs/{job_id}/report`
- `GET /api/jobs/{job_id}/report.md`
- `GET /api/jobs/{job_id}/artifacts`
- `GET /api/jobs/{job_id}/artifacts/file/{artifact_path}`
- `DELETE /api/jobs/{job_id}`

## Output layout per job

Each job is stored under `MocapAnalyzerApp/jobs/job_<timestamp>_<id>/`:

- `videos/input.mp4`
- `images/...`
- `annots/...`
- `output-smpl-3d/...`
- `analysis/report.json`
- `analysis/report.md`
- `run.log`

## Notes

- EasyMocap model checkpoints and dependencies must be installed for full execution.
- Runtime depends on GPU/CPU and video length.
- The report parser is robust to partial outputs and still returns what is available.
