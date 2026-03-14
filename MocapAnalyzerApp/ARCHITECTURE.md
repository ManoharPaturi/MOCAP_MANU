# Mocap Analyzer App Architecture

## 1. Purpose

Mocap Analyzer App provides an upload to analysis pipeline on top of EasyMocap monocular workflows.

Primary user goal:
- Upload one video.
- Run EasyMocap processing automatically.
- Receive a structured analysis report with generated artifacts.

## 2. System Context

Workspace components:
- MocapAnalyzerApp: Web application and report layer.
- EasyMocap: Research engine and workflow execution backend.

Execution boundary:
- MocapAnalyzerApp orchestrates jobs and report generation.
- EasyMocap performs extraction, detection, fitting, and rendering.

## 3. High Level Flow

1. User uploads video in web UI.
2. Backend creates a job folder under jobs.
3. Video is copied into dataset style folder at videos/input.mp4.
4. Backend launches EasyMocap workflow command:
   - apps/demo/mocap.py --work internet <dataset> for fast and balanced mode.
   - apps/demo/mocap.py --work internet-rotate <dataset> for accurate mode.
5. EasyMocap writes images, annots, output-smpl-3d and mesh outputs.
6. Report generator scans outputs and computes motion statistics.
7. Report is stored as analysis/report.json and analysis/report.md.
8. UI polls status and displays logs and final report.

## 4. Component Map

### 4.1 API Layer

File: app.py

Responsibilities:
- Exposes HTTP API and web page.
- Accepts upload requests.
- Starts background processing threads.
- Provides endpoints for job status, logs, and reports.

Key endpoints:
- GET /
- GET /api/health
- GET /api/jobs
- POST /api/jobs
- GET /api/jobs/{job_id}
- GET /api/jobs/{job_id}/log
- GET /api/jobs/{job_id}/report
- GET /api/jobs/{job_id}/report.md
- GET /api/jobs/{job_id}/artifacts
- GET /api/jobs/{job_id}/artifacts/file/{artifact_path}
- DELETE /api/jobs/{job_id}

### 4.2 Job Orchestration Layer

File: mocap_service.py

Responsibilities:
- Maintains persistent job registry backed by `jobs/jobs_index.json`.
- Creates dataset folders from uploaded files.
- Executes EasyMocap commands and writes run logs.
- Tracks stage and progress during processing.
- Triggers report generation when processing completes.

Core objects:
- Job dataclass: Job metadata and state.
- JobStore: Thread safe job manager with JSON index persistence.

### 4.3 Analysis Layer

File: reporting.py

Responsibilities:
- Reads generated outputs and computes analysis.
- Extracts video metadata with ffprobe when available.
- Computes detection and SMPL tracking metrics.
- Writes machine readable and human readable reports.

Primary metrics:
- Video duration, resolution, fps.
- Images per view and total images.
- Annotation file count and average people per frame.
- SMPL frame count and tracked persons.
- Mean and max translation speed from Th trajectories.

### 4.4 Frontend Layer

Files:
- templates/index.html
- static/style.css

Responsibilities:
- Upload form with mode selector.
- Polling UI for runtime status and logs.
- Final report rendering and markdown download link.

## 5. Data Model and Storage

Per job directory layout:
- jobs/job_<timestamp>_<id>/videos/input.mp4
- jobs/job_<timestamp>_<id>/images/...
- jobs/job_<timestamp>_<id>/annots/...
- jobs/job_<timestamp>_<id>/output-smpl-3d/...
- jobs/job_<timestamp>_<id>/analysis/report.json
- jobs/job_<timestamp>_<id>/analysis/report.md
- jobs/job_<timestamp>_<id>/run.log

Runtime state:
- Job metadata is persisted in `jobs/jobs_index.json` via JobStore.
- Job folders and metadata survive service restarts.

## 6. Mode to Workflow Mapping

Defined in mocap_service.py:
- fast -> internet
- balanced -> internet
- accurate -> internet-rotate

Rationale:
- internet is robust default monocular pipeline.
- internet-rotate improves handling of rotational motion dynamics.

## 7. External Dependencies

Python packages:
- fastapi
- uvicorn
- python-multipart

System/runtime tools:
- ffmpeg and ffprobe recommended for extraction and metadata.
- EasyMocap environment and model assets required for full pipeline output.

## 8. Error Handling and Recovery

Current handling:
- Failed subprocess raises job failure state.
- Error message captured in job record.
- Logs preserved in run.log for diagnosis.
- Report generation is attempted only after successful run command.

Known constraints:
- Single process runtime, no distributed queue.
- Heavy jobs depend on local hardware and EasyMocap assets.
- Progress is stage based and inferred from logs, not exact per frame completion.

## 9. Security and Operational Notes

Current posture:
- Local deployment oriented.
- File upload type is extension checked.
- No authentication and no user isolation yet.

Production hardening recommendations:
- Add auth and role checks.
- Add upload size limits and MIME validation.
- Move job registry to persistent DB.
- Use process queue workers for resource isolation.

## 10. Review Checklist

For project review sessions:
- API paths return expected status codes.
- Job lifecycle transitions: queued -> running -> completed or failed.
- EasyMocap command executes from the intended EasyMocap root.
- Report metrics are present even for partial output cases.
- UI correctly displays logs and final report.
- Deleting jobs removes stored artifacts safely.
