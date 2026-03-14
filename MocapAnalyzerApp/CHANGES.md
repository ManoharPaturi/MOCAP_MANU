# Changes Log

## 2026-03-12

## 1. New Application Added

Created a new application folder:
- MocapAnalyzerApp

Goal achieved:
- Complete upload to analysis workflow integrated with EasyMocap.

## 2. Backend Implementation

Added files:
- app.py
- mocap_service.py
- reporting.py
- requirements.txt

What was implemented:
- FastAPI server and endpoints for job lifecycle.
- Video upload handling and dataset folder preparation.
- Background thread execution of EasyMocap workflows.
- Live log retrieval endpoint.
- JSON and Markdown report endpoints.
- Job cleanup endpoint.

## 3. Frontend Implementation

Added files:
- templates/index.html
- static/style.css

What was implemented:
- Upload form with mode selection.
- Polling based status updates.
- Runtime log visualization.
- Structured report rendering in browser.
- Markdown report download link.

## 4. EasyMocap Workflow Integration

Method integration decisions:
- fast and balanced modes mapped to internet workflow.
- accurate mode mapped to internet-rotate workflow.

Execution path used:
- apps/demo/mocap.py --work <workflow> <dataset>

This keeps the app aligned with EasyMocap workflow definitions rather than custom replacement logic.

## 5. Report Generation Added

Generated artifacts per job:
- analysis/report.json
- analysis/report.md

Report currently includes:
- Video metadata when ffprobe is available.
- Frame extraction coverage by camera view.
- Annotation statistics.
- SMPL frame and person tracking counts.
- Translation speed summary metrics.
- Output artifact listing.

## 6. Validation Performed

Checks completed:
- Python compile checks for app.py, mocap_service.py, reporting.py.
- HTTP smoke test of home page via local server.
- Dependency installation in active virtual environment.

## 7. Known Limitations

Current limitations:
- Job state persistence is memory only.
- No auth and multi user access control yet.
- Runtime quality and speed depend on EasyMocap assets and hardware.
- Progress percentage is inferred from status and logs, not exact stage metrics.

## 8. Completion Upgrade Applied

Additional completion changes were applied:
- Added persistent job index: `jobs/jobs_index.json`.
- Added stage and progress fields to jobs.
- Added job listing endpoint and UI history panel.
- Added artifact listing and artifact file download endpoints.
- Added upload size guard via `MOCAP_MAX_UPLOAD_MB`.
- Added health endpoint for environment checks.
- Added review checklist file: `REVIEW_GUIDE.md`.

## 9. Suggested Next Upgrades

Recommended future changes:
- Add queue workers for concurrency and stability.
- Add user authentication and role based access.
- Add optional cloud storage backend for artifact retention.
