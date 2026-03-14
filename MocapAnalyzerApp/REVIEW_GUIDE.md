# Review Guide

## Quick QA in 10 Minutes

## 1. Start application

```bash
source /Users/manoharpaturi/manu/bin/activate
cd /Users/manoharpaturi/Desktop/mocap_manu/MocapAnalyzerApp
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

Open http://localhost:8080

## 2. Check health API

```bash
curl -s http://127.0.0.1:8080/api/health
```

Expected:
- `ok: true`
- `easy_mocap_exists: true`

## 3. Submit job

- Upload a short test video from UI.
- Select mode `balanced`.
- Click Start Analysis.

Expected:
- Job appears under Job History.
- Status panel shows stage and progress updates.
- Logs stream in real time.

## 4. Check job APIs

Replace `<job_id>` from UI result:

```bash
curl -s http://127.0.0.1:8080/api/jobs/<job_id>
curl -s http://127.0.0.1:8080/api/jobs/<job_id>/log
```

Expected:
- status transitions queued -> running -> completed or failed.
- progress and stage fields update while running.

## 5. Validate report output

After completion:

```bash
curl -s http://127.0.0.1:8080/api/jobs/<job_id>/report
```

Expected sections:
- video
- methodology
- frames
- detections
- smpl
- artifacts

Also verify markdown download:

```bash
curl -I http://127.0.0.1:8080/api/jobs/<job_id>/report.md
```

## 6. Validate artifacts endpoint

```bash
curl -s http://127.0.0.1:8080/api/jobs/<job_id>/artifacts
```

Expected:
- list contains report files and any generated videos/json files.

Download one artifact:

```bash
curl -O "http://127.0.0.1:8080/api/jobs/<job_id>/artifacts/file/analysis/report.json"
```

## 7. Validate delete cleanup

```bash
curl -X DELETE -s http://127.0.0.1:8080/api/jobs/<job_id>
```

Expected:
- Job removed from history.
- Job folder removed from `jobs/`.

## Troubleshooting

- If EasyMocap processing fails quickly, check `run.log` first.
- Ensure EasyMocap dependencies and models are installed.
- For large videos, increase `MOCAP_MAX_UPLOAD_MB` before launching app.
