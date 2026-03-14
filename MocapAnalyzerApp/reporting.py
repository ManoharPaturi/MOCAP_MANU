from __future__ import annotations

import json
import math
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any


def _safe_read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _ffprobe_video(input_video: Path) -> dict[str, Any]:
    if not input_video.exists():
        return {"exists": False}
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(input_video),
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        info = json.loads(result.stdout)
    except Exception:
        return {"exists": True, "ffprobe": "unavailable"}

    duration = None
    width = None
    height = None
    fps = None

    streams = info.get("streams", [])
    for stream in streams:
        if stream.get("codec_type") != "video":
            continue
        width = stream.get("width")
        height = stream.get("height")
        fr = stream.get("avg_frame_rate") or "0/1"
        try:
            num, den = fr.split("/")
            if float(den) != 0.0:
                fps = float(num) / float(den)
        except Exception:
            fps = None
        break

    try:
        duration = float(info.get("format", {}).get("duration", "0"))
    except Exception:
        duration = None

    return {
        "exists": True,
        "duration_sec": duration,
        "width": width,
        "height": height,
        "fps": fps,
    }


def _count_images(dataset_dir: Path) -> dict[str, int]:
    images_root = dataset_dir / "images"
    if not images_root.exists():
        return {}
    counts: dict[str, int] = {}
    for sub in sorted(images_root.iterdir()):
        if not sub.is_dir():
            continue
        counts[sub.name] = len(list(sub.glob("*.jpg")))
    return counts


def _collect_annotation_stats(dataset_dir: Path) -> dict[str, Any]:
    ann_root = dataset_dir / "annots"
    if not ann_root.exists():
        return {"annotation_files": 0, "avg_people_per_frame": 0.0}

    ann_files = sorted(ann_root.rglob("*.json"))
    people_counts = []
    for ann_file in ann_files[:3000]:
        data = _safe_read_json(ann_file)
        if isinstance(data, dict):
            people_counts.append(len(data.get("annots", [])))

    avg_people = float(statistics.mean(people_counts)) if people_counts else 0.0
    return {
        "annotation_files": len(ann_files),
        "avg_people_per_frame": round(avg_people, 3),
    }


def _extract_frame_index(path: Path) -> int | None:
    stem = path.stem
    if stem.isdigit():
        return int(stem)
    return None


def _collect_smpl_stats(output_dir: Path, fps_hint: float | None) -> dict[str, Any]:
    smpl_root = output_dir / "smpl"
    if not smpl_root.exists():
        return {
            "smpl_frames": 0,
            "persons_detected": 0,
            "mean_translation_speed": None,
            "max_translation_speed": None,
        }

    smpl_files = sorted(smpl_root.rglob("*.json"))
    person_tracks: dict[int, list[tuple[int, list[float]]]] = defaultdict(list)

    for smpl_file in smpl_files:
        frame_idx = _extract_frame_index(smpl_file)
        if frame_idx is None:
            continue
        data = _safe_read_json(smpl_file)
        if isinstance(data, dict):
            people = data.get("annots", [])
            if not isinstance(people, list):
                people = []
        elif isinstance(data, list):
            people = data
        else:
            people = []

        for person in people:
            if not isinstance(person, dict):
                continue
            pid = int(person.get("id", 0))
            th = person.get("Th")
            if isinstance(th, list) and len(th) > 0:
                if isinstance(th[0], list):
                    if len(th[0]) < 3:
                        continue
                    th = th[0]
                if len(th) < 3:
                    continue
                try:
                    vec = [float(th[0]), float(th[1]), float(th[2])]
                    person_tracks[pid].append((frame_idx, vec))
                except Exception:
                    continue

    fps = fps_hint or 30.0
    speeds = []
    for _, track in person_tracks.items():
        track = sorted(track, key=lambda x: x[0])
        for i in range(1, len(track)):
            f0, p0 = track[i - 1]
            f1, p1 = track[i]
            dt_frame = max(1, f1 - f0)
            dt = dt_frame / fps
            dist = math.sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2 + (p1[2] - p0[2]) ** 2)
            speeds.append(dist / dt)

    mean_speed = float(statistics.mean(speeds)) if speeds else None
    max_speed = float(max(speeds)) if speeds else None

    return {
        "smpl_frames": len(smpl_files),
        "persons_detected": len(person_tracks),
        "mean_translation_speed": round(mean_speed, 6) if mean_speed is not None else None,
        "max_translation_speed": round(max_speed, 6) if max_speed is not None else None,
    }


def _collect_artifacts(dataset_dir: Path, output_dir: Path) -> list[str]:
    artifacts = []
    for pattern in ["*.mp4", "smplmesh/**/*.mp4", "smplmesh/*.mp4"]:
        for p in output_dir.glob(pattern):
            if p.is_file():
                artifacts.append(str(p.relative_to(dataset_dir)))
    artifacts = sorted(set(artifacts))
    return artifacts


def generate_report(dataset_dir: Path, output_dir: Path, input_video: Path, workflow: str) -> dict[str, Any]:
    video_info = _ffprobe_video(input_video)
    image_counts = _count_images(dataset_dir)
    annotation_stats = _collect_annotation_stats(dataset_dir)
    smpl_stats = _collect_smpl_stats(output_dir, video_info.get("fps"))
    artifacts = _collect_artifacts(dataset_dir, output_dir)

    methodology = {
        "workflow": workflow,
        "stages": [
            "Frame extraction from uploaded video into dataset/images/<camera_id>",
            "2D keypoint detection (default yolo-hrnet in EasyMocap internet workflow)",
            "Monocular SMPL fitting with robust optimization (mono-smpl-robust)",
            "Rendering and export of mesh visualization videos",
        ],
    }

    return {
        "video": video_info,
        "methodology": methodology,
        "frames": {
            "images_per_view": image_counts,
            "total_images": int(sum(image_counts.values())),
        },
        "detections": annotation_stats,
        "smpl": smpl_stats,
        "artifacts": artifacts,
    }


def write_reports(dataset_dir: Path, report: dict[str, Any]) -> tuple[Path, Path]:
    analysis_dir = dataset_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    json_path = analysis_dir / "report.json"
    md_path = analysis_dir / "report.md"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Motion Analysis Report",
        "",
        "## Workflow",
        f"- Workflow: {report['methodology']['workflow']}",
        "",
        "## Video",
        f"- Duration (s): {report['video'].get('duration_sec')}",
        f"- Resolution: {report['video'].get('width')} x {report['video'].get('height')}",
        f"- FPS: {report['video'].get('fps')}",
        "",
        "## Frames",
        f"- Total images: {report['frames']['total_images']}",
        "",
        "## Detections",
        f"- Annotation files: {report['detections']['annotation_files']}",
        f"- Avg people per frame: {report['detections']['avg_people_per_frame']}",
        "",
        "## SMPL",
        f"- SMPL frames: {report['smpl']['smpl_frames']}",
        f"- Persons tracked: {report['smpl']['persons_detected']}",
        f"- Mean translation speed: {report['smpl']['mean_translation_speed']}",
        f"- Max translation speed: {report['smpl']['max_translation_speed']}",
        "",
        "## Artifacts",
    ]

    for artifact in report.get("artifacts", []):
        lines.append(f"- {artifact}")

    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path
