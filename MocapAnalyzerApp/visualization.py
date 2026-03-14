from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import cv2


# Body-25 style skeleton pairs; extra indices are safely ignored if absent.
BODY25_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (10, 11),
    (8, 12), (12, 13), (13, 14),
    (0, 15), (15, 17), (0, 16), (16, 18),
    (14, 19), (19, 20), (14, 21),
    (11, 22), (22, 23), (11, 24),
]

# COCO-17 edges (common detector output).
COCO17_EDGES = [
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

# MediaPipe pose-33 edges.
MEDIAPIPE33_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32),
]

MEDIAPIPE_HAND21_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def _choose_edges(num_keypoints: int) -> list[tuple[int, int]]:
    if num_keypoints >= 33:
        return MEDIAPIPE33_EDGES
    if num_keypoints >= 25:
        return BODY25_EDGES
    return COCO17_EDGES


def _extract_keypoints(person: dict[str, Any], w: int, h: int) -> list[list[float]]:
    # EasyMocap/OpenPose style: keypoints = [[x,y,c], ...]
    if isinstance(person.get("keypoints"), list):
        kpts = person["keypoints"]
        if kpts and isinstance(kpts[0], list):
            return kpts

    # MediaPipe style alternatives used in some custom repos.
    alt_keys = ["pose_landmarks", "landmarks", "mp_pose", "pose_keypoints"]
    for key in alt_keys:
        val = person.get(key)
        if not isinstance(val, list) or not val:
            continue
        if not isinstance(val[0], dict):
            continue
        out = []
        for lm in val:
            x = lm.get("x", 0.0)
            y = lm.get("y", 0.0)
            c = lm.get("visibility", lm.get("score", 1.0))
            try:
                x = float(x)
                y = float(y)
                c = float(c)
            except Exception:
                continue
            # Convert normalized coordinates to pixels when needed.
            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                x *= w
                y *= h
            out.append([x, y, c])
        if out:
            return out

    return []


def _safe_read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _extract_frame_idx(path: Path) -> int | None:
    if path.stem.isdigit():
        return int(path.stem)
    return None


def _load_people_and_conf(dataset_dir: Path) -> tuple[list[int], list[float], list[float]]:
    ann_root = dataset_dir / "annots"
    ann_files = sorted(ann_root.rglob("*.json"))
    frame_idx: list[int] = []
    people_count: list[float] = []
    mean_conf: list[float] = []

    for ann_file in ann_files:
        idx = _extract_frame_idx(ann_file)
        if idx is None:
            continue
        data = _safe_read_json(ann_file)
        if not isinstance(data, dict):
            continue
        annots = data.get("annots", [])
        if not isinstance(annots, list):
            annots = []

        conf_vals = []
        for person in annots:
            if not isinstance(person, dict):
                continue
            for kp in person.get("keypoints", []):
                if not isinstance(kp, list) or len(kp) < 3:
                    continue
                c = kp[2]
                try:
                    c = float(c)
                except Exception:
                    continue
                if c > 0:
                    conf_vals.append(c)

        frame_idx.append(idx)
        people_count.append(float(len(annots)))
        mean_conf.append(sum(conf_vals) / len(conf_vals) if conf_vals else 0.0)

    return frame_idx, people_count, mean_conf


def _load_main_track(output_dir: Path) -> tuple[list[int], list[list[float]]]:
    smpl_root = output_dir / "smpl"
    files = sorted(smpl_root.rglob("*.json"))
    tracks: dict[int, list[tuple[int, list[float]]]] = {}

    for fp in files:
        idx = _extract_frame_idx(fp)
        if idx is None:
            continue
        data = _safe_read_json(fp)
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
            th = person.get("Th", None)
            if isinstance(th, list) and len(th) > 0:
                if isinstance(th[0], list):
                    th = th[0]
                if len(th) < 3:
                    continue
                try:
                    vec = [float(th[0]), float(th[1]), float(th[2])]
                except Exception:
                    continue
                tracks.setdefault(pid, []).append((idx, vec))

    if not tracks:
        return [], []

    best_pid = max(tracks, key=lambda k: len(tracks[k]))
    seq = sorted(tracks[best_pid], key=lambda x: x[0])
    frames = [x[0] for x in seq]
    vals = [x[1] for x in seq]
    return frames, vals


def _scale(val: float, lo: float, hi: float, out_lo: int, out_hi: int) -> int:
    if hi <= lo:
        return (out_lo + out_hi) // 2
    t = (val - lo) / (hi - lo)
    t = max(0.0, min(1.0, t))
    return int(out_lo + t * (out_hi - out_lo))


def _draw_axes(img, x0: int, y0: int, w: int, h: int, title: str) -> None:
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (70, 70, 70), 1)
    cv2.putText(img, title, (x0 + 8, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2, cv2.LINE_AA)


def generate_motion_plot_video(dataset_dir: Path, output_dir: Path, fps: float = 30.0) -> Path:
    out_path = dataset_dir / "analysis" / "motion_plots.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    f_ann, people_count, mean_conf = _load_people_and_conf(dataset_dir)
    f_trk, track = _load_main_track(output_dir)

    max_frame = 0
    if f_ann:
        max_frame = max(max_frame, max(f_ann))
    if f_trk:
        max_frame = max(max_frame, max(f_trk))
    if max_frame <= 0:
        max_frame = max(len(f_ann), 1) - 1

    people_by_frame = {f: v for f, v in zip(f_ann, people_count)}
    conf_by_frame = {f: v for f, v in zip(f_ann, mean_conf)}
    trk_by_frame = {f: v for f, v in zip(f_trk, track)}

    speeds_by_frame: dict[int, float] = {}
    if f_trk and track:
        for i in range(1, len(f_trk)):
            f0, f1 = f_trk[i - 1], f_trk[i]
            p0, p1 = track[i - 1], track[i]
            dt = max(1, f1 - f0) / max(1e-6, fps)
            dist = math.sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2 + (p1[2] - p0[2]) ** 2)
            speeds_by_frame[f1] = dist / dt

    xs = [v[0] for v in track] if track else [0.0]
    zs = [v[2] for v in track] if track else [0.0]
    speed_vals = list(speeds_by_frame.values()) if speeds_by_frame else [0.0]

    x_min, x_max = min(xs), max(xs)
    z_min, z_max = min(zs), max(zs)
    p_min, p_max = 0.0, max(1.0, max(people_count) if people_count else 1.0)
    c_min, c_max = 0.0, 1.0
    s_min, s_max = 0.0, max(1.0, max(speed_vals))

    w, h = 1280, 720
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video: {out_path}")

    left_x, top_y = 30, 40
    pad = 20
    pw = (w - left_x * 2 - pad) // 2
    ph = (h - top_y * 2 - pad) // 2

    traj_pts: list[tuple[int, int]] = []
    speed_series: list[float] = []
    people_series: list[float] = []
    conf_series: list[float] = []

    for f in range(max_frame + 1):
        canvas = cv2.cvtColor(cv2.UMat(h, w, cv2.CV_8UC1, 18).get(), cv2.COLOR_GRAY2BGR)

        _draw_axes(canvas, left_x, top_y, pw, ph, "SMPL trajectory (X vs Z)")
        _draw_axes(canvas, left_x + pw + pad, top_y, pw, ph, "Translation speed")
        _draw_axes(canvas, left_x, top_y + ph + pad, pw, ph, "People per frame")
        _draw_axes(canvas, left_x + pw + pad, top_y + ph + pad, pw, ph, "Mean keypoint confidence")

        if f in trk_by_frame:
            x, _, z = trk_by_frame[f]
            px = _scale(x, x_min, x_max, left_x + 40, left_x + pw - 20)
            py = _scale(z, z_min, z_max, top_y + ph - 25, top_y + 40)
            traj_pts.append((px, py))

        if len(traj_pts) >= 2:
            for i in range(1, len(traj_pts)):
                cv2.line(canvas, traj_pts[i - 1], traj_pts[i], (90, 220, 255), 2, cv2.LINE_AA)
        if traj_pts:
            cv2.circle(canvas, traj_pts[-1], 5, (255, 255, 255), -1, cv2.LINE_AA)

        speed_series.append(float(speeds_by_frame.get(f, speed_series[-1] if speed_series else 0.0)))
        people_series.append(float(people_by_frame.get(f, people_series[-1] if people_series else 0.0)))
        conf_series.append(float(conf_by_frame.get(f, conf_series[-1] if conf_series else 0.0)))

        def draw_series(series: list[float], x0: int, y0: int, ww: int, hh: int, lo: float, hi: float, color: tuple[int, int, int]) -> None:
            if not series:
                return
            n = max(1, len(series) - 1)
            prev = None
            for i, v in enumerate(series):
                px = x0 + 30 + int((ww - 50) * (i / n))
                py = _scale(v, lo, hi, y0 + hh - 20, y0 + 35)
                if prev is not None:
                    cv2.line(canvas, prev, (px, py), color, 2, cv2.LINE_AA)
                prev = (px, py)
            cv2.circle(canvas, prev, 4, (255, 255, 255), -1, cv2.LINE_AA)

        draw_series(speed_series, left_x + pw + pad, top_y, pw, ph, s_min, s_max, (80, 180, 255))
        draw_series(people_series, left_x, top_y + ph + pad, pw, ph, p_min, p_max, (120, 255, 120))
        draw_series(conf_series, left_x + pw + pad, top_y + ph + pad, pw, ph, c_min, c_max, (255, 170, 90))

        cv2.putText(canvas, f"Frame: {f:06d}", (w - 220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA)
        writer.write(canvas)

    writer.release()
    return out_path


def generate_stickman_video(dataset_dir: Path, fps: float = 30.0, min_conf: float = 0.2) -> Path:
    img_root = dataset_dir / "images"
    ann_root = dataset_dir / "annots"
    out_path = dataset_dir / "analysis" / "stickman_animation.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not img_root.exists() or not ann_root.exists():
        raise RuntimeError("Missing images or annots folder for stickman generation")

    subdirs = sorted([p.name for p in img_root.iterdir() if p.is_dir()])
    if not subdirs:
        raise RuntimeError("No image subfolders found")
    sub = subdirs[0]

    images = sorted((img_root / sub).glob("*.jpg"))
    if not images:
        raise RuntimeError("No images found for stickman generation")

    first = cv2.imread(str(images[0]))
    if first is None:
        raise RuntimeError("Failed to read first image")
    h, w = first.shape[:2]

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video: {out_path}")

    palette = [
        (255, 110, 110),
        (110, 255, 120),
        (120, 180, 255),
        (255, 220, 110),
        (220, 120, 255),
        (100, 255, 230),
        (255, 140, 210),
    ]

    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        ann_path = ann_root / sub / f"{img_path.stem}.json"
        data = _safe_read_json(ann_path) if ann_path.exists() else None
        people = data.get("annots", []) if isinstance(data, dict) else []

        for person in people:
            if not isinstance(person, dict):
                continue
            pid = int(person.get("personID", 0))
            color = palette[pid % len(palette)]
            kpts = _extract_keypoints(person, w=w, h=h)
            if not kpts:
                continue
            edges = _choose_edges(len(kpts))

            for i0, i1 in edges:
                if i0 >= len(kpts) or i1 >= len(kpts):
                    continue
                p0 = kpts[i0]
                p1 = kpts[i1]
                if not isinstance(p0, list) or not isinstance(p1, list):
                    continue
                if len(p0) < 3 or len(p1) < 3:
                    continue
                x0, y0, c0 = p0[0], p0[1], p0[2]
                x1, y1, c1 = p1[0], p1[1], p1[2]
                try:
                    x0, y0, c0 = float(x0), float(y0), float(c0)
                    x1, y1, c1 = float(x1), float(y1), float(c1)
                except Exception:
                    continue
                if c0 < min_conf or c1 < min_conf:
                    continue
                if x0 <= 0 or y0 <= 0 or x1 <= 0 or y1 <= 0:
                    continue
                cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), color, 3, cv2.LINE_AA)

            for kp in kpts:
                if not isinstance(kp, list) or len(kp) < 3:
                    continue
                x, y, c = kp[0], kp[1], kp[2]
                try:
                    x, y, c = float(x), float(y), float(c)
                except Exception:
                    continue
                if c < min_conf or x <= 0 or y <= 0:
                    continue
                cv2.circle(frame, (int(x), int(y)), 4, (245, 245, 245), -1, cv2.LINE_AA)

        writer.write(frame)

    writer.release()
    return out_path


def _to_xyv(lm: dict[str, Any], w: int, h: int) -> tuple[float, float, float] | None:
    if not isinstance(lm, dict):
        return None
    x = lm.get("x", None)
    y = lm.get("y", None)
    v = lm.get("v", lm.get("visibility", lm.get("score", 1.0)))
    if x is None or y is None:
        return None
    try:
        x = float(x)
        y = float(y)
        v = float(v)
    except Exception:
        return None
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        x *= w
        y *= h
    return x, y, v


def generate_stickman_from_motion_capture_db(
    motion_capture_dir: Path,
    output_path: Path | None = None,
    table_name: str | None = None,
    fps: float = 30.0,
    width: int = 1280,
    height: int = 720,
    min_conf: float = 0.2,
    draw_face: bool = False,
    draw_hands: bool = True,
) -> Path:
    db_path = motion_capture_dir / "mocap_data.db"
    if not db_path.exists():
        raise RuntimeError(f"Database not found: {db_path}")

    out_path = output_path or (motion_capture_dir / "results" / "stickman_motion_capture.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(str(db_path))
    cur = con.cursor()

    def _safe_table_name(name: str) -> str:
        # Table names come from user/session metadata; keep them SQLite-identifier safe.
        if not name:
            raise RuntimeError("Empty table name")
        if not all(ch.isalnum() or ch == "_" for ch in name):
            raise RuntimeError(f"Invalid table name: {name}")
        return name

    def _parse_json_array(payload: Any) -> list[Any]:
        if not isinstance(payload, str) or not payload:
            return []
        try:
            data = json.loads(payload)
        except Exception:
            return []
        return data if isinstance(data, list) else []

    if table_name is None:
        sessions = cur.execute("select table_name from sessions order by start_time desc").fetchall()
        chosen = None
        for (tname,) in sessions:
            safe_tname = _safe_table_name(str(tname))
            count = cur.execute(f"select count(*) from {safe_tname}").fetchone()[0]
            if count > 0:
                chosen = safe_tname
                break
        if chosen is None:
            con.close()
            raise RuntimeError("No recorded frames found in mocap_data.db sessions")
        table_name = chosen
    else:
        table_name = _safe_table_name(table_name)

    cols = {
        row[1]
        for row in cur.execute(f"PRAGMA table_info({table_name})").fetchall()
        if len(row) > 1
    }
    required = {"timestamp", "pose_data", "face_data", "hand_data"}
    missing = required - cols
    if missing:
        con.close()
        raise RuntimeError(f"Missing expected columns in {table_name}: {sorted(missing)}")

    select_cols = ["timestamp", "pose_data", "face_data", "hand_data"]
    has_pc2_pose = "pc2_pose_data" in cols
    has_pc2_face = "pc2_face_data" in cols
    has_pc2_hands = "pc2_hand_data" in cols
    if has_pc2_pose:
        select_cols.append("pc2_pose_data")
    if has_pc2_face:
        select_cols.append("pc2_face_data")
    if has_pc2_hands:
        select_cols.append("pc2_hand_data")

    rows = cur.execute(f"select {', '.join(select_cols)} from {table_name} order by timestamp").fetchall()
    con.close()
    if not rows:
        raise RuntimeError(f"No rows in session table: {table_name}")

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video: {out_path}")

    colors = [
        (255, 120, 120), (120, 255, 140), (120, 190, 255),
        (255, 220, 120), (220, 140, 255), (120, 255, 230),
    ]

    def _iter_hand_sets(hand_payload: Any) -> list[list[dict[str, Any]]]:
        out: list[list[dict[str, Any]]] = []
        if not isinstance(hand_payload, list):
            return out

        # Case 1: [ [lm...21], [lm...21], ... ]
        if hand_payload and isinstance(hand_payload[0], list) and hand_payload[0] and isinstance(hand_payload[0][0], dict):
            # Could be per-person [[handA, handB], ...] or flat [handA, handB]
            if len(hand_payload[0]) == 21 and isinstance(hand_payload[0][0], dict):
                out.extend(hand_payload)
            else:
                for item in hand_payload:
                    if isinstance(item, list) and item:
                        if len(item) == 21 and isinstance(item[0], dict):
                            out.append(item)
                        elif isinstance(item[0], list):
                            for sub in item:
                                if isinstance(sub, list) and len(sub) == 21 and sub and isinstance(sub[0], dict):
                                    out.append(sub)
            return out

        # Case 2: flattened 42 landmarks in one list
        if hand_payload and isinstance(hand_payload[0], dict) and len(hand_payload) >= 21:
            if len(hand_payload) >= 42:
                out.append(hand_payload[:21])
                out.append(hand_payload[21:42])
            else:
                out.append(hand_payload[:21])
        return out

    def _iter_face_sets(face_payload: Any) -> list[list[dict[str, Any]]]:
        out: list[list[dict[str, Any]]] = []
        if not isinstance(face_payload, list) or not face_payload:
            return out
        # Expected: [ [468 dicts], ... ]
        if isinstance(face_payload[0], list):
            for item in face_payload:
                if isinstance(item, list) and item and isinstance(item[0], dict):
                    out.append(item)
            return out
        # Fallback: single flat list of dicts
        if isinstance(face_payload[0], dict):
            out.append(face_payload)
        return out

    for row in rows:
        pose_json = row[1]
        face_json = row[2]
        hand_json = row[3]
        idx = 4
        pc2_pose_json = row[idx] if has_pc2_pose else None
        idx += 1 if has_pc2_pose else 0
        pc2_face_json = row[idx] if has_pc2_face else None
        idx += 1 if has_pc2_face else 0
        pc2_hand_json = row[idx] if has_pc2_hands else None

        # Node-only rendering on blank canvas (not avatar/mesh overlay).
        frame = cv2.cvtColor(cv2.UMat(height, width, cv2.CV_8UC1, 8).get(), cv2.COLOR_GRAY2BGR)

        pose_people = _parse_json_array(pose_json)
        face_people = _parse_json_array(face_json)
        hand_people = _parse_json_array(hand_json)
        pose_people.extend(_parse_json_array(pc2_pose_json))
        face_people.extend(_parse_json_array(pc2_face_json))
        hand_people.extend(_parse_json_array(pc2_hand_json))

        node_count_pose = 0
        node_count_face = 0
        node_count_hands = 0

        for pid, pose in enumerate(pose_people):
            if not isinstance(pose, list):
                continue
            color = colors[pid % len(colors)]

            for i0, i1 in MEDIAPIPE33_EDGES:
                if i0 >= len(pose) or i1 >= len(pose):
                    continue
                p0 = _to_xyv(pose[i0], width, height)
                p1 = _to_xyv(pose[i1], width, height)
                if p0 is None or p1 is None:
                    continue
                x0, y0, c0 = p0
                x1, y1, c1 = p1
                if c0 < min_conf or c1 < min_conf:
                    continue
                cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), color, 3, cv2.LINE_AA)

            for lm in pose:
                p = _to_xyv(lm, width, height)
                if p is None:
                    continue
                x, y, c = p
                if c < min_conf:
                    continue
                cv2.circle(frame, (int(x), int(y)), 3, (245, 245, 245), -1, cv2.LINE_AA)
                node_count_pose += 1

        if draw_hands:
            for hand in _iter_hand_sets(hand_people):
                for i0, i1 in MEDIAPIPE_HAND21_EDGES:
                    if i0 >= len(hand) or i1 >= len(hand):
                        continue
                    p0 = _to_xyv(hand[i0], width, height)
                    p1 = _to_xyv(hand[i1], width, height)
                    if p0 is None or p1 is None:
                        continue
                    x0, y0, c0 = p0
                    x1, y1, c1 = p1
                    if c0 < min_conf or c1 < min_conf:
                        continue
                    cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), (170, 255, 255), 2, cv2.LINE_AA)
                for lm in hand:
                    p = _to_xyv(lm, width, height)
                    if p is None:
                        continue
                    x, y, c = p
                    if c < min_conf:
                        continue
                    cv2.circle(frame, (int(x), int(y)), 2, (220, 255, 255), -1, cv2.LINE_AA)
                    node_count_hands += 1

        if draw_face:
            for face in _iter_face_sets(face_people):
                for lm in face:
                    p = _to_xyv(lm, width, height)
                    if p is None:
                        continue
                    x, y, c = p
                    if c < min_conf:
                        continue
                    cv2.circle(frame, (int(x), int(y)), 1, (120, 220, 255), -1, cv2.LINE_AA)
                    node_count_face += 1

        cv2.putText(
            frame,
            f"pose:{node_count_pose} face:{node_count_face} hands:{node_count_hands}",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (230, 230, 230),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)

    writer.release()
    return out_path


def generate_full_nodes_analysis_video(
    input_video: Path,
    output_path: Path,
    min_conf: float = 0.2,
    slow_factor: float = 1.0,
    pose_model_variant: str = "lite",
) -> Path:
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except Exception as exc:
        raise RuntimeError(
            "mediapipe is required for full-node analysis video. Install it with: pip install mediapipe"
        ) from exc

    if not input_video.exists():
        raise RuntimeError(f"Input video not found: {input_video}")

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    if slow_factor <= 0:
        slow_factor = 1.0
    out_fps = max(1.0, fps * slow_factor)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Could not infer video width/height")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to create video: {output_path}")

    mrudula_models = Path("/Users/manoharpaturi/Desktop/Mrudula/Motion-capture/models")
    model_map = {
        "lite": "pose_landmarker_lite.task",
        "full": "pose_landmarker_full.task",
        "heavy": "pose_landmarker_heavy.task",
    }
    variant = (pose_model_variant or "lite").strip().lower()
    if variant not in model_map:
        raise RuntimeError(f"Unknown pose model variant: {pose_model_variant}. Use one of: lite, full, heavy")

    pose_model = mrudula_models / model_map[variant]
    face_model = mrudula_models / "face_landmarker.task"
    hand_model = mrudula_models / "hand_landmarker.task"
    for model_path in (pose_model, face_model, hand_model):
        if not model_path.exists():
            cap.release()
            writer.release()
            raise RuntimeError(f"Required Mrudula model not found: {model_path}")

    pose_options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(pose_model)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=2,
        min_pose_detection_confidence=min_conf,
        min_pose_presence_confidence=min_conf,
        min_tracking_confidence=min_conf,
    )
    face_options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(face_model)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=2,
        min_face_detection_confidence=min_conf,
        min_face_presence_confidence=min_conf,
        min_tracking_confidence=min_conf,
    )
    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(hand_model)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=4,
        min_hand_detection_confidence=min_conf,
        min_hand_presence_confidence=min_conf,
        min_tracking_confidence=min_conf,
    )

    total_frames = 0
    with vision.PoseLandmarker.create_from_options(pose_options) as pose_detector, vision.FaceLandmarker.create_from_options(face_options) as face_detector, vision.HandLandmarker.create_from_options(hand_options) as hand_detector:
        def _draw_face_mesh(frame_img, face_pts: list[tuple[int, int]]) -> None:
            # Use Delaunay triangulation to render dense face mesh even when static FACEMESH tables are unavailable.
            if len(face_pts) < 3:
                return
            rect = (0, 0, width, height)
            subdiv = cv2.Subdiv2D(rect)
            seen: set[tuple[int, int]] = set()
            for x, y in face_pts:
                if x < 0 or y < 0 or x >= width or y >= height:
                    continue
                pt = (int(x), int(y))
                if pt in seen:
                    continue
                seen.add(pt)
                try:
                    subdiv.insert(pt)
                except Exception:
                    continue

            triangles = subdiv.getTriangleList()
            for t in triangles:
                x1, y1, x2, y2, x3, y3 = t
                p1 = (int(x1), int(y1))
                p2 = (int(x2), int(y2))
                p3 = (int(x3), int(y3))
                if p1 not in seen or p2 not in seen or p3 not in seen:
                    continue
                cv2.line(frame_img, p1, p2, (90, 170, 220), 1, cv2.LINE_AA)
                cv2.line(frame_img, p2, p3, (90, 170, 220), 1, cv2.LINE_AA)
                cv2.line(frame_img, p3, p1, (90, 170, 220), 1, cv2.LINE_AA)

        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int((frame_idx / max(1e-6, fps)) * 1000.0)

            pose_result = pose_detector.detect_for_video(mp_image, ts_ms)
            face_result = face_detector.detect_for_video(mp_image, ts_ms)
            hand_result = hand_detector.detect_for_video(mp_image, ts_ms)

            pose_count = 0
            face_count = 0
            hand_count = 0

            pose_people = pose_result.pose_landmarks if pose_result and pose_result.pose_landmarks else []
            for pid, pose_lms in enumerate(pose_people):
                color = [(255, 120, 120), (120, 255, 140), (120, 190, 255)][pid % 3]
                pose_list = [
                    {"x": lm.x * width, "y": lm.y * height, "v": getattr(lm, "visibility", 1.0)}
                    for lm in pose_lms
                ]
                for i0, i1 in MEDIAPIPE33_EDGES:
                    if i0 >= len(pose_list) or i1 >= len(pose_list):
                        continue
                    p0 = pose_list[i0]
                    p1 = pose_list[i1]
                    if float(p0["v"]) < min_conf or float(p1["v"]) < min_conf:
                        continue
                    cv2.line(frame, (int(p0["x"]), int(p0["y"])), (int(p1["x"]), int(p1["y"])), color, 2, cv2.LINE_AA)
                for p in pose_list:
                    if float(p["v"]) < min_conf:
                        continue
                    cv2.circle(frame, (int(p["x"]), int(p["y"])), 2, (250, 250, 250), -1, cv2.LINE_AA)
                    pose_count += 1

            face_people = face_result.face_landmarks if face_result and face_result.face_landmarks else []
            for face_lms in face_people:
                face_pts: list[tuple[int, int]] = []
                for lm in face_lms:
                    x = int(lm.x * width)
                    y = int(lm.y * height)
                    if x < 0 or y < 0 or x >= width or y >= height:
                        continue
                    face_pts.append((x, y))
                    cv2.circle(frame, (x, y), 1, (120, 220, 255), -1, cv2.LINE_AA)
                    face_count += 1
                _draw_face_mesh(frame, face_pts)

            hand_sets = hand_result.hand_landmarks if hand_result and hand_result.hand_landmarks else []
            for hand_lms in hand_sets:
                hand_list = [{"x": lm.x * width, "y": lm.y * height, "v": 1.0} for lm in hand_lms]
                for i0, i1 in MEDIAPIPE_HAND21_EDGES:
                    if i0 >= len(hand_list) or i1 >= len(hand_list):
                        continue
                    p0 = hand_list[i0]
                    p1 = hand_list[i1]
                    cv2.line(frame, (int(p0["x"]), int(p0["y"])), (int(p1["x"]), int(p1["y"])), (170, 255, 255), 2, cv2.LINE_AA)
                for p in hand_list:
                    cv2.circle(frame, (int(p["x"]), int(p["y"])), 2, (220, 255, 255), -1, cv2.LINE_AA)
                    hand_count += 1

            total_nodes = pose_count + face_count + hand_count
            cv2.putText(
                frame,
                f"nodes:{total_nodes} pose:{pose_count} face:{face_count} hands:{hand_count} speed:x{slow_factor:.2f}",
                (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            writer.write(frame)
            total_frames += 1
            frame_idx += 1

    cap.release()
    writer.release()

    if total_frames == 0:
        raise RuntimeError("No frames decoded from input video")

    return output_path
