import argparse
import json
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description="Create moving keypoints overlay video from EasyMocap annots")
    parser.add_argument("job_dir", type=Path, help="Path to job directory")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--min-conf", type=float, default=0.2)
    parser.add_argument("--output", type=Path, default=None, help="Output mp4 path")
    args = parser.parse_args()

    img_dir = args.job_dir / "images" / "input"
    ann_dir = args.job_dir / "annots" / "input"
    out_path = args.output or (args.job_dir / "analysis" / "keypoints_overlay.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    images = sorted(img_dir.glob("*.jpg"))
    if not images:
        raise SystemExit(f"No images found in {img_dir}")

    first = cv2.imread(str(images[0]))
    if first is None:
        raise SystemExit(f"Cannot read image: {images[0]}")

    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
    if not writer.isOpened():
        raise SystemExit(f"Cannot open writer for {out_path}")

    palette = [
        (255, 80, 80),
        (80, 255, 80),
        (80, 160, 255),
        (255, 200, 80),
        (180, 80, 255),
        (80, 255, 220),
        (255, 120, 200),
        (220, 220, 80),
    ]

    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        ann_path = ann_dir / f"{img_path.stem}.json"
        if ann_path.exists():
            data = json.loads(ann_path.read_text(encoding="utf-8"))
            for person in data.get("annots", []):
                pid = int(person.get("personID", 0))
                color = palette[pid % len(palette)]
                for x, y, conf in person.get("keypoints", []):
                    if conf is None or conf < args.min_conf:
                        continue
                    if x <= 0 or y <= 0:
                        continue
                    cv2.circle(frame, (int(x), int(y)), 3, color, -1, lineType=cv2.LINE_AA)
                bbox = person.get("bbox", None)
                if bbox and len(bbox) >= 4:
                    x1, y1, x2, y2 = map(int, bbox[:4])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        f"ID {pid}",
                        (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                        cv2.LINE_AA,
                    )
        writer.write(frame)

    writer.release()
    print(out_path)


if __name__ == "__main__":
    main()
