#!/usr/bin/env python3
"""V1 runner for EasyMocap with fast, balanced, and accurate presets."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_PRESETS = {
    "fast": {
        "model": "smpl",
        "body": "body25",
        "gender": "neutral",
        "thres2d": 0.4,
        "smooth3d": 0,
        "max_repro_error": 70,
        "vis_det": False,
        "vis_repro": False,
        "vis_smpl": False,
    },
    "balanced": {
        "model": "smpl",
        "body": "body25",
        "gender": "neutral",
        "thres2d": 0.3,
        "smooth3d": 3,
        "max_repro_error": 50,
        "vis_det": False,
        "vis_repro": False,
        "vis_smpl": False,
    },
    "accurate": {
        "model": "smplx",
        "body": "bodyhandface",
        "gender": "neutral",
        "thres2d": 0.2,
        "smooth3d": 7,
        "max_repro_error": 35,
        "vis_det": True,
        "vis_repro": True,
        "vis_smpl": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EasyMocap with V1 presets tuned for speed and quality."
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to dataset root (must contain intri.yml/extri.yml and videos or images).",
    )
    parser.add_argument(
        "--mode",
        choices=["fast", "balanced", "accurate"],
        default="balanced",
        help="Preset mode for runtime and quality tradeoff.",
    )
    parser.add_argument(
        "--easymocap-root",
        default=str((Path(__file__).resolve().parent.parent / "EasyMocap").resolve()),
        help="Path to EasyMocap repository root.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory. Defaults to <data>/output_v1/<mode>.",
    )
    parser.add_argument(
        "--extract-videos",
        action="store_true",
        help="Run EasyMocap video extraction before reconstruction.",
    )
    parser.add_argument(
        "--handface",
        action="store_true",
        help="Use --handface during extraction.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start frame index.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=100000,
        help="End frame index.",
    )
    parser.add_argument(
        "--sub",
        nargs="+",
        default=None,
        help="Optional camera ids to process (e.g. --sub 1 7 13).",
    )
    parser.add_argument(
        "--sub-vis",
        nargs="+",
        default=None,
        help="Optional camera ids for visualization.",
    )
    parser.add_argument(
        "--preset-file",
        default=str((Path(__file__).resolve().parent / "configs" / "presets.json").resolve()),
        help="Path to preset json file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only without running them.",
    )
    return parser.parse_args()


def load_presets(preset_file: Path) -> dict:
    if not preset_file.exists():
        return DEFAULT_PRESETS
    with preset_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    merged = DEFAULT_PRESETS.copy()
    merged.update(data)
    return merged


def validate_paths(data_path: Path, easymocap_root: Path) -> None:
    if not easymocap_root.exists():
        raise FileNotFoundError(f"EasyMocap root does not exist: {easymocap_root}")
    if not (easymocap_root / "apps" / "demo" / "mv1p.py").exists():
        raise FileNotFoundError("Could not find apps/demo/mv1p.py inside EasyMocap root")
    if not data_path.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")
    if not (data_path / "intri.yml").exists() or not (data_path / "extri.yml").exists():
        raise FileNotFoundError("Data path must contain intri.yml and extri.yml")


def build_extract_cmd(python_exe: str, easymocap_root: Path, data_path: Path, handface: bool) -> list[str]:
    cmd = [
        python_exe,
        str(easymocap_root / "scripts" / "preprocess" / "extract_video.py"),
        str(data_path),
    ]
    if handface:
        cmd.append("--handface")
    return cmd


def build_mv1p_cmd(
    python_exe: str,
    easymocap_root: Path,
    data_path: Path,
    output_path: Path,
    preset: dict,
    start: int,
    end: int,
    sub: list[str] | None,
    sub_vis: list[str] | None,
) -> list[str]:
    cmd = [
        python_exe,
        str(easymocap_root / "apps" / "demo" / "mv1p.py"),
        str(data_path),
        "--out",
        str(output_path),
        "--model",
        str(preset["model"]),
        "--body",
        str(preset["body"]),
        "--gender",
        str(preset["gender"]),
        "--thres2d",
        str(preset["thres2d"]),
        "--smooth3d",
        str(preset["smooth3d"]),
        "--MAX_REPRO_ERROR",
        str(preset["max_repro_error"]),
        "--undis",
        "--start",
        str(start),
        "--end",
        str(end),
    ]

    if preset.get("vis_det", False):
        cmd.append("--vis_det")
    if preset.get("vis_repro", False):
        cmd.append("--vis_repro")
    if preset.get("vis_smpl", False):
        cmd.append("--vis_smpl")

    if sub:
        cmd.append("--sub")
        cmd.extend(sub)
    if sub_vis:
        cmd.append("--sub_vis")
        cmd.extend(sub_vis)

    return cmd


def run_cmd(cmd: list[str], cwd: Path, dry_run: bool) -> None:
    print("$", " ".join(shlex.quote(part) for part in cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> int:
    args = parse_args()

    data_path = Path(args.data).expanduser().resolve()
    easymocap_root = Path(args.easymocap_root).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else (data_path / "output_v1" / args.mode).resolve()
    )

    presets = load_presets(Path(args.preset_file).expanduser().resolve())
    if args.mode not in presets:
        raise KeyError(f"Mode '{args.mode}' is not defined in preset file")

    validate_paths(data_path, easymocap_root)
    output_path.mkdir(parents=True, exist_ok=True)

    preset = presets[args.mode]
    python_exe = sys.executable

    if args.extract_videos:
        extract_cmd = build_extract_cmd(python_exe, easymocap_root, data_path, args.handface)
        run_cmd(extract_cmd, cwd=easymocap_root, dry_run=args.dry_run)

    mv1p_cmd = build_mv1p_cmd(
        python_exe=python_exe,
        easymocap_root=easymocap_root,
        data_path=data_path,
        output_path=output_path,
        preset=preset,
        start=args.start,
        end=args.end,
        sub=args.sub,
        sub_vis=args.sub_vis,
    )
    run_cmd(mv1p_cmd, cwd=easymocap_root, dry_run=args.dry_run)

    print(f"Completed mode '{args.mode}'. Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
