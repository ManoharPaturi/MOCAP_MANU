# V1 EasyMocap Runner

This folder contains a V1 application wrapper around EasyMocap to run motion capture with practical presets:

- `fast`: lower runtime, lower visualization overhead.
- `balanced`: default tradeoff.
- `accurate`: stricter thresholds and richer model/visualization.

## Folder

```text
V1/
├── run_v1.py
├── configs/
│   └── presets.json
└── README.md
```

## Quick Start

From `mocap_manu`:

```bash
source /Users/manoharpaturi/manu/bin/activate
cd /Users/manoharpaturi/Desktop/mocap_manu
python V1/run_v1.py --data /path/to/your/sequence --mode balanced
```

## Requirements for input data

Your data folder should include:

```text
<sequence>/
├── intri.yml
├── extri.yml
└── videos/ or images/
```

## Commands

Run fast mode:

```bash
python V1/run_v1.py --data /path/to/sequence --mode fast
```

Run accurate mode:

```bash
python V1/run_v1.py --data /path/to/sequence --mode accurate
```

Run with extraction first:

```bash
python V1/run_v1.py --data /path/to/sequence --mode balanced --extract-videos --handface
```

Dry run (show generated command only):

```bash
python V1/run_v1.py --data /path/to/sequence --mode accurate --dry-run
```

## Optional tuning

Edit `V1/configs/presets.json` to customize thresholds, smoothing, model, and visualization flags.

## Notes

- The script expects EasyMocap at `../EasyMocap` by default.
- Output defaults to `<data>/output_v1/<mode>`.
- If you want camera subset processing, pass `--sub` and `--sub-vis`.
