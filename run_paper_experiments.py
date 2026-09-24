# ITNG YMYFA 80asra YA
# IMZZ


"""Run the six experiments behind the Allerton 2026 figures and tables.

The settings below are taken from the merged metadata saved with the paper
results. A smoke run checks the pipeline with one setup. Each original run
uses 100 independently executable chunks, followed by a merge.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = PROJECT_ROOT / "generated_results"
BASE_SEED = 20260606
PAPER_CHUNKS = 100


@dataclass(frozen=True)
class Experiment:
    setups: int
    pilot_length: int | None
    association_initialization: str


EXPERIMENTS = {
    "final_tauK_random_S200": Experiment(200, None, "random"),
    "final_tauKhalf_random_S200": Experiment(200, 5, "random"),
    "sen_tauK_strongest_S100": Experiment(100, None, "strongest_singleton"),
    "sen_tauK_full_S100": Experiment(100, None, "full_candidate"),
    "sen_tauKhalf_strongest_S100": Experiment(100, 5, "strongest_singleton"),
    "sen_tauKhalf_full_S100": Experiment(100, 5, "full_candidate"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=EXPERIMENTS)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="Run one setup in a separate output folder")
    mode.add_argument("--chunk-id", type=int, metavar="N", help="Run original chunk N, from 0 to 99")
    mode.add_argument("--merge", action="store_true", help="Merge all 100 completed original chunks")
    parser.add_argument("--dry-run", action="store_true", help="Show the command without running it")
    args = parser.parse_args()

    if args.chunk_id is not None and not 0 <= args.chunk_id < PAPER_CHUNKS:
        parser.error("--chunk-id must be between 0 and 99")

    exp = EXPERIMENTS[args.experiment]
    is_smoke = args.smoke
    run_dir = RESULTS_ROOT / ("smoke" if is_smoke else "full") / args.experiment
    command = [
        sys.executable,
        str(PROJECT_ROOT / "convergence_behavior.py"),
        "--mode", "merge" if args.merge else "run_chunk",
        "--run-dir", str(run_dir),
        "--tag", args.experiment,
        "--total-setups", str(1 if is_smoke else exp.setups),
        "--total-chunks", str(1 if is_smoke else PAPER_CHUNKS),
        "--base-seed", str(BASE_SEED),
        "--config-overrides-json", json.dumps({"pilots.pilot_len": exp.pilot_length}),
        "--assoc-init-action-mode", exp.association_initialization,
    ]
    if not args.merge:
        command.extend(("--chunk-id", str(0 if is_smoke else args.chunk_id)))

    if args.merge and not args.dry_run:
        chunks = run_dir / "chunks"
        missing = [i for i in range(PAPER_CHUNKS) if not (chunks / f"data_chunk_{i:03d}.npz").is_file()]
        if missing:
            parser.error(f"Cannot merge: {len(missing)} chunk files are missing. First missing ID: {missing[0]}")
        for i in range(PAPER_CHUNKS):
            metadata_path = chunks / f"metadata_chunk_{i:03d}.json"
            if not metadata_path.is_file():
                parser.error(f"Cannot merge: missing {metadata_path}")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if any(details.get("status") != "ok" for details in metadata["scenarios"].values()):
                parser.error(f"Cannot merge: incomplete simulation stages in {metadata_path}")

    print("Running:", subprocess.list2cmdline(command), flush=True)
    if args.dry_run:
        return

    subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    if not args.merge:
        chunk_id = 0 if is_smoke else args.chunk_id
        metadata_path = run_dir / "chunks" / f"metadata_chunk_{chunk_id:03d}.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        failed = [
            name for name, details in metadata["scenarios"].items()
            if details.get("status") != "ok"
        ]
        if failed:
            raise SystemExit(
                f"Simulation finished with incomplete stages in {', '.join(failed)}. "
                f"Inspect {metadata_path} before using these results."
            )
    print("Completed:", run_dir)


if __name__ == "__main__":
    main()
