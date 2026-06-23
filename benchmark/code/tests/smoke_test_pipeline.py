#!/usr/bin/env python3
"""End-to-end smoke test for the open-source WHO-CXRBench pipeline."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run_command(cmd: list[str], *, env: dict[str, str]) -> None:
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def assert_stage1(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, list) and data, "Stage 1 output must be a non-empty list"
    record = data[0]
    assert record["disorder_name"] == "Pectus excavatum"
    assert "AI Response" in record
    assert record["source_documents"][0][0]["metadata"]["start_line"] > 0
    print(f"[smoke] Stage 1 OK: {path}")


def assert_stage2(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "summary" in data and "all_rules" in data
    assert data["summary"]["total_cxr_rules_formalized"] >= 1
    rule = data["all_rules"][0]
    assert rule["final_certainty_rating"] in {"High", "Moderate", "Low", "Very Low"}
    assert "grade_ra_assessment" in rule
    print(f"[smoke] Stage 2 OK: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one-disease Stage 1 -> Stage 2 smoke test.")
    parser.add_argument("--dry-run", action="store_true", help="Do not call SiliconFlow; validate local wiring only.")
    parser.add_argument("--output-dir", type=Path, default=project_root() / "outputs" / "smoke_test")
    parser.add_argument("--stage1-max-tokens", type=int, default=2000)
    parser.add_argument("--stage2-max-tokens", type=int, default=4096)
    args = parser.parse_args()

    root = project_root()
    stage1_output = args.output_dir / "stage1_pectus.json"
    stage2_output_dir = args.output_dir / "stage2"
    stage2_output = stage2_output_dir / "stage2_grade_ra_all_rules.json"

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if not args.dry_run and not env.get("SILICONFLOW_API_KEY"):
        raise SystemExit("SILICONFLOW_API_KEY is required unless --dry-run is set.")

    stage1_cmd = [
        sys.executable,
        str(root / "stage1" / "run_stage1.py"),
        "--disease",
        "Pectus excavatum",
        "--output",
        str(stage1_output),
        "--max-tokens",
        str(args.stage1_max_tokens),
    ]
    if args.dry_run:
        stage1_cmd.append("--dry-run")
    run_command(stage1_cmd, env=env)
    assert_stage1(stage1_output)

    stage2_cmd = [
        sys.executable,
        str(root / "stage2" / "run_stage2.py"),
        "--input",
        str(stage1_output),
        "--output-dir",
        str(stage2_output_dir),
        "--max-fields",
        "1",
        "--max-tokens",
        str(args.stage2_max_tokens),
    ]
    if args.dry_run:
        stage2_cmd.append("--dry-run")
    run_command(stage2_cmd, env=env)
    assert_stage2(stage2_output)

    print("[smoke] Pipeline smoke test passed.")


if __name__ == "__main__":
    main()
