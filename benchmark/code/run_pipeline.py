#!/usr/bin/env python3
"""Run the WHO-CXRBench construction pipeline from one YAML config."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parent


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Pipeline config must be a YAML mapping: {path}")
    return config


def as_rel(path: str | Path) -> str:
    return Path(path).as_posix()


def run_command(cmd: Iterable[str], *, root: Path, env: Dict[str, str]) -> None:
    command = list(cmd)
    print("+ " + " ".join(command))
    subprocess.run(command, cwd=root, env=env, check=True)


def optional_value(args: list[str], flag: str, value: Any) -> None:
    if value is not None:
        args.extend([flag, str(value)])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run WHO-CXRBench Stage 1 -> Stage 2 -> Stage 3.")
    parser.add_argument("--config", type=Path, default=project_root() / "config" / "pipeline.yml")
    parser.add_argument("--dry-run", action="store_true", help="Validate wiring without external LLM calls.")
    parser.add_argument("--skip-stage1", action="store_true")
    parser.add_argument("--skip-stage2", action="store_true")
    parser.add_argument("--skip-stage3", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    root = project_root()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = load_config(config_path)
    paths = config.get("paths", {})
    stage1 = config.get("stage1", {})
    stage2 = config.get("stage2", {})
    stage3 = config.get("stage3", {})
    smoke = config.get("smoke_test", {})

    output_dir = Path(paths.get("output_dir", "outputs/pipeline"))
    stage1_output = Path(smoke.get("stage1_output", output_dir / "stage1_rules.json"))
    stage2_output_dir = Path(smoke.get("stage2_output_dir", output_dir / "stage2"))
    default_stage3_output = output_dir / "stage3" / "stage3_matched_samples.json"
    stage3_output_file = Path(smoke.get("stage3_output_file", stage3.get("output_file", default_stage3_output)))

    env = os.environ.copy()
    mimic_root = paths.get("mimic_cxr_jpg_root", "data/mimic-cxr-jpg")
    env.setdefault("MIMIC_CXR_JPG_ROOT", str(mimic_root))

    if not args.dry_run and not env.get("SILICONFLOW_API_KEY") and (not args.skip_stage1 or not args.skip_stage2):
        raise SystemExit("SILICONFLOW_API_KEY is required unless --dry-run is set or Stage 1/2 are skipped.")

    python_cmd = "python"

    if not args.skip_stage1:
        stage1_cmd = [
            python_cmd,
            "stage1/run_stage1.py",
            "--markdown",
            as_rel(paths.get("who_guideline_markdown", "data/chest_x_ray_imaging.md")),
            "--output",
            as_rel(stage1_output),
            "--base-url",
            str(stage1.get("base_url", "https://api.siliconflow.cn/v1")),
            "--model",
            str(stage1.get("model", "deepseek-ai/DeepSeek-R1")),
            "--max-tokens",
            str(stage1.get("max_tokens", 2000)),
            "--temperature",
            str(stage1.get("temperature", 0.3)),
            "--top-p",
            str(stage1.get("top_p", 0.8)),
            "--max-retries",
            str(stage1.get("max_retries", 5)),
        ]
        diseases = stage1.get("diseases") or []
        if diseases:
            for disease in diseases:
                stage1_cmd.extend(["--disease", str(disease)])
        else:
            stage1_cmd.extend(["--disease-file", as_rel(paths.get("default_disease_list", "data/default_diseases.txt"))])
        if args.dry_run:
            stage1_cmd.append("--dry-run")
        run_command(stage1_cmd, root=root, env=env)

    if not args.skip_stage2:
        stage2_cmd = [
            python_cmd,
            "stage2/run_stage2.py",
            "--input",
            as_rel(stage1_output),
            "--output-dir",
            as_rel(stage2_output_dir),
            "--base-url",
            str(stage2.get("base_url", "https://api.siliconflow.cn/v1")),
            "--model",
            str(stage2.get("model", "deepseek-ai/DeepSeek-R1")),
            "--max-tokens",
            str(stage2.get("max_tokens", 16384)),
            "--temperature",
            str(stage2.get("temperature", 0.1)),
            "--top-p",
            str(stage2.get("top_p", 0.9)),
            "--max-retries",
            str(stage2.get("max_retries", 3)),
            "--thinking-budget",
            str(stage2.get("thinking_budget", 32768)),
        ]
        optional_value(stage2_cmd, "--limit-diseases", stage2.get("limit_diseases"))
        optional_value(stage2_cmd, "--max-fields", stage2.get("max_fields"))
        if stage2.get("enable_thinking", False):
            stage2_cmd.append("--enable-thinking")
        if args.dry_run:
            stage2_cmd.append("--dry-run")
        run_command(stage2_cmd, root=root, env=env)

    if not args.skip_stage3:
        stage3_dry_run = args.dry_run or bool(stage3.get("dry_run", False))
        stage3_rules_path = stage3.get("cxr_rules_path", stage2_output_dir / "stage2_grade_ra_all_rules.json")
        stage3_cmd = [
            python_cmd,
            "stage3/optimized_batch_processor_local.py",
            "--input",
            as_rel(stage3.get("input_samples", "examples/stage3_samples.json")),
            "--output",
            as_rel(stage3_output_file),
            "--server-url",
            str(stage3.get("server_url", "http://localhost:8000")),
            "--model",
            str(stage3.get("model", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")),
            "--cxr-rules",
            as_rel(stage3_rules_path),
            "--disease-mapping",
            as_rel(paths.get("stage3_disease_mapping", "stage3/disease_in_MIMIC_mapto_WHO.csv")),
            "--mimic-cxr-path",
            as_rel(Path(mimic_root) / "files"),
            "--checkpoint-interval",
            str(stage3.get("checkpoint_interval", 100)),
            "--start-idx",
            str(stage3.get("start_idx", 0)),
            "--log-dir",
            as_rel(output_dir / "logs"),
        ]
        optional_value(stage3_cmd, "--max-samples", stage3.get("max_samples"))
        if stage3_dry_run:
            stage3_cmd.append("--dry-run")
        run_command(stage3_cmd, root=root, env=env)

    print("[pipeline] completed")


if __name__ == "__main__":
    main()
