# Stage3 Reproducible Code

This directory contains the reproducible code release for stage3 LLM rule matching. Large model weights are not included; the LLM is referenced by Hugging Face model ID and can be served through vLLM or any compatible OpenAI-style chat-completions API.

The commands below assume they are run from the repository root after entering `benchmark/code`:

```bash
cd benchmark/code
```

## Layout

- `smart_rule_matcher_local.py`: LLM-based report-to-rule semantic matching.
- `optimized_batch_processor_local.py`: batch runner with checkpointing and validation.
- `serve_vllm.sh`: reproducible vLLM serving command.
- `tests/api_smoke_test.py`: minimal OpenAI-compatible API smoke test.
- `environment/`: full conda environment export and pip package freeze.
- `cxr_grade_er_conversion_final_20250712_200006.json`: CXR rule file.
- `disease_in_MIMIC_mapto_WHO.csv`: disease-name mapping file.

## Environment

Create the environment from the exported YAML:

```bash
conda env create -f stage3/environment/environment.yml
conda activate ms-swift
```

The key runtime versions used for stage3 were:

- `python=3.10.18`
- `torch=2.6.0+cu124`
- `transformers=4.51.3`
- `vllm=0.8.5.post1`

## Data Paths

The scripts avoid hard-coded local paths. Configure local data locations through environment variables when needed:

```bash
export MIMIC_CXR_JPG_ROOT=data/mimic-cxr-jpg
```

## Large Files

Do not commit model weights or MIMIC-CXR images/reports into this repository. The `.gitignore` excludes common model checkpoint formats and local data folders.
