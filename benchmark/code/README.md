# WHO-CXRBench Reproducible Pipeline Code

This folder is located at `benchmark/code/` in the repository and contains the reproducible code for the WHO-CXRBench construction stages:

1. `stage1/`: Prior-guided WHO CXR guideline rule extraction.
2. `stage2/`: GRADE-RA rule assessment and formalization.
3. `stage3/`: MIMIC-CXR sample matching against the retained WHO CXR rule corpus.

The implementation is intentionally self-contained. It uses the WHO guideline markdown in `data/chest_x_ray_imaging.md`, a default disease list in `data/default_diseases.txt`, and SiliconFlow chat-completions calls through the `SILICONFLOW_API_KEY` environment variable for Stage 1/2. Stage 3 uses a local OpenAI-compatible vLLM endpoint by default. API keys and machine-specific dataset locations are not stored in code or config.

## Directory Layout

```text
benchmark/code/
  README.md
  environment.yml
  requirements.txt
  run_pipeline.py
  .env.example
  config/
    environment.yml
    pipeline.yml
    paths.json
    stage3_vllm_environment.yml
  LLM_JUDGE_TRANSPARENCY.md
  data/
    chest_x_ray_imaging.md
    abnormalitiesDisease4WHO.txt
    default_diseases.txt
    mimic-cxr-jpg/              # ignored; symlink or local dataset mount
  examples/
    stage3_samples.json
  stage1/
    run_stage1.py
  stage2/
    run_stage2.py
  stage3/
    optimized_batch_processor_local.py
    smart_rule_matcher_local.py
    disease_in_MIMIC_mapto_WHO.csv
    cxr_grade_er_conversion_final_20250712_200006.json
  tests/
    smoke_test_pipeline.py
```

## Environment

Create the conda environment:

```bash
cd benchmark/code
conda env create -f environment.yml
conda activate whocxrbench-pipeline
```

Or install with pip:

```bash
cd benchmark/code
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set the SiliconFlow API key in the shell before running API-backed reproduction:

```bash
export SILICONFLOW_API_KEY="..."
```

Do not write the key into tracked files.

## Dataset Paths

All paths in this release are relative to the `benchmark/code/` directory. The canonical path configuration is:

```text
config/pipeline.yml
```

The unified MIMIC-CXR-JPG location is:

```text
data/mimic-cxr-jpg
```

If the dataset lives elsewhere on your machine, keep the repository path relative by creating a symlink:

```bash
cd benchmark/code
mkdir -p data
ln -s ../../external_datasets/MIMIC-CXR-JPG data/mimic-cxr-jpg
```

Alternatively set:

```bash
export MIMIC_CXR_JPG_ROOT="data/mimic-cxr-jpg"
```

Do not hard-code machine-specific absolute dataset paths in scripts, configs, or committed outputs.

## Run All Stages

Use the centralized YAML config:

```bash
cd benchmark/code
python run_pipeline.py --config config/pipeline.yml --dry-run
```

The dry run executes Stage 1, Stage 2, and Stage 3 without external LLM calls. It is intended for wiring and output-schema validation.

For API-backed Stage 1/2 plus local-vLLM Stage 3:

```bash
cd benchmark/code
export SILICONFLOW_API_KEY="..."
python run_pipeline.py --config config/pipeline.yml
```

Stage 3 expects an OpenAI-compatible local server at the `stage3.server_url` in `config/pipeline.yml`. The bundled vLLM environment is in:

```text
config/stage3_vllm_environment.yml
```

Pipeline outputs default to:

```text
outputs/pipeline/stage1_rules.json
outputs/pipeline/stage2/stage2_grade_ra_all_rules.json
outputs/pipeline/stage2/guideline_rule_corpus_high_moderate.json
outputs/pipeline/stage3/stage3_matched_samples.json
```

## One-Disease Reproducibility Smoke Test

This runs Stage 1 for `Pectus excavatum`, then runs Stage 2 on one extracted CXR field:

```bash
cd benchmark/code
python tests/smoke_test_pipeline.py
```

For local wiring validation without API calls:

```bash
python tests/smoke_test_pipeline.py --dry-run
```

Expected outputs:

```text
outputs/smoke_test/stage1_pectus.json
outputs/smoke_test/stage2/stage2_grade_ra_all_rules.json
outputs/smoke_test/stage2/guideline_rule_corpus_high_moderate.json
```

## Full Stage 1

Run the complete default disease list:

```bash
python stage1/run_stage1.py \
  --disease-file data/default_diseases.txt \
  --output outputs/stage1_rules.json
```

Run a small subset:

```bash
python stage1/run_stage1.py \
  --disease "Pectus excavatum" \
  --output outputs/stage1_pectus.json
```

Stage 1 writes a list of records compatible with the historical pipeline. The key field for Stage 2 is `AI Response.extraction_summary`.

## Full Stage 2

Run GRADE-RA over Stage 1 output:

```bash
python stage2/run_stage2.py \
  --input outputs/stage1_rules.json \
  --output-dir outputs/stage2
```

For a low-cost subset:

```bash
python stage2/run_stage2.py \
  --input outputs/stage1_pectus.json \
  --output-dir outputs/stage2_pectus \
  --max-fields 1
```

Stage 2 writes:

- `stage2_grade_ra_all_rules.json`: all certainty levels.
- `guideline_rule_corpus_high_moderate.json`: the paper-facing rule corpus after retaining `High` and `Moderate` rules.

## Full Stage 3

Run MIMIC-CXR sample matching with the Stage 2 rule output:

```bash
python stage3/optimized_batch_processor_local.py \
  --input examples/stage3_samples.json \
  --output outputs/stage3/stage3_matched_samples.json \
  --cxr-rules outputs/stage2/stage2_grade_ra_all_rules.json \
  --disease-mapping stage3/disease_in_MIMIC_mapto_WHO.csv \
  --mimic-cxr-path data/mimic-cxr-jpg/files
```

For local schema validation without a running vLLM server:

```bash
python stage3/optimized_batch_processor_local.py \
  --input examples/stage3_samples.json \
  --output outputs/stage3/stage3_matched_samples.json \
  --cxr-rules stage3/cxr_grade_er_conversion_final_20250712_200006.json \
  --disease-mapping stage3/disease_in_MIMIC_mapto_WHO.csv \
  --mimic-cxr-path data/mimic-cxr-jpg/files \
  --dry-run
```

If an input sample contains `report_text` or `medical_report`, Stage 3 uses that text directly. Otherwise it reads the MIMIC report from:

```text
data/mimic-cxr-jpg/files/pXX/pSUBJECT/sSTUDY.txt
```

## Reproducibility Decisions

- The cleaned Stage 1 code uses deterministic markdown heading extraction for disease-specific WHO sections. This preserves the prior-guided disease isolation used in the original scripts while avoiding non-portable local Chroma caches.
- The Stage 2 code uses the paper-aligned four GRADE-RA downgrading domains: extraction bias, inconsistency, indirectness, and imprecision.
- The final certainty rating is recomputed deterministically from the four returned domain scores:
  - total score `>= 0`: `High`
  - total score `= -1`: `Moderate`
  - total score `= -2`: `Low`
  - total score `<= -3`: `Very Low`
- `LLM_JUDGE_TRANSPARENCY.md` records the original prompt and implementation details used to answer reviewer concerns about the LLM judge.
- The Stage 3 code separates global WHO rule retrieval from report-specific semantic consolidation. The first step is deterministic disease mapping; the second step uses the local LLM unless `--dry-run` is set.

## Main Parameters

Stage 1 defaults:

- model: `deepseek-ai/DeepSeek-R1`
- base URL: `https://api.siliconflow.cn/v1`
- max tokens: `2000`
- temperature: `0.3`
- top-p: `0.8`

Stage 2 defaults:

- model: `deepseek-ai/DeepSeek-R1`
- base URL: `https://api.siliconflow.cn/v1`
- max tokens: `16384`
- temperature: `0.1`
- top-p: `0.9`
- JSON mode: `response_format={"type": "json_object"}`

## Citation and Provenance

The guideline text is derived from the WHO manual of diagnostic imaging: radiographic anatomy and interpretation of the chest and pulmonary system. Each Stage 1 record stores source line numbers and the extracted source section so downstream rules remain traceable to the guideline text.
