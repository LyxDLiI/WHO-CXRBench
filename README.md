# WHO-CXRBench (Preview)

WHO-CXRBench is a guideline-grounded benchmark for fine-grained diagnostic rule retrieval and alignment on chest radiography. The benchmark is built around WHO chest X-ray (CXR) guidelines and provides rule-level supervision for linking CXR findings and disease answers to guideline-derived diagnostic rules.

## Current Release

Because the full benchmark depends on MIMIC-CXR data governed by a data-use agreement, this public preview does not redistribute raw MIMIC-CXR images, reports, identifiers, or image paths. We are preparing the complete data release through the appropriate controlled-access channel.

This repository currently includes:

- `rule_corpus/cxr_grade_er_rules.json`: the released WHO-derived CXR rule corpus, containing 300 High/Moderate rules.
- `rule_corpus/who_pdf_files/`: WHO source PDFs used as guideline references.
- `benchmark/WHO_CXRBench_100_samples.json`: 100 de-identified preview samples showing the benchmark schema.
- `benchmark/code/`: reproducible construction code and documentation for Stage 1, Stage 2, and Stage 3.

## Repository Layout

```text
WHO-CXRBench/
  README.md
  benchmark/
    WHO_CXRBench_100_samples.json
    code/
      README.md
      run_pipeline.py
      config/pipeline.yml
      stage1/
      stage2/
      stage3/
      LLM_JUDGE_TRANSPARENCY.md
  rule_corpus/
    cxr_grade_er_rules.json
    who_pdf_files/
```

## Data Format

Each preview sample contains a natural-language query, answer labels, and two rule sets:

- `global_who_rules`: corpus-level candidate WHO CXR rules matched from the answer labels.
- `personal_who_rules`: sample-specific rules aligned to the original radiology report during offline construction.

Raw MIMIC-CXR reports are not released. The preview samples preserve the output schema without exposing protected report text or MIMIC-CXR-JPG paths.

```json
{
  "split": "test",
  "idx": 1,
  "question": "Provide a catalogue of all anatomical findings and diseases seen.",
  "semantic_type": "query",
  "content_type": "attribute",
  "template": "Provide a catalogue of all ${category_1} and ${category_2} seen.",
  "template_program": "program_26",
  "template_arguments": {},
  "answer": ["..."],
  "global_who_rules": [
    {
      "rule_id": "...",
      "disease": "...",
      "original_statement": "..."
    }
  ],
  "personal_who_rules": [
    {
      "rule_id": "...",
      "original_statement": "...",
      "semantic_evidence": "...",
      "confidence_rating": 0.0,
      "detailed_reasoning": "..."
    }
  ],
  "medical_report_found": true
}
```

## Construction Pipeline

The released code implements a three-stage offline construction pipeline:

| Stage | Location | Purpose | LLM/runtime |
|---|---|---|---|
| Stage 1 | `benchmark/code/stage1/` | Extract disease-specific CXR rule candidates from the WHO guideline markdown. | `deepseek-ai/DeepSeek-R1` through the SiliconFlow OpenAI-compatible chat API. |
| Stage 2 | `benchmark/code/stage2/` | Perform GRADE-RA assessment, decompose CXR statements, formalize rules, and deterministically recompute final certainty ratings from four domains. | `deepseek-ai/DeepSeek-R1` through the SiliconFlow OpenAI-compatible chat API. |
| Stage 3 | `benchmark/code/stage3/` | Match retained WHO rules to report-derived evidence to produce `personal_who_rules`. | `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` served locally through vLLM or another OpenAI-compatible endpoint. |

The code release is intentionally self-contained for reproducibility. API keys are read from environment variables and are not stored in the repository. MIMIC-CXR-JPG is referenced through the relative path `benchmark/code/data/mimic-cxr-jpg` or the environment variable `MIMIC_CXR_JPG_ROOT`.

## Quick Start

Run the local wiring test without external LLM calls:

```bash
cd benchmark/code
python run_pipeline.py --config config/pipeline.yml --dry-run
```

For API-backed Stage 1/2 and local-vLLM Stage 3:

```bash
cd benchmark/code
export SILICONFLOW_API_KEY="..."
export MIMIC_CXR_JPG_ROOT=data/mimic-cxr-jpg
python run_pipeline.py --config config/pipeline.yml
```

Detailed setup instructions are in `benchmark/code/README.md`.

## LLM Judge Transparency

The LLM-assisted components are documented in:

```text
benchmark/code/LLM_JUDGE_TRANSPARENCY.md
```

The released Stage 2 code uses the paper-aligned GRADE-RA downgrading domains:

- extraction bias
- inconsistency
- indirectness
- imprecision

The final certainty rating is recomputed deterministically from those four scores. The released rule corpus keeps the High and Moderate rules.

## Intended Use

- Rule retrieval and rule-ranking evaluation for CXR questions.
- Confidence modeling and calibration using `confidence_rating`.
- Explainable alignment analysis using `global_who_rules`, `personal_who_rules`, `semantic_evidence`, and `detailed_reasoning`.

## Status

This repository is a public preview. The complete benchmark release is pending controlled-access data release approval. Updates will be posted in this repository.

## License

WHO source files follow the original WHO terms. The preview annotations and reproducible code are provided for research use; a final repository license will be added before the complete release.
