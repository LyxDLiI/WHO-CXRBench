# WHO-CXRBench (Preview)

This repository provides a **guideline-grounded dataset** for **fine-grained diagnostic rule retrieval and alignment** on chest radiography, built around WHO chest X-ray (CXR) guidelines.

## What is released (current)
Due to the **MIMIC-CXR data-use agreement**, we cannot redistribute the full underlying clinical data at this moment. We are **actively applying to release the complete dataset via PhysioNet**.

For now, this repo includes:
- **Rule corpus** (machine-readable WHO diagnostic rule statements)
- **Original WHO source files** used to derive the rules (WHO documents are redistributed for research reference; all rights remain with WHO.)
- **100 de-identified samples** to illustrate the data format (no MIMIC-CXR-JPG identifiers or image paths; no report text)

## Data format (per sample)
Each sample contains a natural-language query, a structured answer list, and two rule sets:
- `global_who_rules`: corpus-level candidate rules (with IDs, disease names, and original statements)
- `personal_who_rules`: sample-specific rules with `confidence_rating` and `detailed_reasoning`

```json
{
  "idx": 000001,
  "question": "Provide a catalogue of all anatomical findings and diseases seen.",
  "semantic_type": "query",
  "content_type": "attribute",
  "template": "Provide a catalogue of all ${category_1} and ${category_2} seen.",
  "template_program": "program_26",
  "template_arguments": { ... },

  "answer": ["...", "..."],

  "global_who_rules": [ ... ],

  "personal_who_rules": [
    {
      "rule_id": "...",
      "original_statement": "...",
      "confidence_rating": 0-1,
      "detailed_reasoning": "..."
    }
  ]
}
```

See `/benchmark/` for the 100-sample preview and field definitions.

## LLM-assisted construction transparency

WHO-CXRBench is built by a three-stage offline construction pipeline. To make the LLM-based components reproducible, we disclose the judge/extractor identity, runtime, decoding settings, input/output contract, deterministic post-processing, and known reproducibility boundaries below. API keys and raw MIMIC-CXR report text are not released. Full prompt templates and expanded construction notes are provided in `LLM_JUDGE_DETAILS.md`.

The construction pipeline uses reports only for offline label construction. At retrieval time, models receive only the CXR image query and rank the released WHO-derived rule corpus.

| Stage | Purpose | LLM / runtime | Main input | Output / decision |
|---|---|---|---|---|
| **1. Prior-Guided Content RAG** | Extract disease-specific candidate rules from WHO CXR guideline sections. | `deepseek-ai/DeepSeek-R1` via SiliconFlow OpenAI-compatible chat API. | Disease-filtered WHO guideline chunks retrieved with `Qwen/Qwen3-Embedding-8B`. | Schema-constrained JSON containing CXR-relevant rule candidates, including radiological features, typical/atypical appearances, clinical correlation, differential diagnosis, anatomical location, distribution pattern, and follow-up fields. |
| **2. GRADE-RA Rule Assessment** | Score, cross-check, and formalize extracted CXR rule candidates. | Two independent LLM graders: `deepseek-ai/DeepSeek-R1` via SiliconFlow and OpenAI `gpt-4.1` via the OpenAI API. | Stage-1 CXR statements plus linked WHO source context. | Each grader independently assigns per-domain GRADE-RA downgrade scores. The two outputs are ensembled by deterministic post-processing. Only the final ensembled `High` and `Moderate` rules are retained, resulting in the released 300-rule corpus. |
| **3. Report-based Personalized Rule Matching** | Align each patient report to applicable WHO rules to create `personal_who_rules`. | `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`, deployed locally with vLLM. | Retained WHO rule corpus plus the report `FINDINGS` and `IMPRESSION` sections. Reports are used only offline for label construction and are not used at retrieval time. | JSON object with matched `rule_id`, `original_statement`, report-derived evidence, `confidence_rating`, and brief rationale. A rule is positive when `confidence_rating >= 0.5`. |

### Stage-level settings

**Stage 1: WHO rule extraction**
- Embedding model: `Qwen/Qwen3-Embedding-8B`.
- Vector store: Chroma disease-specific index with metadata filtering by disease name.
- Retrieval constraint: all retrieval is performed under a disease-name metadata filter; cross-disease chunks are not eligible for generation.
- Similarity metric: cosine similarity over normalized embedding vectors.
- Chat model: `deepseek-ai/DeepSeek-R1`.
- API: SiliconFlow `/chat/completions`.
- Decoding: `temperature=0.3`, `top_p=0.8`, `max_tokens=2000`, `stream=false`.
- Prompt constraint: the model is instructed to use only the retrieved WHO disease-specific section and return a single valid JSON object.
- Output constraint: outputs are parsed as JSON and canonicalized by deterministic post-processing scripts before Stage 2.

> **Implementation-specific RAG parameters.** Fill these values from the released construction script before public release:
> - `chunk_size = <FILL_FROM_SCRIPT>`
> - `chunk_overlap = <FILL_FROM_SCRIPT>`
> - `retrieval_top_k = <FILL_FROM_SCRIPT>`
> - `max_retrieved_chunks_per_disease = <FILL_FROM_SCRIPT>`

**Stage 2: GRADE-RA rule assessment**
- Primary grader: `deepseek-ai/DeepSeek-R1`.
- Primary API: SiliconFlow `/chat/completions`.
- Secondary grader: OpenAI `gpt-4.1`, a contemporaneous ChatGPT-available model from May 2025.
- Rationale for secondary grader choice: `gpt-4.1` was selected as a same-period OpenAI ChatGPT/API model with strong instruction-following and long-context capabilities, suitable for structured GRADE-style judging.
- Secondary API: OpenAI Chat Completions / Responses API, depending on the construction-time implementation.
- Primary decoding: `temperature=0.1`, `top_p=0.9`, `max_tokens=16384`, `stream=false`, `response_format={"type":"json_object"}`.
- Secondary decoding: `temperature=<FILL_FROM_SCRIPT>`, `top_p=<FILL_FROM_SCRIPT>`, `max_tokens=<FILL_FROM_SCRIPT>`, `stream=false`, `response_format={"type":"json_object"}`.
- GRADE-RA domains: extraction bias, inconsistency, indirectness, and imprecision, each scored in `{-2, -1, 0}`.
- Each model independently scores the same candidate rule and the same linked WHO source context.
- Deterministic single-model grade mapping:

```text
S = extraction_bias + inconsistency + indirectness + imprecision

S = 0    -> High
S = -1   -> Moderate
S = -2   -> Low
S <= -3  -> Very Low
```

Because all downgrade scores are non-positive, `S > 0` cannot occur.

- Two-model ensemble: the two graders' per-domain scores are combined by a deterministic ensemble rule before final filtering.
- Recommended release wording for the ensemble rule:

```text
For each candidate rule, we first compute the total downgrade score from each grader.
If the two graders agree on the final grade, that grade is used directly.
If they disagree, we use the more conservative grade unless the disagreement is only between High and Moderate, in which case the rule is still retained but marked with the lower of the two grades.
Rules whose final ensembled grade is High or Moderate are retained.
```

- Release filtering: retain only the final ensembled `High` and `Moderate` rules as the guideline rule corpus.
- Released rule corpus size: 300 rules.
- Output usage: auxiliary fields such as atomic propositions, logical forms, and knowledge-graph triples are used for audit and traceability. They do not override the deterministic GRADE-RA ensemble and filtering rule.

> **Implementation-specific Stage-2 parameters.** Fill these values from the released construction script before public release:
> - `OPENAI_MODEL_ID = gpt-4.1`
> - `openai_api_version_or_snapshot_date = May 2025 ChatGPT/API generation; exact request date should be filled from the API log if available`
> - `secondary_temperature = <FILL_FROM_SCRIPT>`
> - `secondary_top_p = <FILL_FROM_SCRIPT>`
> - `secondary_max_tokens = <FILL_FROM_SCRIPT>`
> - `ensemble_policy = <FILL_FROM_SCRIPT>`  
>
> We use the explicit OpenAI model ID `gpt-4.1` to avoid the vague phrase “a ChatGPT model”. If the exact construction-time request date is available from the API log, include it in the release notes.

**Stage 3: report-rule semantic matching**
- Model ID: `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`.
- Hugging Face snapshot/revision: `6e8885a6ff5c1dc5201574c8fd700323f23c25fa`.
- Architecture: `Qwen3ForCausalLM`; 36 layers; hidden size 4096; 32 attention heads; 8 KV heads; `bfloat16`; no quantization.
- vLLM version: `0.8.5.post1`.
- Runtime: CUDA; tensor parallel size 4; maximum model length 32768.
- Precision: `bfloat16`.
- Quantization: none.
- Rule threshold: `confidence_rating >= 0.5`.

- Serving command:

```bash
vllm serve deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.9 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --generation-config vllm \
  --revision 6e8885a6ff5c1dc5201574c8fd700323f23c25fa \
  --tokenizer-revision 6e8885a6ff5c1dc5201574c8fd700323f23c25fa \
  --trust-remote-code
```

- Seed note: the original Stage-3 construction run did not explicitly set a seed. To support release-level reproducibility, we release the generated matching outputs and deterministic post-processing scripts. If rerunning the construction pipeline, users may additionally set a fixed vLLM seed.

- Decoding request:

```json
{
  "model": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
  "temperature": 0.0,
  "top_p": 1.0,
  "max_tokens": 8192,
  "stream": false,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0,
  "response_format": {"type": "json_object"}
}
```

`temperature=0.0` is used to make report-rule matching as deterministic as possible. `top_p` is set to `1.0` because nucleus sampling is inactive under greedy decoding.

- Output schema:

```json
{
  "matches": [
    {
      "rule_id": "exact_rule_id",
      "original_statement": "clinical rule text",
      "semantic_evidence": "evidence extracted from the report",
      "confidence_rating": 0.0,
      "brief_rationale": "brief explanation of the assigned score"
    }
  ]
}
```

## Intended use
- Confidence modeling / calibration using `confidence_rating`
- Explainable alignment analysis via `detailed_reasoning`

## Status
Full release is pending **PhysioNet** approval. Updates will be posted in this repository.

We will release the automated construction pipeline (code and documentation) in a subsequent update.

## License
- WHO source files: follow the original WHO terms.
- Released annotations/examples in this repo: see `LICENSE` (or add one before release).
