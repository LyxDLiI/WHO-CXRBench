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
| **2. GRADE-RA Rule Assessment** | Score, cross-check, and formalize extracted CXR rule candidates. | Two independent LLM graders: `deepseek-ai/DeepSeek-R1` via SiliconFlow and OpenAI `gpt-4.1` via the OpenAI API. | Stage-1 CXR statements plus linked WHO source context. | Each grader independently assigns GRADE-RA scores and structured outputs. The two outputs are ensembled by deterministic post-processing. Only final ensembled `High` and `Moderate` rules are retained, resulting in the released 300-rule corpus. |
| **3. Report-based Personalized Rule Matching** | Align each patient report to applicable WHO rules to create `personal_who_rules`. | `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`, deployed locally with vLLM. | Retained WHO rule corpus plus the report `FINDINGS` and `IMPRESSION` sections. Reports are used only offline for label construction and are not used at retrieval time. | JSON object with matched `rule_id`, `original_statement`, report-derived evidence, `confidence_rating`, and brief rationale. A rule is positive when `confidence_rating >= 0.5`. |

### Stage-level settings

**Stage 1: WHO rule extraction**
- Embedding model: `Qwen/Qwen3-Embedding-8B`.
- Vector store: Chroma disease-specific index with metadata filtering by disease name.
- Source markdown: `chest_x_ray_imaging/PDFDocling/chest_x_ray_imaging.md`.
- Disease section extraction: from `## Disease Name` to the next same-level `##` header.
- Exact retrieval: `similarity_search_with_score(disorder_name, k=10, filter={"disease_name": disorder_name})`.
- Fallback retrieval: broader disease-name similarity search with `k=5` and score threshold `<= 0.8`.
- Metadata attached to chunks: source path, disease name, chapter, start/end lines, section title/path, and content length.
- Chat model: `deepseek-ai/DeepSeek-R1`.
- API: SiliconFlow `/chat/completions`.
- Decoding: `temperature=0.3`, `top_p=0.8`, `max_tokens=2000`, `stream=false`.
- Retry policy: 5 attempts with exponential backoff.
- Timeout: connect 60 seconds, read 600 seconds.
- Prompt constraint: the model is instructed to use only the retrieved WHO disease-specific section and return a single valid JSON object.
- Output constraint: outputs are parsed as JSON and canonicalized by deterministic post-processing scripts before Stage 2.

**Stage 2: GRADE-RA rule assessment**
- Primary grader: `deepseek-ai/DeepSeek-R1`.
- Primary API: SiliconFlow `/chat/completions`.
- Secondary grader: OpenAI `gpt-4.1`, used as a same-period ChatGPT/OpenAI model for independent cross-checking.
- Secondary API: OpenAI API.
- Primary decoding: `temperature=0.1`, `top_p=0.9`, `max_tokens=16384`, `stream=false`, `response_format={"type":"json_object"}`.
- Secondary decoding: `temperature=0.1`, `top_p=0.9`, `max_tokens=16384`, `stream=false`, `response_format={"type":"json_object"}`.
- Additional Stage-2 runtime options for the DeepSeek call: `enable_thinking=true`, `thinking_budget=32768`.
- Retry policy: 3 attempts.
- Timeout: 120 seconds.
- Input fields consumed from Stage 1:
  - `radiological_features.data.primary_signs`
  - `radiological_features.data.secondary_signs`
  - `radiological_features.data.distribution_patterns`
  - `typical_appearance.data`
  - `atypical_presentations.data`, only if `is_present=true`
- Clinical correlation, differential diagnosis, and follow-up recommendations are retained in the Stage-1 output but are not converted into final CXR GRADE-RA rules.
- GRADE-RA domains: extraction bias, inconsistency, indirectness, imprecision, and implementation-specific upgrading factors.
- Two-model ensemble:
  - Each model independently scores the same candidate rule and the same linked WHO source context.
  - If the two graders agree on the final certainty rating, that rating is used directly.
  - If they disagree, the more conservative certainty rating is used.
  - If the disagreement is only between `High` and `Moderate`, the rule is still retained but marked with the lower of the two grades.
  - Rules whose final ensembled rating is `High` or `Moderate` are retained.
- Release filtering: retain only final ensembled `High` and `Moderate` rules as the guideline rule corpus.
- Released rule corpus size: 300 rules.
- Output usage: auxiliary fields such as atomic propositions, logical forms, and knowledge-graph triples are used for audit and traceability. They do not override the deterministic GRADE-RA ensemble and filtering rule.

#### Stage 2 GRADE-RA prompt

**System prompt**

```text
You are a Clinical Radiologist and Medical Informatics Specialist, an expert in chest X-ray interpretation, medical terminology, and evidence-based radiology. Your task is to receive a chest X-ray imaging statement from WHO guidelines and perform rigorous assessment, decomposition, and formalization with emphasis on radiological precision. Your output must be a single, valid JSON object using standardized English medical terminology.
```

**User prompt**

```text
# ROLE
You are a Clinical Radiologist and Medical Informatics Specialist, an expert in chest X-ray interpretation, medical terminology, logic, and evidence-based radiology. You specialize in converting radiological statements into machine-readable knowledge graphs using standardized English medical terminology.

# CONTEXT
You are building a high-fidelity knowledge base for an advanced medical AI focused on chest X-ray interpretation. The goal is to generate precise radiological reasoning paths and a medical logic consistency benchmark for CXR interpretation. You will use the GRADE-ER framework to assess evidence certainty of radiological statements.

# INPUT
You will receive a JSON object containing a single chest X-ray imaging statement for analysis:
{
  "source_chapter": "{source_chapter}",
  "cxr_field_type": "{cxr_field_type}",
  "subfield_type": "{subfield_type}",
  "source_statement": "{statement}",
  "full_ai_response_context": "{full_ai_response}"
}

# CXR-SPECIFIC CONTENT FILTERING
**CRITICAL**: Before processing, evaluate if the `source_statement` contains meaningful radiological information. If the statement indicates:
- "No content available" or "null"
- "Not addressed in provided blocks"
- "Not specified" or "Not mentioned"
- "Not described beyond imaging findings"
- Or similar phrases indicating absence of radiological information

Then assign a final certainty rating of "Very Low" and provide minimal processing.

# TASK: Execute the following four steps for CXR imaging statements.

## STEP 1: Radiological Statement Decomposition
**Only proceed if the statement contains meaningful radiological information.** Decompose the `source_statement` into atomic propositions focused on:
- Specific radiological signs or features
- Anatomical locations and distributions
- X-ray appearance characteristics
- Radiological patterns or configurations
Each proposition should be a verifiable radiological fact using precise medical terminology.

## STEP 2: CXR-Optimized GRADE-ER Assessment
Assume initial certainty is "High" due to WHO guideline source. Systematically review the `source_statement` against five GRADE-ER domains with radiological focus:

- **Extraction Bias (Risk of Bias)**: Assess if extraction process might have distorted radiological terminology or imaging concepts.
  - **Score**: [-2, -1, 0]. -2 for severe radiological term distortion, 0 for accurate extraction.
- **Inconsistency**: Assess if the statement contradicts established radiological principles or other CXR findings in `full_ai_response_context`.
  - **Score**: [-2, -1, 0]. -2 for radiological contradiction, 0 for consistency.
- **Indirectness**: Assess if critical radiological context is lost (e.g., projection type, patient positioning, technical factors).
  - **Score**: [-2, -1, 0]. -2 for loss of critical radiological context, 0 for complete context.
- **Imprecision**: Assess vague radiological terminology ("some shadowing" vs "discrete opacity") or ambiguous anatomical references.
  - **Score**: [-2, -1, 0]. -2 for high imprecision or absence of radiological information, 0 for precise.
- **Upgrading Factors**: Assess for pathognomonic radiological signs, highly specific imaging patterns, or emphasized key criteria.
  - **Score**: [0, +1, +2]. +2 for pathognomonic radiological signs.

## STEP 3: Final Certainty Rating
Calculate the final GRADE-ER certainty rating:
- **Rating Scale**:
  - `Total Score >= 0`: "High"
  - `Total Score = -1`: "Moderate"  
  - `Total Score = -2`: "Low"
  - `Total Score <= -3`: "Very Low"

## STEP 4: CXR Knowledge Graph Formalization
**Only perform detailed formalization for radiologically meaningful statements.** Create knowledge graphs using STRICT STANDARDIZED ENGLISH MEDICAL TERMINOLOGY:

1. **Logical Form**: Combine atomic propositions using operators (¬, ∧, ∨, →) with identifiers p1, p2, etc.

2. **Knowledge Graph Triples**: For each atomic proposition, create triples using STANDARDIZED ENGLISH MEDICAL TERMINOLOGY ONLY:
   - Use precise anatomical terms: "chest wall", "mediastinum", "lung field", "cardiac silhouette", "diaphragm", "pleural space"
   - Use standardized radiological relationships: "has radiological sign", "demonstrates feature", "located in", "shows", "exhibits", "characterized by", "obscures", "displaces"
   - Use specific imaging descriptors: "opacity", "lucency", "consolidation", "displacement", "enlargement", "straightening", "configuration"
   - Use standardized disease/condition terms: "pectus excavatum", "kyphosis", "scoliosis", etc.

**CRITICAL REQUIREMENT**: All entities and relationships MUST be in standardized English medical terminology using LOWERCASE letters with SPACES separating words. Avoid underscores, camelCase, or abbreviations.

# OUTPUT FORMAT
Your entire response MUST be a single, valid JSON object with standardized English medical terminology:

{
  "original_statement": "{statement}",
  "cxr_field_analysis": {
    "field_type": "{cxr_field_type}",
    "subfield_type": "{subfield_type}",
    "radiological_significance": "high/moderate/low"
  },
  "atomic_propositions": [
    "specific radiological proposition 1 using precise medical terms",
    "specific radiological proposition 2 using precise medical terms"
  ],
  "grade_er_assessment": {
    "extraction_bias": { "score": 0, "justification": "Radiological terminology accurately preserved." },
    "inconsistency": { "score": 0, "justification": "Consistent with established CXR interpretation principles." },
    "indirectness": { "score": 0, "justification": "Complete radiological context maintained." },
    "imprecision": { "score": -1, "justification": "Some radiological terms could be more specific." },
    "upgrading_factors": { "score": 1, "justification": "Contains pathognomonic radiological sign." }
  },
  "final_certainty_rating": "High",
  "formalization": {
    "logical_form": "(p1 ∧ p2) → p3",
    "knowledge_graph_triples": [
      {
        "atomic_proposition": "radiological proposition text",
        "triples": [
          {"head_entity": "chest x ray", "relationship": "has radiological sign", "tail_entity": "specific finding"},
          {"head_entity": "specific finding", "relationship": "located in", "tail_entity": "anatomical region"}
        ]
      }
    ]
  }
}
```

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
