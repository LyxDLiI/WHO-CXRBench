# WHO-CXRBench LLM Judge and Prompt Transparency

This README summarizes the LLM-assisted components used by the first two benchmark-construction stages described in `WHO-CXRBech-paper-505.pdf`, with implementation details from:

- `who_chest_xray_rag_chapter_based_en_v4.py`
- `enhanced_rag_to_snomed_converter_v5_cxr_nolimit.py`

The two scripts correspond to the first two stages in Fig. 2 and Sec. 2.2-2.3 of the paper:

1. **Prior-Guided Content RAG**: extract a traceable, structured, disease-specific rule corpus from the WHO chest radiography guideline.
2. **GRADE-RA / GRADE-based Rule Assessment**: judge extracted radiological rule quality, decompose statements, assign evidence-quality scores, and formalize rules.

The third paper stage, **Report-based Personalized Rule Matching** in Sec. 2.4, is not implemented in these two scripts and is therefore outside the scope of this README.

## Reviewer-Facing Summary

For reproducibility, the LLM judge was not a hidden black box. In the first stage, DeepSeek-R1 was used through the SiliconFlow chat-completions API to perform schema-constrained extraction from disease-specific WHO guideline sections retrieved by metadata-filtered RAG. In the second stage, DeepSeek-R1 was used as a radiology-focused GRADE-RA judge to score rule quality domains, decompose statements into atomic propositions, and generate logical forms and knowledge-graph triples. The model, API parameters, inputs, outputs, scoring domains, and exact prompts are disclosed below.

No external medical knowledge was intended to be used during rule extraction: the prompt explicitly restricts the model to the retrieved WHO guideline section. The GRADE-RA stage also supplies the full structured rule context and asks the model to evaluate extraction bias, inconsistency, indirectness, and imprecision. The implementation prompt additionally contains an `upgrading_factors` domain; if the camera-ready paper follows the four-domain description in Sec. 2.3 exactly, this should either be removed from code or disclosed as an implementation detail. The v5 script preserves all certainty levels; the paper-level corpus `G` should be reproduced by filtering to `High` and `Moderate`.

## Stage 1: Prior-Guided Content RAG Rule Extraction

**Paper location:** Sec. 2.2, "WHO Rule Extraction via Prior-Guided Content RAG".

**Script:** `who_chest_xray_rag_chapter_based_en_v4.py`

**LLM role:** schema-constrained guideline information extractor. It reads one disease-specific WHO guideline section and produces a structured JSON object containing radiological features, typical appearance, atypical presentations, clinical correlation, differential diagnosis, and follow-up recommendations.

**Retrieval and evidence source**

- Source markdown: `chest_x_ray_imaging/PDFDocling/chest_x_ray_imaging.md`
- Disease section extraction: from `## Disease Name` to the next same-level `##` header.
- Vector store: `./chroma_db_v4/WHO_chest_xray_disease_specific`
- Embedding model: `Qwen/Qwen3-Embedding-8B`
- Exact retrieval: `similarity_search_with_score(disorder_name, k=10, filter={"disease_name": disorder_name})`
- Fallback retrieval: broader disease-name similarity search with `k=5` and score threshold `<= 0.8`
- Metadata attached to chunks: source path, disease name, chapter, start/end lines, section title/path, and content length.

**Generation API parameters**

- Chat model: passed as `model_name` from Hydra config; project configs use `deepseek-ai/DeepSeek-R1`.
- Base URL: `https://api.siliconflow.cn/v1`
- Endpoint: `/chat/completions`
- `stream`: `false`
- `max_tokens`: `2000`
- `temperature`: `0.3`
- `top_p`: `0.8`
- Retry policy: 5 attempts with exponential backoff.
- Timeout: connect 60 seconds, read 600 seconds.
- API key: read from config in this script family; publish only the variable/config field name, not the key value.

**Stage 1 output schema**

The expected LLM output is a single JSON object:

```json
{
  "disease_name": "<disease>",
  "source_document": "WHO Chest X-ray Imaging Guideline Chapter <chapter>",
  "extraction_summary": {
    "radiological_features": {
      "is_present": true,
      "data": {
        "primary_signs": "...",
        "secondary_signs": "...",
        "distribution_patterns": "..."
      }
    },
    "typical_appearance": {
      "is_present": true,
      "data": "..."
    },
    "atypical_presentations": {
      "is_present": true,
      "data": "..."
    },
    "clinical_correlation": {
      "is_present": true,
      "data": {
        "patient_population": "...",
        "symptom_correlation": "...",
        "risk_factors": "..."
      }
    },
    "differential_diagnosis": {
      "is_present": true,
      "data": "..."
    },
    "follow_up_recommendations": {
      "is_present": true,
      "data": "..."
    }
  }
}
```

### Stage 1 Main System Prompt

Source: `who_chest_xray_rag_chapter_based_en_v4.py`, `extract_diagnostic_info_disease_specific_fallback`.

```text
You are an expert AI assistant specializing in medical information extraction from clinical guidelines. Your task is to act as a professional radiologist and meticulously analyze the provided section on a specific disease from the WHO Chest X-ray Imaging Guideline.

CRITICAL INSTRUCTIONS:
- Your analysis MUST be based EXCLUSIVELY on the provided text for "{disorder_name}".
- DO NOT use any external knowledge or make assumptions beyond the provided content.
- Your goal is to READ, COMPREHEND, and SYNTHESIZE the information into a structured JSON format.
- For each field in the JSON schema, you must determine if the information is present in the text.
- If information for a field is present, synthesize it concisely using precise radiological terminology from the text and populate the "data" sub-field.
- If information for a field is NOT present in the text, you MUST set the "is_present" flag to false and the "data" field to null.
- Adhere STRICTLY to the JSON schema provided below. Do not add, remove, or rename any keys. The final output must be a single, valid JSON object and nothing else.

CONTENT METADATA:
{metadata_info}

DISEASE/CONDITION TO ANALYZE: {disorder_name}

JSON OUTPUT SCHEMA:
{
  "disease_name": "{disorder_name}",
  "source_document": "WHO Chest X-ray Imaging Guideline Chapter {chapter_num}",
  "extraction_summary": {
    "radiological_features": {
      "is_present": true,
      "data": {
        "primary_signs": "Synthesized description of primary radiological signs.",
        "secondary_signs": "Synthesized description of secondary signs and other findings.",
        "distribution_patterns": "Synthesized description of distribution and anatomical locations."
      }
    },
    "typical_appearance": {
      "is_present": true,
      "data": "A synthesized summary of the typical overall chest X-ray appearance."
    },
    "atypical_presentations": {
      "is_present": true,
      "data": "A synthesized summary of any described variations or atypical presentations."
    },
    "clinical_correlation": {
      "is_present": true,
      "data": {
        "patient_population": "Synthesized description of typical patient populations.",
        "symptom_correlation": "Synthesized description of correlations with clinical symptoms.",
        "risk_factors": "Synthesized description of predisposing conditions or risk factors."
      }
    },
    "differential_diagnosis": {
      "is_present": true,
      "data": "A synthesized list or description of conditions to be considered in the differential diagnosis, based on the text."
    },
    "follow_up_recommendations": {
      "is_present": true,
      "data": "A synthesized summary of any recommendations for follow-up or further imaging (e.g., CT)."
    }
  }
}

COMPLETE SECTION CONTENT FOR {disorder_name}:
{context}

Now, perform the analysis and generate the JSON output.
```

### Stage 1 Main User Prompt

```text
QUERY: WHO guideline-based diagnostic criteria and chest X-ray imaging findings for {disorder_name}

STRICT REQUIREMENTS:
- Analyze ONLY the provided WHO {disorder_name} section content
- Extract diagnostic standards directly from this disease-specific guideline content
- Use English language exclusively
- Quote specific content when making statements
- Focus on evidence-based chest X-ray diagnostic criteria

Provide comprehensive diagnostic standards for {disorder_name} based strictly on the WHO guideline content provided above.
```

### Stage 1 Alternative Enhanced Prompt

The same file also contains an `enhanced_mode` path using `get_enhanced_disease_vectorstore()`. This path creates `./chroma_db_enhanced/WHO_chest_xray_hierarchical_path` and uses the following Chinese extraction prompt. It is active only when `enhanced_mode=true`.

```text
你是一位专业的医学影像AI助手，专门从WHO胸部X光成像指南中提取诊断信息。你的任务是基于提供的特定疾病内容进行详细分析。

关键说明：
- 你的分析必须完全基于提供的"{disorder_name}"文本内容
- 不要使用任何外部知识或做出超出提供内容的假设
- 重点关注X光影像诊断特征和表现
- 使用精确的放射学术语

疾病/病症：{disorder_name}
源文档：WHO胸部X光成像指南章节 {chapter_num}
基于：{len(retrieved_docs)} 个相关内容块

请提供基于WHO指南的诊断信息提取，包括：
1. 主要放射学特征
2. 典型X光表现
3. 鉴别诊断要点
4. 临床相关性
5. 随访建议

完整的{disorder_name}相关内容：
{context}
```

Alternative enhanced user message:

```text
查询: 基于WHO指南的{disorder_name}诊断标准和胸部X光成像发现

基于上述WHO指南内容，提供{disorder_name}的全面诊断信息分析。
```

## Stage 2: GRADE-RA Rule Assessment and Formalization

**Paper location:** Sec. 2.3, "GRADE-RA: GRADE-Based Rule Assessment".

**Script:** `enhanced_rag_to_snomed_converter_v5_cxr_nolimit.py`

**Implementation note:** The script names this module `GRADE-ER`, while the paper calls the method `GRADE-RA`. The operational content corresponds to the paper's GRADE-based rule assessment stage, but the script prompt includes one additional domain, `upgrading_factors`, beyond the paper's four downgrading domains.

**LLM role:** radiology-specialist judge for rule quality and formalization. For each CXR statement extracted in Stage 1, the LLM:

- filters non-meaningful CXR content;
- decomposes the statement into atomic propositions;
- scores GRADE-style quality domains;
- assigns a final certainty rating;
- emits a logical form;
- emits standardized knowledge-graph triples.

**Input fields consumed from Stage 1**

Only these Stage 1 fields are formalized:

- `radiological_features.data.primary_signs`
- `radiological_features.data.secondary_signs`
- `radiological_features.data.distribution_patterns`
- `typical_appearance.data`
- `atypical_presentations.data`, only if `is_present=true`

Clinical correlation, differential diagnosis, and follow-up recommendations are retained in Stage 1 output but are not converted into CXR GRADE-RA rules by this v5 script.

**Generation API parameters**

- Chat model: `deepseek-ai/DeepSeek-R1`
- Base URL: `https://api.siliconflow.cn/v1`
- Endpoint: `/chat/completions`
- API key: `SILICONFLOW_API_KEY` environment variable should be used for reproducible runs.
- `stream`: `false`
- `max_tokens`: `16384`
- `temperature`: `0.1`
- `top_p`: `0.9`
- `response_format`: `{"type": "json_object"}`
- `enable_thinking`: `true`
- `thinking_budget`: `32768`
- Retry policy: 3 attempts.
- Timeout: 120 seconds.

**GRADE-RA scoring domains in the paper**

The paper describes four downgrading domains:

- Extraction Bias: faithfulness to the verbatim source snippet.
- Inconsistency: conflicts with other rules for the same disease.
- Indirectness: missing acquisition, positional, or clinical qualifiers needed for CXR interpretation.
- Imprecision: hedged or vague language that weakens matchability.

Each domain is scored in `{-2, -1, 0}`. The paper maps the total score deterministically:

```text
S >= 0  -> High
S = -1  -> Moderate
S = -2  -> Low
S <= -3 -> Very Low
```

The v5 implementation prompt also asks for:

- Upgrading Factors: pathognomonic signs, highly specific imaging patterns, or emphasized key criteria, scored in `{0, +1, +2}`.

The v5 code preserves all certainty levels in the output. To reproduce the paper's final guideline corpus `G`, filter the final output to `High` and `Moderate`.

### Stage 2 Exact System Prompt

Source: `enhanced_rag_to_snomed_converter_v5_cxr_nolimit.py`, `CXRGradeErRuleProcessor.get_cxr_grade_er_prompt`.

```text
You are a Clinical Radiologist and Medical Informatics Specialist, an expert in chest X-ray interpretation, medical terminology, and evidence-based radiology. Your task is to receive a chest X-ray imaging statement from WHO guidelines and perform rigorous assessment, decomposition, and formalization with emphasis on radiological precision. Your output must be a single, valid JSON object using standardized English medical terminology.
```

### Stage 2 Exact User Prompt

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

### Stage 2 Output Object

The script parses the LLM JSON into:

```json
{
  "rule_id": "CXR_RULE_<DISEASE>_<FIELD>",
  "disease_name": "<disease>",
  "source_chapter": "<WHO chapter>",
  "cxr_field_type": "radiological_features | typical_appearance | atypical_presentations",
  "subfield_type": "primary_signs | secondary_signs | distribution_patterns | general",
  "original_statement": "<input statement>",
  "atomic_propositions": ["..."],
  "grade_er_assessment": {
    "extraction_bias": {"score": 0, "justification": "..."},
    "inconsistency": {"score": 0, "justification": "..."},
    "indirectness": {"score": 0, "justification": "..."},
    "imprecision": {"score": -1, "justification": "..."},
    "upgrading_factors": {"score": 1, "justification": "..."}
  },
  "final_certainty_rating": "High | Moderate | Low | Very Low",
  "formalization": {
    "logical_form": "(p1 ∧ p2) → p3",
    "knowledge_graph_triples": [
      {
        "head_entity": "chest x ray",
        "relationship": "has radiological sign",
        "tail_entity": "specific finding"
      }
    ]
  },
  "llm_thinking_process": "<provider thinking field, if returned>"
}
```

## Reproducibility Checklist

To reproduce the first two construction stages:

1. Convert the WHO manual to markdown at `chest_x_ray_imaging/PDFDocling/chest_x_ray_imaging.md`.
2. Run Stage 1 with `who_chest_xray_rag_chapter_based_en_v4.py` using:
   - embedding model `Qwen/Qwen3-Embedding-8B`;
   - chat model `deepseek-ai/DeepSeek-R1`;
   - disease-specific metadata filtering;
   - the Stage 1 prompt disclosed above.
3. Save the Stage 1 JSON output containing `AI Response.extraction_summary`.
4. Run Stage 2 with:

```bash
python enhanced_rag_to_snomed_converter_v5_cxr_nolimit.py <stage1_output.json> -o <output_dir>
```

5. For the paper's final rule corpus `G`, filter the v5 output to `final_certainty_rating in {"High", "Moderate"}`.

## Implementation Details to Disclose in the Paper or Rebuttal

- **Judge model:** `deepseek-ai/DeepSeek-R1` via SiliconFlow.
- **Embedding model:** `Qwen/Qwen3-Embedding-8B`.
- **Stage 1 prompt:** the exact schema-constrained extraction prompt above.
- **Stage 2 prompt:** the exact GRADE-RA/GRADE-ER prompt above.
- **Stage 2 scores:** extraction bias, inconsistency, indirectness, imprecision, and implementation-specific upgrading factors.
- **Output enforcement:** Stage 2 uses `response_format={"type": "json_object"}`; Stage 1 parses raw JSON or fenced JSON.
- **Final certainty:** the paper describes deterministic mapping from score total to grade. The current v5 script asks the LLM to output the final grade according to the disclosed mapping and then stores it. For strict reproducibility, add a post-processing assertion that recomputes the final grade from the returned domain scores.
- **No secret disclosure:** do not publish API keys; report only model/provider, endpoint family, and decoding parameters.

## Suggested Rebuttal Text

We agree that a benchmark paper should disclose the LLM judge. We have added the exact model, decoding parameters, inputs, output schema, and prompts used in the first two construction stages. Specifically, Prior-Guided Content RAG uses DeepSeek-R1 to extract schema-constrained rules from metadata-filtered WHO disease sections, with prompts restricting the model to the retrieved guideline text only. GRADE-RA uses DeepSeek-R1 as a radiology-focused judge to score extraction bias, inconsistency, indirectness, and imprecision, and to produce atomic propositions, logical forms, and knowledge-graph triples. The final paper rule corpus is obtained by retaining High/Moderate rules. We will include the full prompts and scoring rubric in the appendix/repository to ensure reproducibility.
