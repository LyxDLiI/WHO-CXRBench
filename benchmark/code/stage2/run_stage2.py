#!/usr/bin/env python3
"""Stage 2: GRADE-RA assessment and CXR rule formalization.

This open-source version consumes Stage 1 JSON records, extracts CXR rule
statements, asks an LLM judge for domain-level GRADE-RA scores and structured
formalization, and recomputes the final certainty rating deterministically from
the paper's four downgrading domains.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1"
DEFAULT_MAX_TOKENS = 16384
DEFAULT_TEMPERATURE = 0.1
DEFAULT_TOP_P = 0.9
GRADE_RA_DOMAINS = ("extraction_bias", "inconsistency", "indirectness", "imprecision")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_stage1_records(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "records" in data and isinstance(data["records"], list):
        return data["records"]
    raise ValueError(f"Unsupported Stage 1 input shape: {path}")


def stringify_statement(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def extract_cxr_fields(ai_response: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    extraction_summary = ai_response.get("extraction_summary", {})
    cxr_fields: Dict[str, Dict[str, str]] = {}

    radiological_features = extraction_summary.get("radiological_features", {})
    if isinstance(radiological_features, dict) and radiological_features.get("is_present", False):
        data = radiological_features.get("data", {})
        if isinstance(data, dict):
            for subfield in ("primary_signs", "secondary_signs", "distribution_patterns"):
                content = stringify_statement(data.get(subfield))
                if content:
                    cxr_fields[f"radiological_features_{subfield}"] = {
                        "field_type": "radiological_features",
                        "subfield_type": subfield,
                        "content": content,
                    }
        else:
            content = stringify_statement(data)
            if content:
                cxr_fields["radiological_features_general"] = {
                    "field_type": "radiological_features",
                    "subfield_type": "general",
                    "content": content,
                }

    typical_appearance = extraction_summary.get("typical_appearance", {})
    if isinstance(typical_appearance, dict) and typical_appearance.get("is_present", False):
        content = stringify_statement(typical_appearance.get("data"))
        if content:
            cxr_fields["typical_appearance"] = {
                "field_type": "typical_appearance",
                "subfield_type": "general",
                "content": content,
            }

    atypical_presentations = extraction_summary.get("atypical_presentations", {})
    if isinstance(atypical_presentations, dict) and atypical_presentations.get("is_present", False):
        content = stringify_statement(atypical_presentations.get("data"))
        if content:
            cxr_fields["atypical_presentations"] = {
                "field_type": "atypical_presentations",
                "subfield_type": "general",
                "content": content,
            }

    return cxr_fields


def build_grade_ra_prompt(
    *,
    source_chapter: str,
    statement: str,
    cxr_field_type: str,
    subfield_type: str,
    full_ai_response: str,
) -> List[Dict[str, str]]:
    system_prompt = (
        "You are a Clinical Radiologist and Medical Informatics Specialist, "
        "an expert in chest X-ray interpretation, medical terminology, and "
        "evidence-based radiology. Your task is to receive a chest X-ray "
        "imaging statement from WHO guidelines and perform rigorous "
        "GRADE-RA assessment, decomposition, and formalization with emphasis "
        "on radiological precision. Your output must be a single, valid JSON "
        "object using standardized English medical terminology."
    )

    user_prompt = f"""
# ROLE
You are a Clinical Radiologist and Medical Informatics Specialist, an expert in chest X-ray interpretation, medical terminology, logic, and evidence-based radiology. You specialize in converting radiological statements into machine-readable knowledge graphs using standardized English medical terminology.

# CONTEXT
You are building a high-fidelity knowledge base for an advanced medical AI focused on chest X-ray interpretation. The goal is to generate precise radiological reasoning paths and a medical logic consistency benchmark for CXR interpretation. You will use the paper's GRADE-RA framework to assess operational quality of radiological rule text.

# INPUT
You will receive a JSON object containing a single chest X-ray imaging statement for analysis:
{{
  "source_chapter": "{source_chapter}",
  "cxr_field_type": "{cxr_field_type}",
  "subfield_type": "{subfield_type}",
  "source_statement": "{statement}",
  "full_ai_response_context": "{full_ai_response}"
}}

# CXR-SPECIFIC CONTENT FILTERING
Before processing, evaluate if the `source_statement` contains meaningful radiological information. If the statement indicates "No content available", "null", "Not addressed in provided blocks", "Not specified", "Not mentioned", "Not described beyond imaging findings", or similar absence language, provide minimal processing and assign domain scores that lead to "Very Low".

# TASK: Execute the following four steps.

## STEP 1: Radiological Statement Decomposition
Only proceed with detailed decomposition if the statement contains meaningful radiological information. Decompose the `source_statement` into atomic propositions focused on:
- specific radiological signs or features;
- anatomical locations and distributions;
- X-ray appearance characteristics;
- radiological patterns or configurations.

Each proposition should be a verifiable radiological fact using precise medical terminology.

## STEP 2: GRADE-RA Domain Assessment
Assume initial certainty is "High" because the source is a WHO guideline section. Score exactly four downgrading domains. Each score MUST be one of -2, -1, or 0.

- **Extraction Bias**: Does the extracted statement faithfully preserve the verbatim source meaning and radiological terminology?
  - Score 0 for faithful extraction, -1 for minor distortion, -2 for severe distortion.
- **Inconsistency**: Does the statement conflict with other CXR findings or rules in `full_ai_response_context`?
  - Score 0 for no conflict, -1 for possible tension, -2 for clear radiological contradiction.
- **Indirectness**: Are acquisition qualifiers, positional qualifiers, anatomic qualifiers, or clinical qualifiers needed for CXR interpretation missing?
  - Score 0 for sufficient context, -1 for some missing context, -2 for critical missing context.
- **Imprecision**: Is the rule too vague, hedged, or anatomically ambiguous to be reliably matched?
  - Score 0 for precise wording, -1 for moderate imprecision, -2 for high imprecision or absent radiological content.

## STEP 3: Final Certainty Rating
The final certainty is computed from the four domain scores:
- Total Score >= 0: "High"
- Total Score = -1: "Moderate"
- Total Score = -2: "Low"
- Total Score <= -3: "Very Low"

You may include `final_certainty_rating`, but downstream code will recompute it deterministically from the four scores above.

## STEP 4: CXR Knowledge Graph Formalization
Only perform detailed formalization for radiologically meaningful statements.

1. **Logical Form**: Combine atomic propositions using symbolic or textual logical operators with identifiers p1, p2, etc.
2. **Knowledge Graph Triples**: For each atomic proposition, create triples using standardized English medical terminology only.
   - Use precise anatomical terms such as "chest wall", "mediastinum", "lung field", "cardiac silhouette", "diaphragm", "pleural space".
   - Use standardized radiological relationships such as "has radiological sign", "demonstrates feature", "located in", "shows", "exhibits", "characterized by", "obscures", "displaces".
   - Use specific imaging descriptors such as "opacity", "lucency", "consolidation", "displacement", "enlargement", "straightening", "configuration".

# OUTPUT FORMAT
Return a single valid JSON object and nothing else:
{{
  "original_statement": "{statement}",
  "cxr_field_analysis": {{
    "field_type": "{cxr_field_type}",
    "subfield_type": "{subfield_type}",
    "radiological_significance": "high/moderate/low"
  }},
  "atomic_propositions": [
    "specific radiological proposition 1 using precise medical terms"
  ],
  "grade_ra_assessment": {{
    "extraction_bias": {{ "score": 0, "justification": "Faithfully preserves the source meaning." }},
    "inconsistency": {{ "score": 0, "justification": "No conflict with the provided context." }},
    "indirectness": {{ "score": 0, "justification": "Sufficient CXR context is preserved." }},
    "imprecision": {{ "score": 0, "justification": "Terminology is sufficiently precise." }}
  }},
  "final_certainty_rating": "High",
  "formalization": {{
    "logical_form": "p1",
    "knowledge_graph_triples": [
      {{
        "atomic_proposition": "radiological proposition text",
        "triples": [
          {{"head_entity": "chest x ray", "relationship": "has radiological sign", "tail_entity": "specific finding"}}
        ]
      }}
    ]
  }}
}}
"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def call_siliconflow_chat(
    messages: List[Dict[str, str]],
    *,
    api_key: str,
    base_url: str,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    max_retries: int,
    enable_thinking: bool,
    thinking_budget: int,
) -> Dict[str, Any]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "response_format": {"type": "json_object"},
    }
    if enable_thinking:
        payload["enable_thinking"] = True
        payload["thinking_budget"] = thinking_budget

    last_error: Optional[BaseException] = None
    for attempt in range(max_retries):
        try:
            if attempt:
                time.sleep(min(2**attempt, 30))
            response = requests.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
            if attempt == max_retries - 1:
                raise
    raise RuntimeError("SiliconFlow call failed") from last_error


def parse_json_content(content: str) -> Dict[str, Any]:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if fenced:
        content = fenced.group(1)
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object from LLM, got {type(parsed)}")
    return parsed


def extract_message_parts(response: Dict[str, Any]) -> Tuple[str, str]:
    try:
        message = response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected chat completion response shape: {response}") from exc
    return message.get("content", "{}"), message.get("thinking", "")


def score_to_int(value: Any) -> int:
    if isinstance(value, int):
        score = value
    elif isinstance(value, str):
        score = int(value.strip())
    else:
        raise ValueError(f"Invalid GRADE-RA score: {value!r}")
    if score not in {-2, -1, 0}:
        raise ValueError(f"GRADE-RA downgrading score must be -2, -1, or 0, got {score}")
    return score


def canonical_assessment(parsed: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    assessment = parsed.get("grade_ra_assessment") or parsed.get("grade_er_assessment")
    if not isinstance(assessment, dict):
        raise ValueError("LLM output is missing grade_ra_assessment")

    canonical: Dict[str, Dict[str, Any]] = {}
    for domain in GRADE_RA_DOMAINS:
        item = assessment.get(domain)
        if not isinstance(item, dict):
            raise ValueError(f"LLM output is missing assessment domain: {domain}")
        canonical[domain] = {
            "score": score_to_int(item.get("score")),
            "justification": stringify_statement(item.get("justification")),
        }
    return canonical


def grade_from_scores(assessment: Dict[str, Dict[str, Any]]) -> Tuple[int, str]:
    total = sum(int(assessment[domain]["score"]) for domain in GRADE_RA_DOMAINS)
    if total >= 0:
        return total, "High"
    if total == -1:
        return total, "Moderate"
    if total == -2:
        return total, "Low"
    return total, "Very Low"


def flatten_kg_triples(formalization: Dict[str, Any]) -> List[Dict[str, str]]:
    flattened: List[Dict[str, str]] = []
    for item in formalization.get("knowledge_graph_triples", []):
        if not isinstance(item, dict):
            continue
        for triple in item.get("triples", []):
            if isinstance(triple, dict):
                flattened.append(
                    {
                        "head_entity": stringify_statement(triple.get("head_entity")),
                        "relationship": stringify_statement(triple.get("relationship")),
                        "tail_entity": stringify_statement(triple.get("tail_entity")),
                    }
                )
    return flattened


def rule_id(disease_name: str, field_key: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", disease_name).strip("_").upper()
    field = re.sub(r"[^A-Za-z0-9]+", "_", field_key).strip("_").upper()
    return f"CXR_RULE_{normalized}_{field}"


def process_record(
    *,
    record: Dict[str, Any],
    args: argparse.Namespace,
    api_key: Optional[str],
    field_counter: List[int],
) -> List[Dict[str, Any]]:
    disease_name = stringify_statement(record.get("disorder_name") or record.get("disease_name"))
    source_chapter = stringify_statement(record.get("chapter_mapping") or record.get("chapter_number") or "Unknown")
    ai_response = record.get("AI Response") or record.get("ai_response")
    if not isinstance(ai_response, dict):
        print(f"[stage2] skipped {disease_name}: Stage 1 AI Response is not a JSON object", file=sys.stderr)
        return []

    full_ai_response = json.dumps(ai_response, ensure_ascii=False, sort_keys=True)
    cxr_fields = extract_cxr_fields(ai_response)
    rules: List[Dict[str, Any]] = []

    for field_key, field_info in cxr_fields.items():
        if args.max_fields is not None and field_counter[0] >= args.max_fields:
            break
        field_counter[0] += 1

        messages = build_grade_ra_prompt(
            source_chapter=source_chapter,
            statement=field_info["content"],
            cxr_field_type=field_info["field_type"],
            subfield_type=field_info["subfield_type"],
            full_ai_response=full_ai_response,
        )

        if args.dry_run:
            parsed = {
                "dry_run": True,
                "prompt_messages": messages,
                "grade_ra_assessment": {
                    domain: {"score": 0, "justification": "Dry run placeholder."}
                    for domain in GRADE_RA_DOMAINS
                },
                "formalization": {"logical_form": "p1", "knowledge_graph_triples": []},
                "atomic_propositions": [field_info["content"]],
            }
            thinking = ""
            raw_content = json.dumps(parsed, ensure_ascii=False)
        else:
            assert api_key is not None
            response = call_siliconflow_chat(
                messages,
                api_key=api_key,
                base_url=args.base_url,
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                max_retries=args.max_retries,
                enable_thinking=args.enable_thinking,
                thinking_budget=args.thinking_budget,
            )
            raw_content, thinking = extract_message_parts(response)
            parsed = parse_json_content(raw_content)

        assessment = canonical_assessment(parsed)
        total_score, deterministic_rating = grade_from_scores(assessment)
        formalization = parsed.get("formalization") if isinstance(parsed.get("formalization"), dict) else {}
        kg_triples = flatten_kg_triples(formalization)

        rules.append(
            {
                "rule_id": rule_id(disease_name, field_key),
                "disease_name": disease_name,
                "source_chapter": source_chapter,
                "cxr_field_type": field_info["field_type"],
                "subfield_type": field_info["subfield_type"],
                "original_statement": field_info["content"],
                "atomic_propositions": parsed.get("atomic_propositions", []),
                "grade_ra_assessment": assessment,
                "grade_ra_total_score": total_score,
                "final_certainty_rating": deterministic_rating,
                "llm_reported_certainty_rating": parsed.get("final_certainty_rating"),
                "formalization": {
                    "logical_form": stringify_statement(formalization.get("logical_form")),
                    "knowledge_graph_triples": kg_triples,
                },
                "raw_llm_content": raw_content,
                "llm_thinking_process": thinking,
                "model": args.model,
                "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
        )
        print(f"[stage2] formalized {disease_name} / {field_key} -> {deterministic_rating}", file=sys.stderr)

    return rules


def generate_summary(rules: List[Dict[str, Any]], diseases: Dict[str, Any]) -> Dict[str, Any]:
    distribution: Dict[str, int] = {}
    for rule in rules:
        rating = rule["final_certainty_rating"]
        distribution[rating] = distribution.get(rating, 0) + 1
    return {
        "total_diseases_processed": len(diseases),
        "total_cxr_rules_formalized": len(rules),
        "overall_certainty_distribution": distribution,
    }


def run_stage2(args: argparse.Namespace) -> Dict[str, Any]:
    api_key = os.environ.get("SILICONFLOW_API_KEY")
    if not api_key and not args.dry_run:
        raise RuntimeError("Set SILICONFLOW_API_KEY before running Stage 2 with API calls.")

    records = load_stage1_records(args.input)
    if args.limit_diseases is not None:
        records = records[: args.limit_diseases]

    diseases: Dict[str, Any] = {}
    all_rules: List[Dict[str, Any]] = []
    field_counter = [0]

    for record in records:
        disease_name = stringify_statement(record.get("disorder_name") or record.get("disease_name"))
        rules = process_record(record=record, args=args, api_key=api_key, field_counter=field_counter)
        if rules:
            diseases[disease_name] = {
                "disease_name": disease_name,
                "source_chapter": stringify_statement(record.get("chapter_mapping") or record.get("chapter_number")),
                "total_cxr_rules_formalized": len(rules),
                "formalized_cxr_rules": rules,
                "certainty_distribution": generate_summary(rules, {disease_name: True})[
                    "overall_certainty_distribution"
                ],
                "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
            all_rules.extend(rules)

    return {
        "summary": generate_summary(all_rules, diseases),
        "diseases": diseases,
        "all_rules": all_rules,
        "guideline_rule_corpus_high_moderate": [
            rule for rule in all_rules if rule["final_certainty_rating"] in {"High", "Moderate"}
        ],
        "metadata": {
            "stage": "stage2_grade_ra",
            "model": args.model,
            "grade_ra_domains": list(GRADE_RA_DOMAINS),
            "final_certainty_is_deterministically_recomputed": True,
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    root = project_root()
    parser = argparse.ArgumentParser(description="Stage 2 WHO-CXRBench GRADE-RA formalization")
    parser.add_argument("--input", type=Path, required=True, help="Stage 1 JSON output.")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs" / "stage2")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--enable-thinking", action="store_true", default=False)
    parser.add_argument("--thinking-budget", type=int, default=32768)
    parser.add_argument("--limit-diseases", type=int, default=None)
    parser.add_argument("--max-fields", type=int, default=None, help="Global field limit for smoke tests.")
    parser.add_argument("--dry-run", action="store_true", help="Build deterministic placeholder output without API calls.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = run_stage2(args)

    final_path = args.output_dir / "stage2_grade_ra_all_rules.json"
    corpus_path = args.output_dir / "guideline_rule_corpus_high_moderate.json"
    final_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    corpus_path.write_text(
        json.dumps(result["guideline_rule_corpus_high_moderate"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[stage2] wrote full output to {final_path}")
    print(f"[stage2] wrote High/Moderate corpus to {corpus_path}")


if __name__ == "__main__":
    main()
