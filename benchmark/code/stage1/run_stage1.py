#!/usr/bin/env python3
"""Stage 1: Prior-guided WHO CXR guideline rule extraction.

This script is the open-source, reproducible version of the Stage 1 pipeline.
It extracts disease-specific sections from the WHO chest radiography markdown
and uses a schema-constrained LLM prompt to produce structured CXR rule fields.

Credentials are intentionally read from SILICONFLOW_API_KEY and are never stored
in config files or outputs.
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
DEFAULT_MAX_TOKENS = 2000
DEFAULT_TEMPERATURE = 0.3
DEFAULT_TOP_P = 0.8


TITLE_ALIASES = {
    "Lymphangioleiomyomatosis (LAM)": "LAM",
    "Langerhans cell histiocytosis (LCH)": "LCH",
    "Acute hypersensitivity pneumonitis": "Acute HP",
    "Sub-acute hypersensitivity pneumonitis": "Sub-acute HP",
    "Chronic hypersensitivity pneumonitis": "Chronic HP",
    "The solitary pulmonary nodule": "The solitary pulmonary nodule The solitary pulmonary nodule",
    "Further imaging": "Further imaging Further imaging",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def relative_project_path(path: Path) -> str:
    """Return a portable path for provenance metadata."""
    return Path(os.path.relpath(path.resolve(), project_root())).as_posix()


def normalize_title(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value


def load_disease_list(path: Path) -> List[str]:
    diseases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            diseases.append(line)
    return diseases


def iter_h2_headings(lines: List[str]) -> Iterable[Tuple[int, str]]:
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("## ") and not stripped.startswith("### "):
            yield idx, stripped[3:].strip()


def heading_matches(title: str, disease: str) -> bool:
    title_norm = normalize_title(title)
    disease_norm = normalize_title(disease)
    alias = TITLE_ALIASES.get(disease)
    alias_norm = normalize_title(alias) if alias else None

    if title_norm == disease_norm or (alias_norm and title_norm == alias_norm):
        return True

    # Docling duplicated a small number of headings, e.g.
    # "The solitary pulmonary nodule The solitary pulmonary nodule".
    if title_norm == f"{disease_norm} {disease_norm}":
        return True
    return False


def infer_chapter(lines: List[str], start_idx: int) -> str:
    for idx in range(start_idx, -1, -1):
        text = lines[idx].strip()
        match = re.search(r"\bCH(?:AP|PA)TER\s+(\d+)\b", text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return "Unknown"


def extract_disease_section(markdown_path: Path, disease: str) -> Dict[str, Any]:
    lines = markdown_path.read_text(encoding="utf-8").splitlines(keepends=True)
    headings = list(iter_h2_headings(lines))

    start_idx: Optional[int] = None
    matched_title: Optional[str] = None
    for idx, title in headings:
        if heading_matches(title, disease):
            start_idx = idx
            matched_title = title
            break

    if start_idx is None or matched_title is None:
        raise ValueError(f"Disease section not found in markdown: {disease}")

    end_idx = len(lines)
    for idx, _ in headings:
        if idx > start_idx:
            end_idx = idx
            break

    content = "".join(lines[start_idx:end_idx]).strip()
    chapter = infer_chapter(lines, start_idx)
    return {
        "disease_name": disease,
        "matched_section_title": matched_title,
        "chapter": chapter,
        "start_line": start_idx + 1,
        "end_line": end_idx,
        "content": content,
        "content_length": len(content),
        "source": relative_project_path(markdown_path),
    }


def build_stage1_prompt(section: Dict[str, Any]) -> List[Dict[str, str]]:
    disease = section["disease_name"]
    chapter = section["chapter"]
    metadata_info = (
        f"Disease: {disease}\n"
        f"Matched Section Title: {section['matched_section_title']}\n"
        f"Chapter: {chapter}\n"
        f"Content Lines: {section['start_line']}-{section['end_line']}\n"
        f"Content Length: {section['content_length']} characters"
    )

    system_prompt = f"""You are an expert AI assistant specializing in medical information extraction from clinical guidelines. Your task is to act as a professional radiologist and meticulously analyze the provided section on a specific disease from the WHO Chest X-ray Imaging Guideline.

CRITICAL INSTRUCTIONS:
- Your analysis MUST be based EXCLUSIVELY on the provided text for "{disease}".
- DO NOT use any external knowledge or make assumptions beyond the provided content.
- Your goal is to READ, COMPREHEND, and SYNTHESIZE the information into a structured JSON format.
- For each field in the JSON schema, you must determine if the information is present in the text.
- If information for a field is present, synthesize it concisely using precise radiological terminology from the text and populate the "data" sub-field.
- If information for a field is NOT present in the text, you MUST set the "is_present" flag to false and the "data" field to null.
- Adhere STRICTLY to the JSON schema provided below. Do not add, remove, or rename any keys. The final output must be a single, valid JSON object and nothing else.

CONTENT METADATA:
{metadata_info}

DISEASE/CONDITION TO ANALYZE: {disease}

JSON OUTPUT SCHEMA:
{{
  "disease_name": "{disease}",
  "source_document": "WHO Chest X-ray Imaging Guideline Chapter {chapter}",
  "extraction_summary": {{
    "radiological_features": {{
      "is_present": true,
      "data": {{
        "primary_signs": "Synthesized description of primary radiological signs.",
        "secondary_signs": "Synthesized description of secondary signs and other findings.",
        "distribution_patterns": "Synthesized description of distribution and anatomical locations."
      }}
    }},
    "typical_appearance": {{
      "is_present": true,
      "data": "A synthesized summary of the typical overall chest X-ray appearance."
    }},
    "atypical_presentations": {{
      "is_present": true,
      "data": "A synthesized summary of any described variations or atypical presentations."
    }},
    "clinical_correlation": {{
      "is_present": true,
      "data": {{
        "patient_population": "Synthesized description of typical patient populations.",
        "symptom_correlation": "Synthesized description of correlations with clinical symptoms.",
        "risk_factors": "Synthesized description of predisposing conditions or risk factors."
      }}
    }},
    "differential_diagnosis": {{
      "is_present": true,
      "data": "A synthesized list or description of conditions to be considered in the differential diagnosis, based on the text."
    }},
    "follow_up_recommendations": {{
      "is_present": true,
      "data": "A synthesized summary of any recommendations for follow-up or further imaging (e.g., CT)."
    }}
  }}
}}

COMPLETE SECTION CONTENT FOR {disease}:
{section["content"]}

Now, perform the analysis and generate the JSON output."""

    query = f"WHO guideline-based diagnostic criteria and chest X-ray imaging findings for {disease}"
    user_prompt = f"""QUERY: {query}

STRICT REQUIREMENTS:
- Analyze ONLY the provided WHO {disease} section content
- Extract diagnostic standards directly from this disease-specific guideline content
- Use English language exclusively
- Quote specific content when making statements
- Focus on evidence-based chest X-ray diagnostic criteria

Provide comprehensive diagnostic standards for {disease} based strictly on the WHO guideline content provided above."""

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
) -> Dict[str, Any]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }

    last_error: Optional[BaseException] = None
    for attempt in range(max_retries):
        try:
            if attempt:
                time.sleep(min(2**attempt, 30))
            response = requests.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
                timeout=(60, 600),
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
            if attempt == max_retries - 1:
                raise
    raise RuntimeError("SiliconFlow call failed") from last_error


def parse_json_content(content: str) -> Any:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if fenced:
        content = fenced.group(1)
    return json.loads(content)


def extract_completion_content(response: Dict[str, Any]) -> str:
    try:
        return response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected chat completion response shape: {response}") from exc


def make_dry_run_ai_response(section: Dict[str, Any], messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """Return a minimal valid Stage 1 schema for no-API wiring tests."""
    disease = section["disease_name"]
    chapter = section["chapter"]
    snippet = re.sub(r"\s+", " ", section["content"]).strip()[:500]
    return {
        "dry_run": True,
        "disease_name": disease,
        "source_document": f"WHO Chest X-ray Imaging Guideline Chapter {chapter}",
        "extraction_summary": {
            "radiological_features": {
                "is_present": True,
                "data": {
                    "primary_signs": f"Dry-run placeholder extracted from source section: {snippet}",
                    "secondary_signs": None,
                    "distribution_patterns": None,
                },
            },
            "typical_appearance": {"is_present": False, "data": None},
            "atypical_presentations": {"is_present": False, "data": None},
            "clinical_correlation": {
                "is_present": False,
                "data": {
                    "patient_population": None,
                    "symptom_correlation": None,
                    "risk_factors": None,
                },
            },
            "differential_diagnosis": {"is_present": False, "data": None},
            "follow_up_recommendations": {"is_present": False, "data": None},
        },
        "dry_run_prompt_messages": messages,
    }


def make_output_record(
    *,
    section: Dict[str, Any],
    query: str,
    ai_response: Any,
    raw_content: str,
    model: str,
) -> Dict[str, Any]:
    return {
        "stage": "stage1_prior_guided_content_rag",
        "disorder_name": section["disease_name"],
        "chapter_number": section["chapter"],
        "chapter_mapping": f"WHO Chest X-ray Imaging Guideline Chapter {section['chapter']}",
        "total_docs_retrieved": 1,
        "retrieval_method": "Disease-specific markdown heading extraction",
        "content_type": "Disease-Specific Section",
        "Human Message": [query],
        "AI Response": ai_response,
        "raw_llm_content": raw_content,
        "source_documents": [
            [
                {
                    "page_content": section["content"],
                    "metadata": {
                        "source": section["source"],
                        "disease_name": section["disease_name"],
                        "matched_section_title": section["matched_section_title"],
                        "chapter": section["chapter"],
                        "start_line": section["start_line"],
                        "end_line": section["end_line"],
                        "content_length": section["content_length"],
                    },
                }
            ]
        ],
        "chunking_strategy": {
            "method": "Prior-guided disease-section chunking",
            "description": "The WHO markdown heading hierarchy is used to isolate one disease section.",
            "precision": "Exact heading or documented alias match",
            "coverage": "Complete disease-specific section until the next level-2 heading",
        },
        "search_strategy": {
            "method": "Metadata-style disease heading lookup",
            "precision": "Disease-specific section retrieval",
            "fallback": "Documented title aliases for abbreviations and duplicated headings",
        },
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model": model,
        "embedding_model": "not_required_for_heading_retrieval",
    }


def run_stage1(args: argparse.Namespace) -> List[Dict[str, Any]]:
    api_key = os.environ.get("SILICONFLOW_API_KEY")
    if not api_key and not args.dry_run:
        raise RuntimeError("Set SILICONFLOW_API_KEY before running Stage 1 with API calls.")

    if args.disease:
        diseases = args.disease
    else:
        diseases = load_disease_list(args.disease_file)

    if args.limit is not None:
        diseases = diseases[: args.limit]

    results: List[Dict[str, Any]] = []
    for disease in diseases:
        section = extract_disease_section(args.markdown, disease)
        messages = build_stage1_prompt(section)
        query = f"WHO guideline-based diagnostic criteria and chest X-ray imaging findings for {disease}"

        if args.dry_run:
            ai_response: Any = make_dry_run_ai_response(section, messages)
            raw_content = json.dumps(ai_response, ensure_ascii=False)
        else:
            assert api_key is not None
            completion = call_siliconflow_chat(
                messages,
                api_key=api_key,
                base_url=args.base_url,
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                max_retries=args.max_retries,
            )
            raw_content = extract_completion_content(completion)
            try:
                ai_response = parse_json_content(raw_content)
            except json.JSONDecodeError:
                ai_response = raw_content

        results.append(
            make_output_record(
                section=section,
                query=query,
                ai_response=ai_response,
                raw_content=raw_content,
                model=args.model,
            )
        )
        print(f"[stage1] processed {disease}", file=sys.stderr)

    return results


def build_arg_parser() -> argparse.ArgumentParser:
    root = project_root()
    parser = argparse.ArgumentParser(description="Stage 1 WHO-CXRBench rule extraction")
    parser.add_argument("--markdown", type=Path, default=root / "data" / "chest_x_ray_imaging.md")
    parser.add_argument("--disease-file", type=Path, default=root / "data" / "default_diseases.txt")
    parser.add_argument("--disease", action="append", help="Disease to process. May be repeated.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of diseases from disease-file.")
    parser.add_argument("--output", type=Path, default=root / "outputs" / "stage1_rules.json")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true", help="Build prompts and outputs without API calls.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = run_stage1(args)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[stage1] wrote {len(results)} record(s) to {args.output}")


if __name__ == "__main__":
    main()
