#!/usr/bin/env python3
"""Minimal reproducibility smoke test for the stage3 LLM API.

This test uses an OpenAI-compatible chat-completions endpoint, such as a
locally served vLLM endpoint. It does not load model weights in this repository.
"""

import argparse
import json
import sys

import requests


def extract_json_array(text: str):
    text = text.strip().replace("```json", "").replace("```", "")
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON array found in model response")
    return json.loads(text[start:end + 1])


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage3 LLM API smoke test")
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--skip-health", action="store_true")
    args = parser.parse_args()

    server_url = args.server_url.rstrip("/")

    if not args.skip_health:
        health = requests.get(f"{server_url}/health", timeout=10)
        health.raise_for_status()

    prompt = """You are a radiology rule-matching assistant.

MEDICAL REPORT:
FINDINGS: Mild right basilar opacity is present. No pleural effusion.
IMPRESSION: Mild right basilar atelectatic opacity.

RULES:
Rule ID: TEST_RULE_ATELECTASIS_OPACITY
Disease: Atelectasis
Statement: Linear or basilar pulmonary opacity can be compatible with atelectasis.

Return a JSON array only, using this schema:
[
  {
    "rule_id": "TEST_RULE_ATELECTASIS_OPACITY",
    "original_statement": "...",
    "semantic_evidence": "...",
    "confidence_rating": 0.0,
    "detailed_reasoning": "..."
  }
]"""

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 0.9,
        "max_tokens": 512,
        "stream": False,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    }

    response = requests.post(
        f"{server_url}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=args.timeout,
    )
    response.raise_for_status()

    content = response.json()["choices"][0]["message"]["content"]
    parsed = extract_json_array(content)

    if not parsed or parsed[0].get("rule_id") != "TEST_RULE_ATELECTASIS_OPACITY":
        raise AssertionError(f"Unexpected response payload: {parsed!r}")

    print(json.dumps({"status": "ok", "items": len(parsed)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
