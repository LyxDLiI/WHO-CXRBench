# WHO-Grounded CXR Rule Retrieval Dataset (Preview)

This repository provides a **guideline-grounded dataset** for **fine-grained diagnostic rule retrieval and alignment** on chest radiography, built around WHO chest X-ray (CXR) guidelines.

## What is released (current)
Due to the **MIMIC-CXR data-use agreement**, we cannot redistribute the full underlying clinical data at this moment. We are **actively applying to release the complete dataset via PhysioNet**.

For now, this repo includes:
- **Rule corpus** (machine-readable WHO diagnostic rule statements)
- **Original WHO source files** used to derive the rules
- **100 de-identified samples** (JSON/JSONL) to illustrate the data format and enable quick prototyping

## Data format (per sample)

```json
{
  "split": "train",
  "idx": 152028,

  "question": "Provide a catalogue of all anatomical findings and diseases seen.",
  "semantic_type": "query",
  "content_type": "attribute",

  "template": "Provide a catalogue of all ${category_1} and ${category_2} seen.",
  "template_program": "program_26",
  "template_arguments": { ... },

  "answer": ["...", "..."],

  "global_who_rules": [
    {
      "rule_id": "...",
      "disease": "...",
      "original_statement": "...",
      "source": "CXR_RULE",
      "matched_answer": "...",
      "mapped_who_disease": "..."
    }
  ],

  "personal_who_rules": [
    {
      "rule_id": "...",
      "original_statement": "...",
      "confidence_rating": 0.8,
      "detailed_reasoning": "..."
    }
  ],

  "medical_report_found": true
}
```

## Status
Full release is pending PhysioNet approval. Updates will be posted in this repository.

## License
- WHO source files: follow the original WHO terms.
- Released annotations/examples in this repo: see `LICENSE` (or add one before release).
