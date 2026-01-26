# WHO-Grounded CXR Rule Retrieval Dataset (Preview)

This repository provides a **guideline-grounded dataset** for **fine-grained diagnostic rule retrieval and alignment** on chest radiography, built around WHO chest X-ray (CXR) guidelines.

## What is released (current)
Due to the **MIMIC-CXR data-use agreement**, we cannot redistribute the full underlying clinical data at this moment. We are **actively applying to release the complete dataset via PhysioNet**.

For now, this repo includes:
- **Rule corpus** (machine-readable WHO diagnostic rule statements)
- **Original WHO source files** used to derive the rules
- **100 de-identified samples** (JSON/JSONL) to illustrate the data format and enable quick prototyping

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
      "confidence_rating": 0.8,
      "detailed_reasoning": "..."
    }
  ]
}
```

See `data/examples/` for the 100-sample preview and field definitions.

## Intended use
- Rule retrieval / ranking over `global_who_rules`
- Confidence modeling / calibration using `confidence_rating`
- Explainable alignment analysis via `detailed_reasoning`

## Status
Full release is pending PhysioNet approval. Updates will be posted in this repository.

## License
- WHO source files: follow the original WHO terms.
- Released annotations/examples in this repo: see `LICENSE` (or add one before release).
