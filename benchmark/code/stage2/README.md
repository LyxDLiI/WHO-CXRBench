# Stage 2: GRADE-RA Assessment

`run_stage2.py` consumes Stage 1 JSON output, extracts CXR fields, asks the LLM judge for GRADE-RA domain scores and formalization, then recomputes the final certainty rating from the four paper-defined domains.

Example:

```bash
cd benchmark/code
export SILICONFLOW_API_KEY="..."
python stage2/run_stage2.py --input outputs/stage1_pectus.json --output-dir outputs/stage2_pectus --max-fields 1
```

The paper-facing corpus is written to `guideline_rule_corpus_high_moderate.json`.
