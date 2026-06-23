# Stage 1: Prior-Guided Content RAG

`run_stage1.py` extracts one disease-specific WHO guideline section and calls the LLM with the disclosed schema-constrained prompt. The output is a Stage 2-compatible JSON list.

Example:

```bash
cd code
export SILICONFLOW_API_KEY="..."
python stage1/run_stage1.py --disease "Pectus excavatum" --output outputs/stage1_pectus.json
```

Use `--dry-run` to validate prompt construction without an API call.
