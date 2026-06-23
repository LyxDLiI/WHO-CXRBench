# Stage3: LLM Rule Matching

Stage3 matches candidate CXR/WHO-style clinical rules against radiology report text with an LLM and writes structured JSON fields such as semantic evidence, confidence scores, and concise reasoning.

## Included Files

- `smart_rule_matcher_local.py`: rule loading, disease mapping, report loading, prompting, response parsing, and per-sample matching.
- `optimized_batch_processor_local.py`: batch processing with checkpointing and output validation.
- `run_processing_local.sh`: original local batch-processing launcher.
- `serve_vllm.sh`: minimal model-serving command for reproducibility.
- `tests/api_smoke_test.py`: small API-level reproducibility test.
- `cxr_grade_er_conversion_final_20250712_200006.json`: CXR rule file required by stage3.
- `disease_in_MIMIC_mapto_WHO.csv`: disease-name mapping used by stage3.

## LLM Details

- Model ID: `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`
- Local snapshot used in the original run: `6e8885a6ff5c1dc5201574c8fd700323f23c25fa`
- Architecture from model config: `Qwen3ForCausalLM`
- Layers: `36`
- Hidden size: `4096`
- Intermediate size: `12288`
- Attention heads: `32`
- Key/value heads: `8`
- Vocabulary size: `151936`
- Model dtype: `bfloat16`
- Max position embeddings in config: `131072`
- RoPE scaling: `yarn`, factor `4.0`, original max position embeddings `32768`

## vLLM Serving Parameters

The original local run used vLLM `0.8.5.post1` with an OpenAI-compatible endpoint:

```bash
vllm serve deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  --trust-remote-code
```

Runtime settings observed in the original vLLM log:

- Tensor parallel size: `4`
- Pipeline parallel size: `1`
- Data parallel size: `1`
- Max model length: `32768`
- Quantization: `None`
- Load format: `auto`
- Tokenizer mode: `auto`
- Device: `cuda`
- GPU memory utilization: `0.9`
- Swap space: `4 GB`
- Chunked prefill: enabled
- Prefix caching: enabled by vLLM
- Seed: not explicitly set

## Generation Hyperparameters

Stage3 requests used deterministic decoding:

```json
{
  "model": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
  "temperature": 0.0,
  "top_p": 0.9,
  "max_tokens": 8192,
  "stream": false,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0
}
```

## Minimal API Smoke Test

Start the model server:

```bash
cd code
bash stage3/serve_vllm.sh
```

In another shell, run:

```bash
cd code
python stage3/tests/api_smoke_test.py \
  --server-url http://localhost:8000 \
  --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
```

The smoke test sends a tiny synthetic radiology report and one synthetic rule to `/v1/chat/completions`, then verifies that the response contains a parseable JSON array with the expected `rule_id`.

## Full Batch Usage

Prepare local report data and input samples, then run:

```bash
cd code
export MIMIC_CXR_JPG_ROOT=data/mimic-cxr-jpg
python stage3/optimized_batch_processor_local.py \
  --input examples/stage3_samples.json \
  --output outputs/stage3/stage3_matched_samples.json \
  --server-url http://localhost:8000 \
  --cxr-rules stage3/cxr_grade_er_conversion_final_20250712_200006.json \
  --disease-mapping stage3/disease_in_MIMIC_mapto_WHO.csv \
  --mimic-cxr-path "$MIMIC_CXR_JPG_ROOT/files" \
  --checkpoint-interval 25
```

The output schema preserves the original sample fields and adds `global_who_rules`, `personal_who_rules`, and `medical_report_found`.
