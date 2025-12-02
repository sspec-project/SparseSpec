#!/usr/bin/env bash

set -euo pipefail

# Run vLLM self-speculative decoding with n-gram assistance (TriForce-style)
# on selected models × datasets using math_eval_triforce.py from the sspec repo.

cd eval/benchmarks

export PYTHONOPTIMIZE=1
export ENABLE_INTRA_NODE_COMM=1

# Models to evaluate (3 models)
MODELS=(
  "Qwen/Qwen3-1.7B"
  "Qwen/Qwen3-8B"
  "Qwen/Qwen3-14B"
)

# Datasets to evaluate (3 datasets)
DATASETS=(
  "aime24"
  "livecodebench"
  "olympiadbench"
)

SPLIT="test"
NUM_TEST_SAMPLE=-1
PROMPT_TYPE="qwen3-math-thinking"


# Self-spec with n-gram parameters (mirrors sspec/scripts/serve/eval_vllm_triforce_new.sh)
SSPEC_NGRAM_NUM_SPECULATIVE_TOKENS="${SSPEC_NGRAM_NUM_SPECULATIVE_TOKENS:-8}"  # Threshold for ACCUMULATING -> VERIFYING
SSPEC_NGRAM_NUM_DRAFT_TOKENS="${SSPEC_NGRAM_NUM_DRAFT_TOKENS:-1}"              # N-gram draft tokens per step
SSPEC_NGRAM_PROMPT_LOOKUP_MIN="${SSPEC_NGRAM_PROMPT_LOOKUP_MIN:-5}"             # Default: same as num_draft_tokens
SSPEC_NGRAM_PROMPT_LOOKUP_MAX="${SSPEC_NGRAM_PROMPT_LOOKUP_MAX:-5}"             # Default: same as num_draft_tokens
SSPEC_NGRAM_SINK_SIZE="${SSPEC_NGRAM_SINK_SIZE:-32}"                            # Streaming cache sink blocks
SSPEC_NGRAM_RECENT_RATIO="${SSPEC_NGRAM_RECENT_RATIO:-0.05}"                    # Streaming cache recent ratio
SSPEC_NGRAM_BLOCK_SIZE="${SSPEC_NGRAM_BLOCK_SIZE:-1}"                           # Block size (1 disables prefix caching)

REPEAT_DATASET="${REPEAT_DATASET:-10}"  # default; per-dataset overrides below

for model in "${MODELS[@]}"; do
  short_model=${model//\//_}

  # Per-model TP ranks and vLLM GPU memory utilization (match other scripts)
  VLLM_GPU_MEMORY_UTILIZATION=0.9
  TP_RANKS=1
  case "${model}" in
    *"1.7B"*)
      TP_RANKS=1
      VLLM_GPU_MEMORY_UTILIZATION=0.88
      export CUDA_VISIBLE_DEVICES="0"
      ;;
    *"8B"*)
      TP_RANKS=2
      VLLM_GPU_MEMORY_UTILIZATION=0.85
      export CUDA_VISIBLE_DEVICES="0,1"
      ;;
    *"14B"*)
      TP_RANKS=4
      VLLM_GPU_MEMORY_UTILIZATION=0.85
      export CUDA_VISIBLE_DEVICES="0,1,2,3"
      ;;
  esac

  for data_name in "${DATASETS[@]}"; do
    case "${data_name}" in
      "aime24")
        TEMPERATURE=0.6
        REPEAT_DATASET=60
        ;;
      "livecodebench")
        TEMPERATURE=0.6
        REPEAT_DATASET=2
        ;;
      "olympiadbench")
        TEMPERATURE=0.6
        REPEAT_DATASET=3
        ;;
    esac

    echo "Running vLLM TriForce-NGram baseline: model=${model}, dataset=${data_name}, tp_ranks=${TP_RANKS}, mem_fraction=${VLLM_GPU_MEMORY_UTILIZATION}, repeat=${REPEAT_DATASET}"

    python -O math_eval.py \
      --model_name_or_path "${model}" \
      --data_names "${data_name}" \
      --output_dir "vllm_triforce_ngram/${short_model}" \
      --split "${SPLIT}" \
      --prompt_type "${PROMPT_TYPE}" \
      --num_test_sample "${NUM_TEST_SAMPLE}" \
      --seed 0 \
      --start 0 \
      --end -1 \
      --temperature "${TEMPERATURE}" \
      --use_vllm \
      --apply_chat_template \
      --enable_thinking \
      --vllm_enable_sspec_ngram \
      --vllm_sspec_ngram_num_speculative_tokens "${SSPEC_NGRAM_NUM_SPECULATIVE_TOKENS}" \
      --vllm_sspec_ngram_num_ngram_draft_tokens "${SSPEC_NGRAM_NUM_DRAFT_TOKENS}" \
      --vllm_sspec_ngram_sink_size "${SSPEC_NGRAM_SINK_SIZE}" \
      --vllm_sspec_ngram_recent_ratio "${SSPEC_NGRAM_RECENT_RATIO}" \
      --vllm_sspec_ngram_block_size "${SSPEC_NGRAM_BLOCK_SIZE}" \
      --vllm_sspec_ngram_prompt_lookup_min "${SSPEC_NGRAM_PROMPT_LOOKUP_MIN}" \
      --vllm_sspec_ngram_prompt_lookup_max "${SSPEC_NGRAM_PROMPT_LOOKUP_MAX}" \
      --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
      --save_outputs \
      --overwrite \
      --repeat_dataset "${REPEAT_DATASET}"
  done
done
