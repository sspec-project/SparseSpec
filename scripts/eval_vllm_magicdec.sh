#!/usr/bin/env bash

# set -euo pipefail

# Run vLLM self-speculative decoding (MagicDec-style) on 3 models × 3 datasets
# using the original math_eval_triforce.py from the sspec repo.

cd eval/benchmarks

export PYTHONOPTIMIZE=1
export ENABLE_INTRA_NODE_COMM=1

# Models to evaluate (3 models, local paths)
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

# Default prompt types; per-dataset overrides are applied inside the loop
PROMPT_TYPE="qwen3-math-thinking"

NUM_TEST_SAMPLE=-1
TEMPERATURE=0.6

# Self-spec parameters (streaming cache configuration)
SSPEC_NUM_TOKENS="${SSPEC_NUM_TOKENS:-8}"
SSPEC_SINK_SIZE="${SSPEC_SINK_SIZE:-32}"           # blocks (8 blocks = 128 tokens with block_size=16)
SSPEC_RECENT_RATIO="${SSPEC_RECENT_RATIO:-0.05}"  # ratio (5% of computed tokens)
SSPEC_BLOCK_SIZE="${SSPEC_BLOCK_SIZE:-1}"

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
    # Per-dataset repeat count (mirror original MagicDec script)
    REPEAT_DATASET=1
    case "${data_name}" in
      "aime24")
        REPEAT_DATASET=60
        ;;
      "livecodebench")
        REPEAT_DATASET=2
        ;;
      "olympiadbench")
        REPEAT_DATASET=3
        ;;
    esac

    echo "Running vLLM MagicDec baseline: model=${model}, dataset=${data_name}, tp_ranks=${TP_RANKS}, mem_fraction=${VLLM_GPU_MEMORY_UTILIZATION}, repeat=${REPEAT_DATASET}"

    python -O math_eval.py \
      --model_name_or_path "${model}" \
      --data_names "${data_name}" \
      --output_dir "vllm_magicdec/${short_model}" \
      --split test \
      --prompt_type "${PROMPT_TYPE}" \
      --seed 0 \
      --temperature "${TEMPERATURE}" \
      --use_vllm \
      --apply_chat_template \
      --enable_thinking \
      --vllm_enable_sspec \
      --vllm_sspec_num_speculative_tokens "${SSPEC_NUM_TOKENS}" \
      --vllm_sspec_sink_size "${SSPEC_SINK_SIZE}" \
      --vllm_sspec_recent_ratio "${SSPEC_RECENT_RATIO}" \
      --vllm_sspec_block_size "${SSPEC_BLOCK_SIZE}" \
      --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
      --save_outputs \
      --overwrite \
      --repeat_dataset "${REPEAT_DATASET}"
  done
done



