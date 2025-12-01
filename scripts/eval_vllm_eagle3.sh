#!/usr/bin/env bash

# set -euo pipefail

# Run vLLM + EAGLE3 speculative decoding baseline on 3 models × 3 datasets.
# NOTE: Please update the DRAFT_MODELS array to point to the actual EAGLE3 draft models you have.

cd eval/benchmarks

export PYTHONOPTIMIZE=1
export ENABLE_INTRA_NODE_COMM=1

MODELS=(
  "Qwen/Qwen3-1.7B"
  "Qwen/Qwen3-8B"
  "Qwen/Qwen3-14B"
)

# One draft model per base model (same order as MODELS)
DRAFT_MODELS=(
  "AngelSlim/Qwen3-1.7B_eagle3"
  "AngelSlim/Qwen3-8B_eagle3"
  "AngelSlim/Qwen3-14B_eagle3"
)

DATASETS=(
  "aime24"
  "livecodebench"
  "olympiadbench"
)

PROMPT_TYPE="qwen3-math-thinking"
SPLIT="test"
NUM_TEST_SAMPLE=-1
TEMPERATURE=0.6
N_SAMPLING=1
TOP_P=1.0

VLLM_EAGLE3_NUM_SPECULATIVE_TOKENS=3

for idx in "${!MODELS[@]}"; do
  model=${MODELS[$idx]}
  draft_model=${DRAFT_MODELS[$idx]}
  short_model=${model//\//_}

  # Per-model vLLM GPU memory utilization (match SpecGen mem_fraction)
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
      VLLM_GPU_MEMORY_UTILIZATION=0.8
      export CUDA_VISIBLE_DEVICES="0,1,2,3"
      ;;
  esac

  for data_name in "${DATASETS[@]}"; do
    # Per-dataset repeat count (mirrors eval_specgen.sh)
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

    echo "Running vLLM EAGLE3 baseline: model=${model}, draft_model=${draft_model}, dataset=${data_name}, tp_ranks=${TP_RANKS}, mem_fraction=${VLLM_GPU_MEMORY_UTILIZATION}, repeat=${REPEAT_DATASET}"

    python -O math_eval.py \
      --use_vllm \
      --model_name_or_path "${model}" \
      --prompt_type "${PROMPT_TYPE}" \
      --split "${SPLIT}" \
      --start 0 \
      --end -1 \
      --data_names "${data_name}" \
      --temperature "${TEMPERATURE}" \
      --num_test_sample "${NUM_TEST_SAMPLE}" \
      --n_sampling "${N_SAMPLING}" \
      --top_p "${TOP_P}" \
      --repeat_dataset "${REPEAT_DATASET}" \
      --apply_chat_template \
      --enable_thinking \
      --save_outputs \
      --output_dir "vllm_eagle3/${short_model}" \
      --overwrite \
      --vllm_enable_eagle3 \
      --vllm_draft_model_path "${draft_model}" \
      --vllm_eagle3_num_speculative_tokens "${VLLM_EAGLE3_NUM_SPECULATIVE_TOKENS}" \
      --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
  done
done




