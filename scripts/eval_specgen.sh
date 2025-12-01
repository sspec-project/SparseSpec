#!/usr/bin/env bash

# set -euo pipefail

# Run SpecGen baseline on 3 models × 3 datasets using eval/benchmarks/math_eval.py.

cd eval/benchmarks

export PYTHONOPTIMIZE=1
export ENABLE_INTRA_NODE_COMM=1

MODELS=(
  "Qwen/Qwen3-1.7B"
  "Qwen/Qwen3-8B"
  "Qwen/Qwen3-14B"
)

# Datasets to evaluate
DATASETS=(
  "aime24"
  "livecodebench"
  "olympiadbench"
)

PROMPT_TYPE="qwen3-math-thinking"
SPLIT="test"
NUM_TEST_SAMPLE=-1
TEMPERATURE=0.6

# SpecGen engine configuration (aligned with serve/run.py & example-single.sh)
MAX_BATCH_SIZE=128
KV_CACHE="pillar"          # full / streaming / pillar
ADMIT_POLICY="offloading"  # for non-full kv_cache
BATCH_POLICY="greedy"
SPEC_STRIDE=8
BUDGET_RATIO=0.05
NUM_MIN_BUDGET=192
CHUNK_SIZE=256

for model in "${MODELS[@]}"; do
  short_model=${model//\//_}

  # Per-model TP ranks and mem_fraction
  TP_RANKS=1
  MEM_FRACTION=0.9
  case "${model}" in
    *"1.7B"*)
      TP_RANKS=1
      MEM_FRACTION=0.88
      ;;
    *"8B"*)
      TP_RANKS=2
      MEM_FRACTION=0.85
      ;;
    *"14B"*)
      TP_RANKS=4
      MEM_FRACTION=0.85
      ;;
  esac

  for data_name in "${DATASETS[@]}"; do
    # Per-dataset repeat count
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

    echo "Running SpecGen baseline: model=${model}, dataset=${data_name}, tp_ranks=${TP_RANKS}, mem_fraction=${MEM_FRACTION}, repeat=${REPEAT_DATASET}"

    python -O -m torch.distributed.run --standalone --nproc_per_node="${TP_RANKS}" -- math_eval.py \
      --use_specgen \
      --model_name_or_path "${model}" \
      --prompt_type "${PROMPT_TYPE}" \
      --apply_chat_template \
      --enable_thinking \
      --split "${SPLIT}" \
      --start 0 \
      --end -1 \
      --data_names "${data_name}" \
      --temperature "${TEMPERATURE}" \
      --num_test_sample "${NUM_TEST_SAMPLE}" \
      --tp_ranks "${TP_RANKS}" \
      --kv_cache "${KV_CACHE}" \
      --max_batch_size "${MAX_BATCH_SIZE}" \
      --admit_policy "${ADMIT_POLICY}" \
      --cuda_graph \
      --async_cpu \
      --mem_fraction "${MEM_FRACTION}" \
      --spec_stride "${SPEC_STRIDE}" \
      --budget_ratio "${BUDGET_RATIO}" \
      --num_min_budget "${NUM_MIN_BUDGET}" \
      --batch_policy "${BATCH_POLICY}" \
      --repeat_dataset "${REPEAT_DATASET}" \
      --chunk_size "${CHUNK_SIZE}" \
      --save_outputs \
      --output_dir "specgen/${short_model}" \
      --overwrite
  done
done


