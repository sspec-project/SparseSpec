
# Setup wandb api if logging
# export WANDB_API_KEY=

export PYTHONOPTIMIZE=1
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="9.0a"

dir=$(pwd)
result_dir=$dir/result
mkdir -p $result_dir

# runtime configuration
# please change for customization
model=Qwen/Qwen3-8B
tp_ranks=1 # tensor parallelism ranks
num_requests=128 # number of total requests
max_batch_size=256 # max batch size
admit_policy=offloading # enable offloading when OOM
clip_length=16384 # maximum output length
mem_fraction=0.9 # maximal utilization of GPU memory
temperature=0.6 # sampling temperature
chunk_size=2048 # chunked prefill size (reduce peak memory)

# spec config
spec_stride=8 # number of draft token per iteration
budget_ratio=0.05 # sparsity ratio of KV cache
num_min_budget=128 # minimum number of draft tokens, i.e. budget = max(min_budget, int(len_kv_cache * budget_ratio))

# nsys profiling command
# for torch profiler, add `--enable-torch-profiler`
NSYS_CMD=(
    "nsys"
    "profile"
    "--cuda-graph-trace node"
    "--trace=cuda,nvtx,osrt"
    "--delay=500"
    "--duration=3"
    "--force-overwrite true"
    "-o profile"
)

cd $dir/serve

# 1. SpecGen w/ sparse self-speculation (PillarAttn) enabled
kv_cache=pillar # full/streaming/pillar

# cancel comment to enable nsys profiling
# ${NSYS_CMD[@]} \
torchrun --standalone --nproc_per_node=$tp_ranks run.py \
    --model_name $model \
    --tp_ranks $tp_ranks \
    --num_requests $num_requests \
    --max_batch_size $max_batch_size \
    --admit_policy $admit_policy \
    --clip_length $clip_length \
    --mem_fraction $mem_fraction \
    --chunk_size $chunk_size \
    --temperature $temperature \
    --kv_cache $kv_cache \
    --spec_stride $spec_stride \
    --budget_ratio $budget_ratio \
    --num_min_budget $num_min_budget \
    --cuda_graph \
    --async_cpu \
    --result_dir $result_dir/${model}_${kv_cache}_${clip_length}_${mem_fraction}_${chunk_size}_${spec_stride}_${budget_ratio}_${num_min_budget}_${temperature}


# 2. SpecGen w/ sparse self-speculation (StreamingLLM) enabled
# which is essentially our reproduced MagicDec
kv_cache=streaming # full/streaming/pillar

# cancel comment to enable nsys profiling
# ${NSYS_CMD[@]} \
torchrun --standalone --nproc_per_node=$tp_ranks run.py \
    --model_name $model \
    --tp_ranks $tp_ranks \
    --num_requests $num_requests \
    --max_batch_size $max_batch_size \
    --admit_policy $admit_policy \
    --clip_length $clip_length \
    --mem_fraction $mem_fraction \
    --chunk_size $chunk_size \
    --temperature $temperature \
    --kv_cache $kv_cache \
    --spec_stride $spec_stride \
    --budget_ratio $budget_ratio \
    --num_min_budget $num_min_budget \
    --cuda_graph \
    --result_dir $result_dir/${model}_${kv_cache}_${clip_length}_${mem_fraction}_${chunk_size}_${spec_stride}_${budget_ratio}_${num_min_budget}_${temperature}



# 3. SpecGen w/o speculation
kv_cache=full # full/streaming/pillar

# cancel comment to enable nsys profiling
# ${NSYS_CMD[@]} \
torchrun --standalone --nproc_per_node=$tp_ranks run.py \
    --model_name $model \
    --tp_ranks $tp_ranks \
    --num_requests $num_requests \
    --max_batch_size $max_batch_size \
    --admit_policy $admit_policy \
    --clip_length $clip_length \
    --mem_fraction $mem_fraction \
    --chunk_size $chunk_size \
    --temperature $temperature \
    --kv_cache $kv_cache \
    --cuda_graph \
    --async_cpu \
    --result_dir $result_dir/${model}_${kv_cache}_${clip_length}_${mem_fraction}_${chunk_size}_${temperature}
