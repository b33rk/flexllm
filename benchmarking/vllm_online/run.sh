#!/bin/bash

set -x
set -o pipefail

# Cd into directory holding this script
cd "$(dirname "$0")"
SCRIPT_DIR=$(pwd)

VLLM_V1=0
EAGER_MODE=true
MODEL_NAMES=("meta-llama/Llama-3.1-8B-Instruct")
TP_DEGREES=(1)
model_types=("llama")
QPS_vals=(
  4.0
  3.0
  2.0
  1.3 # 4/3
  1.0
  0.7
  0.5
)
trace=sharegpt
BATCH_SIZE=256
MAX_TOKENS_PER_BATCH=256
MAX_NUM_REQUESTS=5000
MAX_SEQ_LEN=8192

# export CUDA_VISIBLE_DEVICES=1

check_gpus() {
  declare -g gpu_count=$(nvidia-smi --list-gpus | wc -l)
  if [[ $gpu_count -gt 0 ]]; then
    echo "GPU found."
  else
    echo "Need at least 1 GPU to run benchmarking."
    exit 1
  fi
  declare -g gpu_type=$(nvidia-smi --query-gpu=name --format=csv,noheader | awk '{print $2}')
  echo "GPU type is $gpu_type"
}

wait_for_server() {
  local max_attempts=120  # 120 * 10 seconds = 1200 seconds
  local attempt=0

  while [ $attempt -lt $max_attempts ]; do
    if curl -s -X POST localhost:8000/v1/completions >/dev/null 2>&1; then
      return 0
    fi
    sleep 10
    ((attempt++))
  done
  return 1
}

kill_gpu_processes() {
  lsof -t -i:8000 | xargs -r kill -9
  pgrep python3 | xargs -r kill -9
  pgrep python  | xargs -r kill -9
  pgrep vllm    | xargs -r kill -9

  # wait until GPU memory usage smaller than 1GB
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1)" -ge 1000 ]; do
    sleep 1
  done

  rm -rf ~/.config/vllm
}

cleanup() {
  echo "Script interrupted, cleaning up..."
  kill_gpu_processes
  exit 130
}
trap cleanup INT TERM

run_serving_tests() {
  local model_name=${1}
  local tp_degree=${2}
  local vllm_use_v1=${3}
  local eager_mode=${4}
  local batch_size=${5}
  local max_num_batched_tokens=${6}
  local qps=${7}
  local max_num_requests=${8}
  local trace=${9}
  local trace_file=${10}

  if [ "$max_num_batched_tokens" -lt "$batch_size" ]; then
    echo "max_num_batched_tokens is less than batch_size, skipping this test."
    return
  fi

  # V100 (SM 7.0) does not support Flash-Attention 2 or BF16 tensor cores.
  # Force xformers backend and FP16.
  server_command="VLLM_USE_V1=${vllm_use_v1} VLLM_ATTENTION_BACKEND=XFORMERS vllm serve ${model_name} \
      --tensor-parallel-size ${tp_degree} \
      --dtype float16 \
      --max-model-len ${MAX_SEQ_LEN} \
      --gpu-memory-utilization 0.92 \
      --enable-chunked-prefill \
      --max-num-seqs ${batch_size} \
      --max-num-batched-tokens ${max_num_batched_tokens} \
      --disable-log-stats \
      --disable-log-requests \
      --swap-space 0"

  if [ "$eager_mode" = true ]; then
    server_command+=" --enforce-eager"
  fi

  echo "Starting VLLM server"
  echo "Server command: $server_command"
  bash -c "$server_command" &
  server_pid=$!

  if wait_for_server; then
    echo ""
    echo "vllm server is up and running."
  else
    echo ""
    echo "vllm failed to start within the timeout period."
    kill -9 $server_pid 2>/dev/null
    kill_gpu_processes
    return 1
  fi

  mkdir -p ../../output/vllm

  result_filename=$(echo "results_${trace}_$( [ "$eager_mode" = true ] && echo "eager_" )$( [ "$vllm_use_v1" = 1 ] && echo "v1_" )${model_name//\//_}_bz_${batch_size}_max_num_batched_tokens_${max_num_batched_tokens}_${qps}_qps_v100_.json" | tr '[:upper:]' '[:lower:]')

  client_command="VLLM_USE_V1=${vllm_use_v1} python3 benchmark_vllm.py \
        --model ${model_name} \
        --backend vllm \
        --ignore-eos \
        --num-prompts ${max_num_requests} \
        --dataset-path ${trace_file} \
        --save-result --save-detailed \
        --result-dir ../../output/vllm \
        --result-filename ${result_filename}"

  echo "Client command: $client_command"
  bash -c "$client_command"

  kill -9 $server_pid
  kill_gpu_processes
}

main() {
    check_gpus
    (which wget && which curl) || (apt-get update && apt-get install -y wget curl)
    (which jq) || (apt-get update && apt-get -y install jq)
    (which lsof) || (apt-get update && apt-get install -y lsof)

    export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
    export VLLM_LOG_LEVEL="WARNING"

    # Pin to a single GPU so the 4-way V100 box runs one experiment at a time.
    # Change to 0,1 / 0,1,2,3 if you bump TP_DEGREES to 2 or 4.
    export CUDA_VISIBLE_DEVICES=0

    for i in "${!MODEL_NAMES[@]}"; do
        for qps in "${QPS_vals[@]}"; do
            model_name="${MODEL_NAMES[$i]}"
            tp_degree="${TP_DEGREES[$i]}"
            MODEL_TYPE=${model_types[$i]}
            trace_file="../../traces/burstgpt/${MODEL_TYPE}/${trace}_${MAX_SEQ_LEN}_${qps}_qps.json"
            if [ ! -f "$trace_file" ]; then
              echo "Error: Trace file $trace_file does not exist!"
              exit 1
            fi
            run_serving_tests "$model_name" "$tp_degree" "$VLLM_V1" "$EAGER_MODE" "$BATCH_SIZE" "$MAX_TOKENS_PER_BATCH" $qps $MAX_NUM_REQUESTS "$trace" $trace_file
        done
    done

    echo "All experiments completed!"
}

main