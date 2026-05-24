#!/bin/bash

set -x
set -o pipefail

# Cd into directory holding this script
cd "$(dirname "$0")"
SCRIPT_DIR=$(pwd)

export VLLM_CHUNKED_PREFILL_ENABLED=0
VLLM_V1=0
EAGER_MODE=true
MODEL_NAMES=("meta-llama/Llama-3.1-8B-Instruct")
TP_DEGREES=(1)
model_types=("llama")
QPS_vals=(
  6.7 # 20/3
  5.3 # 16/3
  2.7 # 8/3
  1.3 # 4/3
  20.0
  16.0
  12.0
  10.0
  8.0
  6.0
  4.0
  2.0
)
trace=sharegpt
BATCH_SIZE=256
MAX_TOKENS_PER_BATCH=2048
MAX_NUM_REQUESTS=5000
MAX_SEQ_LEN=8192

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
  # Kill the vLLM server we just launched (and any children it spawned)
  if [ -n "${server_pid:-}" ]; then
    kill -9 "$server_pid" 2>/dev/null
    pkill -9 -P "$server_pid" 2>/dev/null
  fi

  # Kill anything still bound to port 8000
  lsof -t -i:8000 | xargs -r kill -9 2>/dev/null

  # Kill anything currently using *our* GPU only — not all python on the node
  local my_gpu="${CUDA_VISIBLE_DEVICES:-0}"
  nvidia-smi -i "$my_gpu" --query-compute-apps=pid --format=csv,noheader \
    | xargs -r kill -9 2>/dev/null

  sleep 2

  # Bounded wait — check our GPU specifically, give up after 60s
  local waited=0
  while [ "$(nvidia-smi -i "$my_gpu" --query-gpu=memory.used --format=csv,noheader,nounits)" -ge 1000 ]; do
    sleep 1
    waited=$((waited + 1))
    if [ "$waited" -ge 60 ]; then
      echo "WARNING: GPU ${my_gpu} memory did not drop below 1GB after 60s. Continuing."
      nvidia-smi -i "$my_gpu" --query-compute-apps=pid,process_name,used_memory --format=csv
      break
    fi
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

  # V100 (SM 7.0) constraints:
  #   - No FlashAttention-2  -> VLLM_ATTENTION_BACKEND=XFORMERS
  #   - No BF16 tensor cores -> --dtype float16
  #   - Triton chunked-prefill kernel asserts on FP16 (issue #17152, #11352)
  #       -> --no-enable-chunked-prefill
  #   - V1 engine refuses SM<8.0 -> VLLM_USE_V1=0 (set at top of script)
  server_command="VLLM_USE_V1=${vllm_use_v1} VLLM_ATTENTION_BACKEND=XFORMERS vllm serve ${model_name} \
      --tensor-parallel-size ${tp_degree} \
      --dtype float16 \
      --max-model-len ${MAX_SEQ_LEN} \
      --gpu-memory-utilization 0.92 \
      --max-num-seqs ${batch_size} \
      --max-num-batched-tokens ${max_num_batched_tokens} \
      --disable-custom-all-reduce \
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

  # Note: we always emit "v1_" in the filename so parse_data.py finds the files,
  # even though we're running V0. The file content is V0 data; the tag is for
  # parser compatibility. Document this in the thesis methodology.
  result_filename=$(echo "results_${trace}_$( [ "$eager_mode" = true ] && echo "eager_" )v1_${model_name//\//_}_bz_${batch_size}_max_num_batched_tokens_${max_num_batched_tokens}_${qps}_qps_v100_.json" | tr '[:upper:]' '[:lower:]')

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

  # clean up
  kill -9 $server_pid 2>/dev/null
  kill_gpu_processes
}

main() {
    check_gpus
    (which wget && which curl) || (apt-get update && apt-get install -y wget curl)
    (which jq) || (apt-get update && apt-get -y install jq)
    (which lsof) || (apt-get update && apt-get install -y lsof)

    export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
    export VLLM_LOG_LEVEL="WARNING"

    # Pin to a single GPU. Change index if running in parallel with another job.
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