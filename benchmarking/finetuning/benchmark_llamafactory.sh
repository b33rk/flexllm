set -x
set -e

# Cd into the LLaMA-Factory main directory
cd "$(dirname "$0")"
SCRIPT_DIR=$(pwd)
cd "${SCRIPT_DIR}/../../LLaMA-Factory"

# rm -rf saves

# Single GPU LLAMA-3.1 8B
CUDA_VISIBLE_DEVICES=0 DISABLE_VERSION_CHECK=1 llamafactory-cli train examples/flexllm/t1_llama_8B.yaml

rm -rf ../output/llama-factory || true
mkdir -p ../output/llama-factory
cp -r ./saves/* ../output/llama-factory/

echo "All experiments completed!"
