#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: ./eval_benchmark.sh [options]

Required options:
  --exp_path PATH            Base path for saving experimental results.
  --dataset_path PATH        Base path of datasets for evaluation.

Optional options:
  --eval_models MODEL...     Models for evaluation (default: gpt-3.5-turbo gpt-4).
  --max_data_num NUM         Max number of test cases for one dataset (default: 5000).
  --pool_num NUM             Number of CPU pools (default: 10).
  --gpu_devices DEVICES      Available GPUs (default: 0).
  --max_memory_per_device GB Max memory per GPU device in GB (default: 20).
  --eval_group_key KEY       Key to group evaluation results (default: none).
  --seed SEED                Random seed (default: 1).
  -h, --help                 Show this help message.
USAGE
}

exp_path=""
dataset_path=""
eval_models=("gpt-3.5-turbo" "gpt-4")
max_data_num=5000
pool_num=10
gpu_devices="0"
max_memory_per_device=20
eval_group_key=""
seed=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --exp_path)
      exp_path="$2"
      shift 2
      ;;
    --dataset_path)
      dataset_path="$2"
      shift 2
      ;;
    --eval_models)
      shift
      eval_models=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        eval_models+=("$1")
        shift
      done
      ;;
    --max_data_num)
      max_data_num="$2"
      shift 2
      ;;
    --pool_num)
      pool_num="$2"
      shift 2
      ;;
    --gpu_devices)
      gpu_devices="$2"
      shift 2
      ;;
    --max_memory_per_device)
      max_memory_per_device="$2"
      shift 2
      ;;
    --eval_group_key)
      eval_group_key="$2"
      shift 2
      ;;
    --seed)
      seed="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$exp_path" || -z "$dataset_path" ]]; then
  echo "--exp_path and --dataset_path are required." >&2
  usage
  exit 1
fi

python eval_benchmark.py \
  --exp_path "$exp_path" \
  --dataset_path "$dataset_path" \
  --eval_models "${eval_models[@]}" \
  --max_data_num "$max_data_num" \
  --pool_num "$pool_num" \
  --gpu_devices "$gpu_devices" \
  --max_memory_per_device "$max_memory_per_device" \
  ${eval_group_key:+--eval_group_key "$eval_group_key"} \
  --seed "$seed"
