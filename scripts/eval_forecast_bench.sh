#!/usr/bin/env bash
set -euo pipefail

# Evaluate a model on a ForecastBench question set.
#
# Outcomes live in a separate resolution file from the questions, so pass both
# to get anything scored. For the 2024-07-21 human release:
#   questions:   datasets/question_sets/2024-07-21-human.json
#   resolutions: datasets/resolution_sets/2024-07-21_resolution_set.json
# from github.com/forecastingresearch/forecastbench-datasets
#
# Usage: $0 [gpu] [model] <question_set> [resolution_set] [output_path]

gpu=${1:-0}
model=${2:-"Qwen/Qwen2.5-7B-Instruct"}
dataset_path=${3:?"Usage: $0 [gpu] [model] <question_set> [resolution_set] [output_path]"}
resolution_path=${4:-""}
output_path=${5:-"results/forecast_bench_results.csv"}

args=(
    --model "$model"
    --trait forecast_bench
    --benchmark forecast_bench
    --dataset_path "$dataset_path"
    --output_path "$output_path"
    --n_per_question 1
    --max_tokens 400
    --overwrite True
)

if [[ -n "$resolution_path" ]]; then
    args+=(--resolution_path "$resolution_path")
else
    echo "WARNING: no resolution set given; responses will be generated but" >&2
    echo "         nothing can be scored (no gold outcomes)." >&2
fi

CUDA_VISIBLE_DEVICES="$gpu" python -m eval.eval_persona "${args[@]}"
