#!/bin/bash
# Fast-dLLM eval -- llada-base thread (llada_dist auto-detects non-instruct
# paths and uses raw prompts). Block 32 everywhere (method recipe).
#   CUDA_VISIBLE_DEVICES=0 bash run_eval_llada_base.sh
#   LIMIT=20 FILTER_TASK=gsm8k bash run_eval_llada_base.sh    # smoke

set -u
cd "$(dirname "$0")/llada"

PRETRAINED="GSAI-ML/LLaDA-8B-Base"
PORT=${PORT:-29630}
FOLDER_OUT=${FOLDER_OUT:-../results_baseline_eval/llada_base}
source ../eval_common.sh

#                  task           fs  gen
run_llada_fast_task gsm8k          4   256
run_llada_fast_task minerva_math   4   512
run_llada_fast_task bbh            3   256
run_llada_fast_task mbpp           3   512  --confirm_run_unsafe_code
run_llada_fast_task humaneval      0   512  --confirm_run_unsafe_code
run_llada_fast_task truthfulqa_gen 0   256

echo "[done] llada_base -> $FOLDER_OUT"
