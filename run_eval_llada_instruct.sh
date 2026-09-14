#!/bin/bash
# Fast-dLLM eval -- llada-instruct thread. llada_dist detects 'instruct' in
# the model path and applies the chat template internally (their design; no
# CLI chat flags). Block 32 everywhere (method recipe).
#   CUDA_VISIBLE_DEVICES=0 bash run_eval_llada_instruct.sh

set -u
cd "$(dirname "$0")/llada"

PRETRAINED="GSAI-ML/LLaDA-8B-Instruct"
PORT=${PORT:-29631}
FOLDER_OUT=${FOLDER_OUT:-../results_baseline_eval/llada_instruct}
source ../eval_common.sh

run_llada_fast_task gsm8k          4   256
run_llada_fast_task minerva_math   0   512
run_llada_fast_task bbh            3   256
run_llada_fast_task mbpp           3   512  --confirm_run_unsafe_code
run_llada_fast_task humaneval      0   512  --confirm_run_unsafe_code
run_llada_fast_task truthfulqa_gen 0   256

echo "[done] llada_instruct -> $FOLDER_OUT"
