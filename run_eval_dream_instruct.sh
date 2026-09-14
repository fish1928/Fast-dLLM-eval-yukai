#!/bin/bash
# Fast-dLLM eval -- dream-instruct thread. Same protocol as dream-base
# (no chat template, matching the other baseline wrappers' dream protocol;
# their dream eval supports apply_chat_template=true as a model_arg if we
# ever want the templated variant).
#   CUDA_VISIBLE_DEVICES=1 bash run_eval_dream_instruct.sh

set -u
cd "$(dirname "$0")/dream"

PRETRAINED="Dream-org/Dream-v0-Instruct-7B"
PORT=${PORT:-29633}
FOLDER_OUT=${FOLDER_OUT:-../results_baseline_eval/dream_instruct}
source ../eval_common.sh

run_dream_fast_task gsm8k_cot      8   256  -
run_dream_fast_task minerva_math   4   512  -
run_dream_fast_task bbh            3   256  -
run_dream_fast_task mbpp           3   512  escape_until=true  --confirm_run_unsafe_code
run_dream_fast_task humaneval      0   512  escape_until=true  --confirm_run_unsafe_code
run_dream_fast_task truthfulqa_gen 0   256  -

echo "[done] dream_instruct -> $FOLDER_OUT"
