#!/bin/bash
# Fast-dLLM eval -- dream-base thread. Faithful to their dream scripts:
# add_bos_token=true, no chat template, default temperature; humaneval adds
# escape_until=true (their recipe). gsm8k task is gsm8k_cot at 8-shot for
# cross-baseline protocol consistency.
#   CUDA_VISIBLE_DEVICES=1 bash run_eval_dream_base.sh

set -u
cd "$(dirname "$0")/dream"

PRETRAINED="Dream-org/Dream-v0-Base-7B"
PORT=${PORT:-29632}
FOLDER_OUT=${FOLDER_OUT:-../results_baseline_eval/dream_base}
source ../eval_common.sh

#                  task           fs  len  extra_margs
run_dream_fast_task gsm8k_cot      8   256  -
run_dream_fast_task minerva_math   4   512  -
run_dream_fast_task bbh            3   256  -
run_dream_fast_task mbpp           3   512  escape_until=true  --confirm_run_unsafe_code
run_dream_fast_task humaneval      0   512  escape_until=true  --confirm_run_unsafe_code
run_dream_fast_task truthfulqa_gen 0   256  -

echo "[done] dream_base -> $FOLDER_OUT"
