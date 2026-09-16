#!/bin/bash
#################################################
# Shared launcher for the four run_eval_*.sh wrappers (Fast-dLLM official
# code: llada/eval_llada.py [model llada_dist] and dream/eval.py [model dream]).
#
# Variants per task:
#   nocache = their Baseline arm (one token/step, no cache, no threshold)
#   cache   = their headline "Dual Cache + Parallel" arm
#             (use_cache + dual_cache + confidence threshold 0.9)
# LLaDA block_length=32 everywhere: the block-wise KV cache is intrinsic to
# the method and 32 is the published recipe (their gsm8k AND humaneval).
# steps=gen_length for both arms -- under threshold decoding their loop
# ignores the per-step quota and runs until each block completes (verified in
# llada/generate.py), so steps only shapes the baseline. Dream cache arm uses
# diffusion_steps = len/32 exactly as their scripts.
# llada_dist auto-applies the chat template for *Instruct* model paths
# (is_instruct detection inside eval_llada.py) -- no CLI flag needed.
#
# CANONICAL dllm-meta generation-length spec (same as the dLLM-cache/dKV
# wrappers): gsm8k 256 (dream: gsm8k_cot), minerva_math 512, bbh 256,
# mbpp 512, humaneval 512, truthfulqa_gen 256. ifeval/followbench excluded.
#
# NOTE humaneval needs the authors' post-processing after the run:
#   python postprocess_code.py <samples_*.jsonl under the output folder>
#
# Env: CUDA_VISIBLE_DEVICES, NUM_PROCESSES (default 1), PORT, LIMIT,
#      FILTER_TASK, RUN_BASELINE=0/RUN_CACHE=0, THRESHOLD (default 0.9),
#      FOLDER_OUT
#################################################

# DEVICE convenience (same convention as the dllm-meta scripts): DEVICE=cuda:1
# pins this run to ONE gpu and always wins (an inherited CUDA_VISIBLE_DEVICES
# from a jupyter/scheduler session is not a per-run choice). If the session
# already restricts CUDA_VISIBLE_DEVICES (e.g. "2,3"), cuda:N selects the N-th
# entry of that list -- matching torch device numbering, which is relative to
# the visible set.
DEVICE=${DEVICE:-}
if [ -n "$DEVICE" ]; then
    _idx="${DEVICE#cuda:}"
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        IFS=',' read -r -a _gpus <<< "$CUDA_VISIBLE_DEVICES"
        if [ "$_idx" -ge "${#_gpus[@]}" ]; then
            echo "[error] DEVICE=$DEVICE but CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES has only ${#_gpus[@]} gpu(s)"; exit 1
        fi
        export CUDA_VISIBLE_DEVICES="${_gpus[$_idx]}"
    else
        export CUDA_VISIBLE_DEVICES="$_idx"
    fi
    echo "[device] DEVICE=$DEVICE -> CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

NUM_PROCESSES=${NUM_PROCESSES:-1}
LIMIT=${LIMIT:-}
FILTER_TASK=${FILTER_TASK:-}
RUN_BASELINE=${RUN_BASELINE:-1}
RUN_CACHE=${RUN_CACHE:-1}
THRESHOLD=${THRESHOLD:-0.9}

# LIMIT is a per-TASK budget. lm_eval applies --limit per SUBTASK, so group
# tasks would silently multiply it (minerva_math x7 subtasks, bbh x27) --
# divide (ceil) so LIMIT=500 means ~500 requests for EVERY task. Must match
# the dllm-meta run_bench_* scripts so all suites share identical subsets.
_limit_for_task () {
    if [ -z "$LIMIT" ]; then
        echo ""
        return 0
    fi
    case "$1" in
        minerva_math) echo $(( (LIMIT + 6) / 7 )) ;;
        bbh)          echo $(( (LIMIT + 26) / 27 )) ;;
        *)            echo "$LIMIT" ;;
    esac
}

_skip_task () {
    [ -n "$FILTER_TASK" ] && [ "$1" != "$FILTER_TASK" ]
}

_done_already () {
    [ -d "$1" ] && find "$1" -name "results*.json" 2>/dev/null | grep -q .
}

# _launch <eval_script> <lm_name> <task> <fewshot> <variant> <model_args> [extra flags...]
_launch () {
    local eval_script=$1 lm_name=$2 task=$3 fewshot=$4 variant=$5 model_args=$6
    shift 6

    local folder_task="$FOLDER_OUT/${task}_${variant}"
    if _done_already "$folder_task"; then
        echo "[skip] $folder_task has results"
        return 0
    fi

    local limit_task
    limit_task=$(_limit_for_task "$task")

    echo "[eval] model=$lm_name task=$task variant=$variant fewshot=$fewshot"
    accelerate launch --num_processes "$NUM_PROCESSES" --main_process_port "$PORT" \
        "$eval_script" \
        --model "$lm_name" \
        --tasks "$task" \
        --batch_size 1 \
        --model_args "$model_args" \
        --num_fewshot "$fewshot" \
        --output_path "$folder_task" \
        --log_samples \
        --trust_remote_code \
        ${limit_task:+--limit "$limit_task"} \
        "$@" \
        || echo "[warn] failed: $task/$variant -- continuing"

    if [ "$task" = "humaneval" ]; then
        echo "[note] humaneval requires post-processing: python postprocess_code.py $folder_task/**/samples_*.jsonl"
    fi
}

# run_llada_fast_task <task> <fewshot> <gen_length> [extra flags...]
run_llada_fast_task () {
    local task=$1 fewshot=$2 gen_length=$3
    shift 3
    _skip_task "$task" && return 0

    local margs_common="model_path=$PRETRAINED,gen_length=$gen_length,block_length=32,show_speed=True"

    if [ "$RUN_BASELINE" = "1" ]; then
        _launch eval_llada.py llada_dist "$task" "$fewshot" nocache \
            "$margs_common,steps=$gen_length" "$@"
    fi
    if [ "$RUN_CACHE" = "1" ]; then
        _launch eval_llada.py llada_dist "$task" "$fewshot" cache \
            "$margs_common,steps=$gen_length,use_cache=True,dual_cache=True,threshold=$THRESHOLD" "$@"
    fi
}

# run_dream_fast_task <task> <fewshot> <len> <extra_margs|-> [extra flags...]
run_dream_fast_task () {
    local task=$1 fewshot=$2 len=$3 extra_margs=$4
    shift 4
    _skip_task "$task" && return 0

    local margs_tail=""
    if [ "$extra_margs" != "-" ]; then
        margs_tail=",$extra_margs"
    fi
    local margs_common="pretrained=$PRETRAINED,max_new_tokens=$len,add_bos_token=true,show_speed=True$margs_tail"

    if [ "$RUN_BASELINE" = "1" ]; then
        _launch eval.py dream "$task" "$fewshot" nocache \
            "$margs_common,diffusion_steps=$len,alg=entropy" "$@"
    fi
    if [ "$RUN_CACHE" = "1" ]; then
        _launch eval.py dream "$task" "$fewshot" cache \
            "$margs_common,diffusion_steps=$((len / 32)),alg=confidence_threshold,threshold=$THRESHOLD,use_cache=true,dual_cache=true" "$@"
    fi
}
