#!/bin/bash
# Comprehensive evaluation script for all checkpoints
CHECKPOINTS="base $(seq 500 500 5500) 5760"
TASK="mbpp_instruct"

mkdir -p eval_results/0shot eval_results/3shot

for ckpt in $CHECKPOINTS; do
    echo "Processing Checkpoint: $ckpt"
    
    if [ "$ckpt" == "base" ]; then
        ARGS="pretrained=meta-llama/Meta-Llama-3-8B-Instruct"
    else
        ARGS="pretrained=meta-llama/Meta-Llama-3-8B-Instruct,peft=outputs/llama-8b-python-grpo/checkpoint-$ckpt"
    fi

    # Run 0-shot
    lm_eval --model hf --model_args "$ARGS,dtype=bfloat16" \
        --tasks $TASK --num_fewshot 0 --apply_chat_template \
        --output_path ./eval_results/0shot/mbpp_0shot_${ckpt}.json

    # Run 3-shot
    lm_eval --model hf --model_args "$ARGS,dtype=bfloat16" \
        --tasks $TASK --num_fewshot 3 --apply_chat_template \
        --output_path ./eval_results/3shot/mbpp_3shot_${ckpt}.json
done
