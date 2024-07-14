#!/bin/bash

MODEL_PATH=$1
MODEL_BASENAME=$(basename $MODEL_PATH)

python -m lm_eval --model hf \
    --model_args pretrained=$MODEL_PATH \
    --tasks blimp_filtered,blimp_supplement \
    --device cuda:0 \
    --batch_size 128 \
    --output_path results/blimp/${MODEL_BASENAME}/blimp_results.json

# use `--model_args pretrained=$MODEL_PATH,backend="mlm"` if you're using a custom masked LM
# add --trust_remote_code if you need to load custom config/model files
