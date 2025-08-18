#!/bin/bash

git pull

python SoftT-main.py \
  --exp_name "Llama-3.2-3B-E.T.R"\
  --train_dimension_filter "['nan','Explanations', 'Textual.Evidence', 'Rhetorical.Strategies']" \
  --resample_train


#python SoftT-main.py \
#  --exp_name "Llama-3.2-3B-A.T"\
#  --train_dimension_filter "['nan', 'Argument', 'Thesis']" \
#  --resample_train


python SoftT-main.py \
  --exp_name "Llama-3.2-3B-O.L"\
  --train_dimension_filter "['nan','Organization', 'Language']" \
  --resample_train \




python SoftT-main.py \
  --model_name "Qwen/Qwen3-8B"\
  --exp_name "Qwen3-8B-all"\
  --train_size 20000 \
  --gradient_checkpointing \
  --force_strategy "add_tokens"