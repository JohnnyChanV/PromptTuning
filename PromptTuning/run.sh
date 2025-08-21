#!/bin/bash

git pull

#-----------train with trainsize*filter_num

#python SoftT-main.py \
#  --model_name "Qwen/Qwen3-30B-A3B-Instruct-2507"\
#  --exp_name "Qwen3-30B-ETR"\
#  --train_dimension_filter "['nan','Explanations', 'Textual.Evidence', 'Rhetorical.Strategies']" \
#  --resample_train\
#  --train_size 200
#
#
#python SoftT-main.py \
#  --model_name "Qwen/Qwen3-30B-A3B-Instruct-2507"\
#  --exp_name "Qwen3-30B-AT"\
#  --train_dimension_filter "['nan', 'Argument', 'Thesis']" \
#  --resample_train\
#  --train_size 200
#
#
#python SoftT-main.py \
#  --model_name "Qwen/Qwen3-30B-A3B-Instruct-2507"\
#  --exp_name "Qwen3-30B-OL"\
#  --train_dimension_filter "['nan','Organization', 'Language']" \
#  --resample_train\
#  --train_size 200
#




#-----------train with trainsize

python SoftT-main.py \
  --model_name "meta-llama/Llama-3.1-8B-Instruct"\
  --exp_name "Llama-8B-ETR-1000"\
  --train_dimension_filter "['nan','Explanations', 'Textual.Evidence', 'Rhetorical.Strategies']" \
  --resample_train\
  --train_size 1000

python SoftT-main.py \
  --model_name "meta-llama/Llama-3.1-8B-Instruct"\
  --exp_name "Llama-8B-AT-400"\
  --train_dimension_filter "['nan', 'Argument', 'Thesis']" \
  --resample_train\
  --train_size 1000


python SoftT-main.py \
  --model_name "meta-llama/Llama-3.1-8B-Instruct"\
  --exp_name "Llama-8B-OL-400"\
  --train_dimension_filter "['nan','Organization', 'Language']" \
  --resample_train\
  --train_size 1000





#python SoftT-main.py \
#  --model_name "Qwen/Qwen3-30B-A3B-Instruct-2507"\
#  --exp_name "Qwen3-30B"\
#  --train_size 20000 \
#  --gradient_checkpointing
#
#
#python SoftT-main.py \
#  --model_name "Qwen/Qwen3-30B-A3B-Instruct-2507"\
#  --exp_name "Qwen3-30B-ETR"\
#  --train_dimension_filter "['nan','Explanations', 'Textual.Evidence', 'Rhetorical.Strategies']" \
#  --resample_train\
#  --gradient_checkpointing