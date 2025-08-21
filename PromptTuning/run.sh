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
  --model_name "meta-llama/Llama-3.2-3B-Instruct"\
  --exp_name "Llama-3B-ETR-500"\
  --train_dimension_filter "['nan','Explanations', 'Textual.Evidence', 'Rhetorical.Strategies']" \
  --resample_train\
  --train_size 500

python SoftT-main.py \
  --model_name "meta-llama/Llama-3.2-3B-Instruct"\
  --exp_name "Llama-3B-AT-500"\
  --train_dimension_filter "['nan', 'Argument', 'Thesis']" \
  --resample_train\
  --train_size 500


python SoftT-main.py \
  --model_name "meta-llama/Llama-3.2-3B-Instruct"\
  --exp_name "Llama-3B-OL-500"\
  --train_dimension_filter "['nan','Organization', 'Language']" \
  --resample_train\
  --train_size 500





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