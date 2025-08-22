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
  --exp_name "Llama-8B-ETR-500-bs8"\
  --train_dimension_filter "['Explanations', 'Textual.Evidence', 'Rhetorical.Strategies']" \
  --resample_train\
  --train_size 500\
  --batch_size 4

python SoftT-main.py \
  --model_name "meta-llama/Llama-3.1-8B-Instruct"\
  --exp_name "Llama-8B-AT-500-bs8"\
  --train_dimension_filter "['Argument', 'Thesis']" \
  --resample_train\
  --train_size 500\
  --batch_size 8=4


python SoftT-main.py \
  --model_name "meta-llama/Llama-3.1-8B-Instruct"\
  --exp_name "Llama-8B-OL-500-bs8"\
  --train_dimension_filter "['Organization', 'Language']" \
  --resample_train\
  --train_size 500\
  --batch_size 4





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