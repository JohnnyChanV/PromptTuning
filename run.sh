#!/bin/bash


python SoftT-main.py \
  --exp_name "Llama-3.1-8B-100-Organization"\
  --train_dimension_filter "['Organization']" \


python SoftT-main.py \
  --exp_name "Llama-3.1-8B-100-Explanations"\
  --train_dimension_filter "['Explanations']" \


python SoftT-main.py \
  --exp_name "Llama-3.1-8B-100-TE"\
  --train_dimension_filter "['Textual.Evidence']" \


python SoftT-main.py \
  --exp_name "Llama-3.1-8B-100-RS"\
  --train_dimension_filter "['Rhetorical.Strategies']" \


python SoftT-main.py \
  --exp_name "Llama-3.1-8B-100-Argument"\
  --train_dimension_filter "['Argument']" \


python SoftT-main.py \
  --exp_name "Llama-3.1-8B-100-Thesis"\
  --train_dimension_filter "['Thesis']" \


python SoftT-main.py \
  --exp_name "Llama-3.1-8B-100-Language"\
  --train_dimension_filter "['Language']" \
