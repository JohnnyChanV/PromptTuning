#!/bin/bash


python SoftT-main.py \
  --exp_name "Llama-3.2-3B-100-exclude-Organization"\
  --train_dimension_filter "['Explanations', 'Textual.Evidence', 'Rhetorical.Strategies', 'nan', 'Argument', 'Thesis', 'Language']" \


python SoftT-main.py \
  --exp_name "Llama-3.2-3B-100-exclude-Explanations"\
  --train_dimension_filter "['Organization', 'Textual.Evidence', 'Rhetorical.Strategies', 'nan', 'Argument', 'Thesis', 'Language']" \


python SoftT-main.py \
  --exp_name "Llama-3.2-3B-100-exclude-TE"\
  --train_dimension_filter "['Organization', 'Explanations', 'Rhetorical.Strategies', 'nan', 'Argument', 'Thesis', 'Language']" \


python SoftT-main.py \
  --exp_name "Llama-3.2-3B-100-exclude-RS"\
  --train_dimension_filter "['Organization', 'Explanations', 'Textual.Evidence', 'nan', 'Argument', 'Thesis', 'Language']" \


python SoftT-main.py \
  --exp_name "Llama-3.2-3B-100-exclude-Argument"\
  --train_dimension_filter "['Organization', 'Explanations', 'Textual.Evidence', 'nan', 'Thesis', 'Language']" \


python SoftT-main.py \
  --exp_name "Llama-3.2-3B-100-exclude-Thesis"\
  --train_dimension_filter "['Organization', 'Explanations', 'Textual.Evidence', 'nan', 'Argument', 'Language']" \


python SoftT-main.py \
  --exp_name "Llama-3.2-3B-100-exclude-Language"\
  --train_dimension_filter "['Organization', 'Explanations', 'Textual.Evidence', 'nan', 'Argument', 'Thesis']" \

