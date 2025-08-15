#!/bin/bash


python SoftT-main.py \
  --exp_name "Llama-3.2-3B-100-exclude-Organization"\
  --train_dimension_filter "['nan','Explanations', 'Textual.Evidence', 'Rhetorical.Strategies']" \


python SoftT-main.py \
  --exp_name "Llama-3.2-3B-100-exclude-Explanations"\
  --train_dimension_filter "['nan', 'Argument', 'Thesis']" \


python SoftT-main.py \
  --exp_name "Llama-3.2-3B-100-exclude-TE"\
  --train_dimension_filter "['nan','Organization', 'Language']" \
