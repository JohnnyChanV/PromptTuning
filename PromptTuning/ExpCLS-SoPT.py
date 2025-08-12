import json
import pandas as pd

import re
import torch

from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from tqdm import tqdm

NUM_SPECIAL_TOKENS_IN_PREFIX = 30

# NUM_SPECIAL_TOKENS_IN_PREFIX = 128
LEARNING_RATE = 5e-5
BATCH_SIZE = 4
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01

SEMANTIC_LABEL = {
    0:"Without Explanation",
    1:"With Explanation"
}


#--------------



# model_name = "meta-llama/Llama-3.2-3B-Instruct"
model_name = "meta-llama/Llama-3.1-8B-Instruct"
# model_name = "Qwen/Qwen3-4B"
# model_name = "Qwen/Qwen2.5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# terminators = [
#     tokenizer.eos_token_id,
#     tokenizer.convert_tokens_to_ids("<|eot_id|>")
# ]

#--------------

prefix_token_strs, prefix_token_ids = [], []

if 'llama' in model_name.lower():
    # llama reserved 250 special tokens
    for i in range(NUM_SPECIAL_TOKENS_IN_PREFIX):
        prefix_token_strs.append(f"<|reserved_special_token_{i}|>")
    
    prefix_token_ids = tokenizer.convert_tokens_to_ids(prefix_token_strs)
else:
    # 要添加的新token（比如 soft prompt tokens 或特殊符号）
    prefix_token_strs = [f"<softP_{i}>" for i in range(NUM_SPECIAL_TOKENS_IN_PREFIX)]
    num_added_tokens = tokenizer.add_tokens(prefix_token_strs)

    # 添加token，如果已存在则不会重复添加
    model.resize_token_embeddings(len(tokenizer))
    prefix_token_ids = tokenizer.convert_tokens_to_ids(prefix_token_strs)
    print(f"Added {num_added_tokens} tokens: {prefix_token_ids}")

#----------

train_data = json.load(open("RAG_data/proc_ret_data.json",'r'))
for item in train_data:
    item['sem_label'] = SEMANTIC_LABEL[item['label']]
    del item['Dimension.Name']

test_data = json.load(open("RAG_data/proc_test_data.json",'r'))
for item in test_data:
    item['sem_label'] = SEMANTIC_LABEL[item['label']]
    del item['Dimension.Name']
#----------

train_dataset = Dataset.from_list(train_data).shuffle(seed=42)
test_dataset = Dataset.from_list(test_data)

#----------
system_prompt = open("good_prompt.txt").read()

prompt_template = "<comment>{}</comment>"


#----------

def parse_value_from_xml_with_regex(xml_string, tag_name):
    
    pattern = f'<{tag_name}>(.*?)</{tag_name}>'
    match = re.search(pattern, xml_string, re.DOTALL)  # re.DOTALL allows matching across multiple lines
    
    if match:
        return match.group(1)
    else:
        return ""
#----------

def create_test_messages(row):
    # row['messages'] = [
    #     {"role": "system", "content": system_prompt},
    #     {"role": "user", "content": prompt_template.format(row["input"])}
    # ]
    return {"messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_template.format(row["input"])}
    ]}

test_dataset_no_prefix = test_dataset.map(create_test_messages)


#----------Train

# Setting hyperparameters



# prefix will be comprised of n special tokens 
prefix = "".join(prefix_token_strs[:NUM_SPECIAL_TOKENS_IN_PREFIX])
system_prompt = system_prompt + prefix
# We create a training dataset that includes the answer
# We also create another test dataset, this time with the prefix for the finetuned model

def create_prefix_messages(row):
    return {"text": 
            tokenizer.apply_chat_template([
                    {"role": "system", "content": system_prompt },
                    {"role": "user", "content": prompt_template.format(row["input"])},
                    {"role": "assistant", "content": "<answer>" + row["sem_label"] + "</answer>"}
                ],tokenize=False,add_generation_prompt=False,enable_thinking=False),

            "message":[
                    {"role": "system", "content": system_prompt },
                    {"role": "user", "content": prompt_template.format(row["input"])},
                    {"role": "assistant", "content": "<answer>" + row["sem_label"] + "</answer>"}
                ]
           }

def create_prefix_messages_no_answer(row):
    return {"text": tokenizer.apply_chat_template([
        {"role": "system", "content": system_prompt },
        {"role": "user", "content": prompt_template.format(row["input"])}
    ],tokenize=False,add_generation_prompt=False,enable_thinking=False),
           "message":[
        {"role": "system", "content": system_prompt },
        {"role": "user", "content": prompt_template.format(row["input"])}
    ]}


train_dataset = train_dataset.map(create_prefix_messages)
test_dataset_with_prefix = test_dataset.map(create_prefix_messages_no_answer)

# train_dataset[:2]



# pred_ls, golden_ls = [], []
# num_correct, num_total = 0, 0

# # for i in tqdm(range(len(test_dataset_no_prefix))):
# for i in tqdm(range(10)):
#     messages = test_dataset_with_prefix[i]["message"]
    
#     input_ids = tokenizer.apply_chat_template(
#         messages,
#         add_generation_prompt=True,
#         enable_thinking=False,
#         return_tensors="pt",
#     ).to(model.device)
    
#     outputs = model.generate(
#         input_ids,
#         max_new_tokens=128,
#         # eos_token_id=terminators,
#         do_sample=False,
#         # temperature=0.6,
#         # top_p=0,
#         pad_token_id=tokenizer.eos_token_id,
#         # past_key_value=None
#     )
    
#     response = outputs[0][input_ids.shape[-1]:]
#     response = tokenizer.decode(response, skip_special_tokens=False)
#     print(response)
#     pred = parse_value_from_xml_with_regex(response, "answer")
    
#     pred_ls.append(pred)
#     golden_ls.append(test_dataset_no_prefix[i]["sem_label"])
    
#     if pred == test_dataset_no_prefix[i]["sem_label"]:
#         num_correct += 1
#     num_total += 1

# accuracy = num_correct / num_total
# print(f"Accuracy: {accuracy}")



#--------gradient blocking:--------

# Freeze all parameters except the embedding layer
# Add the hook to zero out non-special token gradients

for param in model.parameters():
    param.requires_grad = False


embeddings_to_update = torch.tensor(prefix_token_ids[:NUM_SPECIAL_TOKENS_IN_PREFIX], dtype=torch.long)

# Ensure embeddings_to_update is on the correct device
embeddings_to_update = embeddings_to_update.to(model.device)

model.get_input_embeddings().weight.requires_grad = True


def grad_hook(grad):
    mask = torch.zeros_like(grad)
    mask[embeddings_to_update] = 1.0
    
    masked_grad = grad * mask
    return masked_grad

hook_handle = model.get_input_embeddings().weight.register_hook(grad_hook)

#-------# only train on completion tokens--------

# only train on completion tokens
if "llama" in model_name.lower():
    response_template = "<|start_header_id|>assistant<|end_header_id|>"
else:
    response_template = "<|im_start|>assistant"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

trainer = SFTTrainer(
    model,
    train_dataset=train_dataset,
    data_collator=collator,
    args = TrainingArguments(
        per_device_train_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = 1,
        warmup_ratio = WARMUP_RATIO,
        num_train_epochs = 5, # Set this for 1 full training run.
        learning_rate = LEARNING_RATE,
        fp16 = False, # switch these depending if you're GPU supports BF16
        bf16 = True,
        logging_steps = 1,
        # optim = "adamw_8bit",
        weight_decay = WEIGHT_DECAY,
        lr_scheduler_type = "linear",
        seed = 42,
        output_dir = "/ix1/xli/zhc195/SoftPrompt_Tuning/SoftP-llama-8b-200-outputs",
        gradient_checkpointing=False,
        save_strategy="steps",  # 👈 每个 epoch 保存一次模型
        save_steps=250
    )
)

trainer.train()
hook_handle.remove()
model.eval()

# -----------


# Running on sample of test dataset; this time with the newly trained prefix

pred_ls, golden_ls = [], []
num_correct, num_total = 0, 0

# for i in tqdm(range(len(test_dataset_no_prefix))):
for i in tqdm(range(10)):
    messages = test_dataset_with_prefix[i]["message"]
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    ).to(model.device)
    
    outputs = model.generate(
        input_ids,
        max_new_tokens=128,
        # eos_token_id=terminators,
        do_sample=False,
        # temperature=0.6,
        # top_p=0,
        pad_token_id=tokenizer.eos_token_id,
        # past_key_value=None
    )
    
    response = outputs[0][input_ids.shape[-1]:]
    response = tokenizer.decode(response, skip_special_tokens=False)
    print(response)
    pred = parse_value_from_xml_with_regex(response, "answer")
    
    pred_ls.append(pred)
    golden_ls.append(test_dataset_no_prefix[i]["sem_label"])
    
    if pred == test_dataset_no_prefix[i]["sem_label"]:
        num_correct += 1
    num_total += 1

accuracy = num_correct / num_total
print(f"Accuracy: {accuracy}")

# print(response)




