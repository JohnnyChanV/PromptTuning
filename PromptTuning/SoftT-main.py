#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Soft-Prompt (Prefix) Tuning with TRL SFTTrainer
- 通过 argparse 管理配置
- 尽可能函数化，便于复用与测试
"""

import argparse
import json
import re
from collections import Counter
from functools import partial
from typing import List, Tuple, Dict, Any, Optional
import random
import torch
import pandas as pd  # 仅保留以兼容原始依赖（未直接使用）
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from tqdm import tqdm


# =========================
# 默认配置与常量
# =========================

SEMANTIC_LABEL = {
    0: "Without Explanation",
    1: "With Explanation",
}

DEFAULT_PROMPT_TEMPLATE = "<comment>{}</comment>"


# =========================
# 工具函数
# =========================

def parse_value_from_xml_with_regex(xml_string: str, tag_name: str) -> str:
    """从简单 XML 片段中提取内容"""
    pattern = f"<{tag_name}>(.*?)</{tag_name}>"
    match = re.search(pattern, xml_string, re.DOTALL)
    return match.group(1) if match else ""


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def prepare_model_and_tokenizer(model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """加载模型与分词器，并设置 pad_token"""
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def build_prefix_tokens(
    model_name: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    num_prefix_tokens: int,
    force_strategy: Optional[str] = None,
) -> Tuple[List[str], List[int], AutoTokenizer, AutoModelForCausalLM]:
    """
    构造 prefix token 列表与其 id，并在需要时扩展词表。
    strategy:
      - "llama_reserved": 使用 Llama 已保留的 <|reserved_special_token_i|>
      - "add_tokens": 为非 Llama 模型新增 <softP_i> 并 resize embedding
      - None: 自动（model_name 含 "llama" 则使用 reserved，否则 add）
    """
    if force_strategy not in {None, "llama_reserved", "add_tokens"}:
        raise ValueError("--force_strategy 只能是 None / 'llama_reserved' / 'add_tokens'")

    is_llama = "llama" in model_name.lower()
    strategy = force_strategy or ("llama_reserved" if is_llama else "add_tokens")

    prefix_token_strs: List[str] = []
    prefix_token_ids: List[int] = []

    if strategy == "llama_reserved":
        prefix_token_strs = [f"<|reserved_special_token_{i}|>" for i in range(num_prefix_tokens)]
        prefix_token_ids = tokenizer.convert_tokens_to_ids(prefix_token_strs)
    else:
        prefix_token_strs = [f"<|reserved_special_token_{i}|>" for i in range(num_prefix_tokens)]
        num_added = tokenizer.add_tokens(prefix_token_strs)
        # 如果新增了 token，需要扩展 embedding
        if num_added > 0 and model.get_input_embeddings().weight.shape[0] < len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))
            print("[Info]: Model EMBEDDING RESIZED")
        prefix_token_ids = tokenizer.convert_tokens_to_ids(prefix_token_strs)
        with torch.no_grad():
            emb = model.get_input_embeddings().weight
            emb[prefix_token_ids] = torch.empty_like(emb[prefix_token_ids]).normal_(mean=0.0, std=0.02)
        print("Embedding Initialization Results:\n",model.get_input_embeddings().weight[prefix_token_ids])
        print(f"Added {num_added} tokens: {prefix_token_ids[:10]}{'...' if len(prefix_token_ids) > 10 else ''}")
        print(f"[INFO] Model Embedding size: {model.get_input_embeddings().weight.shape[0]}. \n [INFO] Tokenizer vocab size: {len(tokenizer)}")

    return prefix_token_strs, prefix_token_ids, tokenizer, model


def prepare_train_data(
    path: str,
    semantic_label_map: Dict[int, str],
) -> List[Dict[str, Any]]:
    data = load_json(path)
    for item in data:
        # 兼容原始字段
        item["sem_label"] = semantic_label_map[item["label"]]
        item["Dimension.Name"] = str(item.get("Dimension.Name", ""))
    # 打印分布，便于 sanity check
    if len(eval(args.train_dimension_filter)) > 0:
        new_data = [item for item in data if item['Dimension.Name'] in eval(args.train_dimension_filter)]
        data = new_data
    # print(len(data))

    cnt = Counter([item["sem_label"] for item in data])

    if args.resample_train:
        max_category = max(cnt.values())
        resample_set = []
        for each in cnt.keys():
            resample_set += random.choices([item for item in data if item['sem_label'] == each],k=max_category - cnt[each])
        data += resample_set


    print(f"[Info] label distribution (top 10): {cnt}")

    if len(eval(args.train_dimension_filter))!=0:
        times = len(eval(args.train_dimension_filter))
    else:
        times = 1

    print(f"[Info] Training set right bound: {args.train_size*times}")
    return data[:args.train_size*times]


def dataset_with_messages(
    data: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    system_prompt: str,
    prompt_template: str,
    add_answer: bool = True,
    seed: int = 42,
) -> Dataset:
    """
    根据是否包含答案，生成 'text' 和 'message' 字段，用于 SFTTrainer。
    """
    def _with_answer(row: Dict[str, Any]) -> Dict[str, Any]:
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_template.format(row["input"])},
            {"role": "assistant", "content": "<answer>" + row["sem_label"] + "</answer>"},
        ]
        return {
            "text": tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False
            ),
            "message": msgs,
        }

    def _no_answer(row: Dict[str, Any]) -> Dict[str, Any]:
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_template.format(row["input"])},
        ]
        return {
            "text": tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False
            ),
            "message": msgs,
        }

    ds = Dataset.from_list(data).shuffle(seed=seed)
    return ds.map(_with_answer if add_answer else _no_answer)


def freeze_all_but_prefix_embeddings(
    model: AutoModelForCausalLM,
    prefix_token_ids: List[int],
    verbose: bool = True,
):
    """
    冻结除 embedding 行（对应 prefix token id）之外的所有参数；
    并在 embedding 权重上注册 grad hook，只保留这些行的梯度。
    返回 hook handle，训练结束后需移除。
    """
    # 冻结全部参数
    for p in model.parameters():
        p.requires_grad = False

    # 只训练词嵌入（整张表先解冻，再用 hook 只放行特定行）
    emb = model.get_input_embeddings()
    emb.weight.requires_grad = True

    e2u = torch.tensor(prefix_token_ids, dtype=torch.long, device=emb.weight.device)

    if verbose:
        print(f"[Info] Trainable embedding rows (prefix tokens): {e2u.tolist()[:20]}{'...' if len(e2u) > 20 else ''}")

    def grad_hook(grad: torch.Tensor) -> torch.Tensor:
        print(grad.shape)
        print("梯度不为0数量：",(grad.abs().sum(-1) != 0).sum().item())
        mask = torch.zeros_like(grad)
        mask[e2u] = 1.0
        print("梯度不为0数量：",((grad * mask).abs().sum(-1) != 0).sum().item())
        return grad * mask

    handle = emb.weight.register_hook(grad_hook)
    return handle


def build_data_collator(tokenizer: AutoTokenizer, model_name: str) -> DataCollatorForCompletionOnlyLM:
    # 仅在 assistant 段回传梯度
    if "llama" in model_name.lower():
        response_template = "<|start_header_id|>assistant<|end_header_id|>"
    else:
        response_template = "<|im_start|>assistant\n"
    return DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)


def build_trainer(
    model: AutoModelForCausalLM,
    train_dataset: Dataset,
    data_collator,
    args: argparse.Namespace,
) -> SFTTrainer:
    targs = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        fp16=args.fp16,
        bf16=args.bf16,
        logging_steps=1,
        weight_decay=args.weight_decay,
        lr_scheduler_type="linear",
        seed=args.seed,
        output_dir=args.output_dir + args.exp_name,
        gradient_checkpointing=args.gradient_checkpointing,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=targs,
        data_collator=data_collator,
    )
    return trainer


def run_eval_sample(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages_list: List[List[Dict[str, str]]],
    gold_labels: List[str],
    sample_n: int = 10,
) -> None:
    """
    简单样例评估：基于若干条样本，采用贪心解码，抽取 <answer>…</answer>
    """
    model.eval()
    pred_ls, golden_ls = [], []
    num_correct, num_total = 0, 0

    n = min(sample_n, len(messages_list))
    print(f"[Eval] Running eval on {n} samples...")

    for i in tqdm(range(n)):
        messages = messages_list[i]
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        response_ids = outputs[0][input_ids.shape[-1]:]
        response = tokenizer.decode(response_ids, skip_special_tokens=False)
        pred = parse_value_from_xml_with_regex(response, "answer")

        pred_ls.append(pred)
        golden_ls.append(gold_labels[i])

        if pred == gold_labels[i]:
            num_correct += 1
        num_total += 1

    if num_total > 0:
        acc = num_correct / num_total
        print(f"[Eval] Accuracy on {n} samples: {acc:.4f}")


# =========================
# 主流程
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Soft-Prompt tuning (prefix tokens) with TRL SFTTrainer"
    )

    # 模型 & 数据
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--train_data", type=str, default="../RAG_data/proc_dev_data.json")
    parser.add_argument("--train_dimension_filter", type=str, default="[]")
    parser.add_argument("--train_size", type=int, default=200)
    # choices = ['Organization', 'Explanations', 'Textual.Evidence', 'Rhetorical.Strategies', 'nan', 'Argument', 'Thesis', 'Language']
    parser.add_argument("--resample_train", action="store_true", default=False)
    parser.add_argument("--system_prompt_file", type=str, default="good_prompt.txt")
    parser.add_argument("--prompt_template", type=str, default=DEFAULT_PROMPT_TEMPLATE)

    # prefix
    parser.add_argument("--num_prefix_tokens", type=int, default=30)
    parser.add_argument("--force_strategy", type=str, default=None,
                        choices=[None, "llama_reserved", "add_tokens"],
                        help="强制选择前缀策略；默认自动（Llama 用 reserved，其它模型 add tokens）")

    # 训练超参
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp_name", type=str, default="Llama-3.2-3B-100-domain")
    parser.add_argument("--output_dir", type=str, default="/ix1/xli/zhc195/SoftPrompt_Tuning/outputs/")

    # 精度 & 训练细节
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--save_strategy", type=str, default="steps", choices=["no", "epoch", "steps"])
    parser.add_argument("--save_steps", type=int, default=600)

    # 评估（可选）
    parser.add_argument("--eval_sample_n", type=int, default=0,
                        help=">0 时，在训练后对若干条样本做简单评估")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    print(f"[Config] {args}")
    random.seed(args.seed)

    # 1) 加载模型与分词器
    model, tokenizer = prepare_model_and_tokenizer(args.model_name)

    # 2) 构造 prefix tokens（可能会扩展词表）
    prefix_token_strs, prefix_token_ids, tokenizer, model = build_prefix_tokens(
        model_name=args.model_name,
        tokenizer=tokenizer,
        model=model,
        num_prefix_tokens=args.num_prefix_tokens,
        force_strategy=args.force_strategy,
    )

    # 3) 读取 system prompt，并将 prefix 拼接到最前面
    prefix = "".join(prefix_token_strs[: args.num_prefix_tokens])
    system_prompt_raw = read_text(args.system_prompt_file)
    system_prompt = prefix + "\n" + system_prompt_raw
    print(system_prompt)

    # 4) 准备训练数据与数据集
    train_data = prepare_train_data(args.train_data, SEMANTIC_LABEL)
    train_dataset = dataset_with_messages(
        train_data,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        prompt_template=args.prompt_template,
        add_answer=True,
        seed=args.seed,
    )

    # 5) 冻结参数，仅训练 prefix 对应的 embedding 行（通过 grad hook 局部放行）
    hook_handle = freeze_all_but_prefix_embeddings(model, prefix_token_ids, verbose=True)

    # 6) collator & trainer
    collator = build_data_collator(tokenizer, args.model_name)
    trainer = build_trainer(model, train_dataset, collator, args)

    # 7) 训练
    trainer.train()

    # 8) 清理 hook
    hook_handle.remove()
    model.eval()

    # 9) 可选：快速评估若干条样本（用训练数据的 message 演示）
    if args.eval_sample_n and args.eval_sample_n > 0:
        messages_list = [ex["message"] for ex in train_dataset.select(range(min(args.eval_sample_n, len(train_dataset))))]
        gold_labels = [ex["sem_label"] for ex in train_data[: len(messages_list)]]
        run_eval_sample(model, tokenizer, messages_list, gold_labels, sample_n=args.eval_sample_n)

    print("[Done] Training finished.")