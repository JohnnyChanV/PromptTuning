import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score
import swanlab
from trl import SFTTrainer
from tqdm import tqdm


class CandidateCollator:
    def __init__(self, processing_class, pad_token_id, candidates=None):
        self.processing_class = processing_class
        self.pad_token_id = pad_token_id
        # 候选答案（可扩展）
        self.candidates = candidates or [
            "With Explanation",
            "Without Explanation"
        ]

    def __call__(self, examples):
        all_ids = []
        candidate_index = []
        labels = []

        for ex_idx, example in enumerate(examples):
            for cand_idx, cand in enumerate(self.candidates):
                ids = self.processing_class.apply_chat_template(
                    example["message"] + [
                        {"role": "assistant", "content": f"<answer>{cand}</answer>"}],
                    add_generation_prompt=False,
                    enable_thinking=False,
                    return_tensors="pt"
                )[0]
                all_ids.append(ids)
                candidate_index.append((ex_idx, cand_idx))
            labels.append(example["sem_label"])

        # pad 成统一长度
        padded_ids = pad_sequence(
            all_ids,
            batch_first=True,
            padding_value=self.pad_token_id
        )
        attention_mask = (padded_ids != self.pad_token_id).long()

        return {
            "input_ids": padded_ids,
            "attention_mask": attention_mask,
            "candidate_index": candidate_index,  # (ex_idx, cand_idx)
            "labels": labels
        }


class MyTrainer(SFTTrainer):
    def evaluate(self, eval_dataset=None, batch_size: int = 16, **kwargs):
        dataset = eval_dataset or self.eval_dataset
        if dataset is None:
            raise ValueError("No eval_dataset provided")

        collator = CandidateCollator(
            processing_class=self.processing_class,
            pad_token_id=self.processing_class.pad_token_id
        )
        loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)

        self.model.eval()
        preds, labels = [], []

        for batch in tqdm(loader):
            input_ids = batch["input_ids"].to(self.model.device)
            attention_mask = batch["attention_mask"].to(self.model.device)
            labels.extend(batch["labels"])
            candidate_index = batch["candidate_index"]

            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits

            # 计算 log-likelihood
            seq_logprobs = []
            for i in range(input_ids.size(0)):
                ids = input_ids[i]
                length = attention_mask[i].sum().item()
                target_ids = ids[1:length]
                pred_logits = logits[i, :length-1, :]
                log_probs = F.log_softmax(pred_logits, dim=-1)
                token_log_probs = log_probs[range(len(target_ids)), target_ids]
                seq_logprobs.append(token_log_probs.sum().item())

            # 按 example 分组
            from collections import defaultdict
            score_dict = defaultdict(dict)
            for (ex_idx, cand_idx), score in zip(candidate_index, seq_logprobs):
                score_dict[ex_idx][cand_idx] = score

            for ex_idx in sorted(score_dict.keys()):
                cand_scores = score_dict[ex_idx]
                best_cand = max(cand_scores, key=cand_scores.get)
                pred = collator.candidates[best_cand]
                preds.append(pred)

        # 指标
        acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)
        kappa = cohen_kappa_score(labels, preds)

        metrics = {
            "gen_accuracy": acc,
            "cohen_kappa": kappa
        }
        print(metrics)
        swanlab.log(metrics)
        return metrics
