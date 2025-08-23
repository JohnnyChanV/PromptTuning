import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm
import swanlab
from trl import SFTTrainer
import re


def parse_value_from_xml_with_regex(xml_string: str, tag_name: str) -> str:
    """从简单 XML 片段中提取内容"""
    pattern = f"<{tag_name}>(.*?)</{tag_name}>"
    match = re.search(pattern, xml_string, re.DOTALL)
    return match.group(1) if match else ""


class MyTrainer(SFTTrainer):
    def evaluate(self, eval_dataset=None, batch_size: int = 16, **kwargs):
        dataset = eval_dataset or self.eval_dataset
        if dataset is None:
            raise ValueError("No eval_dataset provided")

        self.model.eval()
        preds, labels = [], []

        for start in tqdm(range(0, len(dataset), batch_size)):
            batch = dataset[start:start + batch_size]

            # 收集每个 example 的候选输入
            all_ids = []
            all_labels = []
            for example in batch:
                pos_ids = self.processing_class.apply_chat_template(
                    example["message"] + [
                        {"role": "assistant", "content": "<answer>With Explanation</answer>"}],
                    add_generation_prompt=False,
                    return_tensors="pt"
                ).unsqueeze(0)

                neg_ids = self.processing_class.apply_chat_template(
                    example["message"] + [
                        {"role": "assistant", "content": "<answer>Without Explanation</answer>"}],
                    add_generation_prompt=False,
                    return_tensors="pt"
                ).unsqueeze(0)

                all_ids.append(pos_ids)
                all_ids.append(neg_ids)
                all_labels.append(example["sem_label"])

            # pad 成统一长度 (2 * batch_size, max_len)
            padded_ids = pad_sequence(all_ids, batch_first=True,
                                      padding_value=self.processing_class.pad_token_id).to(self.model.device)

            # attention mask
            attention_mask = (padded_ids != self.processing_class.pad_token_id).long()

            with torch.no_grad():
                outputs = self.model(padded_ids, attention_mask=attention_mask)
                logits = outputs.logits

            # 计算 log-likelihood
            seq_logprobs = []
            for i, ids in enumerate(all_ids):
                length = ids.size(0)
                target_ids = padded_ids[i, 1:length].to(self.model.device)
                pred_logits = logits[i, :length-1, :]
                log_probs = F.log_softmax(pred_logits, dim=-1)
                token_log_probs = log_probs[range(len(target_ids)), target_ids]
                seq_logprobs.append(token_log_probs.sum().item())

            # reshape -> (batch_size, 2)
            seq_logprobs = torch.tensor(seq_logprobs).view(len(batch), 2)

            for i, gold in enumerate(all_labels):
                pos_score, neg_score = seq_logprobs[i]
                pred = "With Explanation" if pos_score > neg_score else "Without Explanation"
                preds.append(pred)
                labels.append(gold)

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
