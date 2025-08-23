import torch
import torch.nn.functional as F
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
    def evaluate(self, eval_dataset=None, **kwargs):
        dataset = eval_dataset or self.eval_dataset
        if dataset is None:
            raise ValueError("No eval_dataset provided")

        self.model.eval()
        preds, labels = [], []
        for example in tqdm(dataset):
            # 构造两个候选
            pos_ids = self.processing_class.apply_chat_template(
                example["message"] + [
                    {"role": "assistant", "content": "<answer>With Explanation</answer>"}],
                add_generation_prompt=False,
                return_tensors="pt"
            )
            neg_ids = self.processing_class.apply_chat_template(
                example["message"] + [
                    {"role": "assistant", "content": "<answer>Without Explanation</answer>"}],
                add_generation_prompt=False,
                return_tensors="pt"
            )

            # 拼成 batch，一次 forward
            input_ids = torch.cat([pos_ids, neg_ids], dim=0).to(self.model.device)
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits

            # 计算每个序列的 log-likelihood (逐 token)
            seq_logprobs = []
            for i in range(input_ids.size(0)):
                ids = input_ids[i]
                # shift：预测 token 对应 label = 下一 token
                target_ids = ids[1:]
                pred_logits = logits[i, :-1, :]   # 对应 target_ids
                log_probs = F.log_softmax(pred_logits, dim=-1)
                token_log_probs = log_probs[range(len(target_ids)), target_ids]
                seq_logprobs.append(token_log_probs.sum().item())

            pos_score, neg_score = seq_logprobs
            pred = "With Explanation" if pos_score > neg_score else "Without Explanation"

            preds.append(pred)
            labels.append(example["sem_label"])

        # 计算指标
        acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)
        kappa = cohen_kappa_score(labels, preds)

        metrics = {
            "gen_accuracy": acc,
            "cohen_kappa": kappa
        }
        print(metrics)
        swanlab.log(metrics)
        return metrics
