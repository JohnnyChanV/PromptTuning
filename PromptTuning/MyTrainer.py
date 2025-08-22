import re
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm
from trl import SFTTrainer
import  torch

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
            # 构造输入
            input_ids = self.processing_class.apply_chat_template(
                example["message"],
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=128,
                    do_sample=False,
                    # temperature=0.6,
                    pad_token_id=self.processing_class.eos_token_id,
                )
            response_ids = outputs[0][input_ids.shape[-1]:]
            response = self.processing_class.decode(response_ids, skip_special_tokens=False)
            pred = parse_value_from_xml_with_regex(response, "answer")

            preds.append(pred)
            labels.append(example["sem_label"])

        acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)
        kappa = cohen_kappa_score(labels, preds)

        metrics = {
            "gen_accuracy": acc,
            "cohen_kappa": kappa
        }
        return metrics
