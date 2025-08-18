import torch
from torch.utils.data import Dataset, DataLoader
from transformers import get_scheduler, GenerationConfig
from torch.optim import AdamW
from tqdm.auto import tqdm
from trl.extras.vllm_client import VLLMClient


class SFTDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        texts = item["text"]
        labels = item["sem_label"]

        return {
            "texts": texts,
            "labels": labels
        }


class small_OLSPTrainer:

    def __init__(self, args, model, tokenizer, train_data):

        self.model = model
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.lr = args.learning_rate

        ## generation setup
        self.num_generations = 1
        self.temperature = 0.6
        self.top_p = .95
        self.top_k = 20
        self.min_p = 0
        self.max_completion_length = args.max_completion_length

        if args.use_vllm:
            ## vllm default as server mode, (from trl)
            ## vllm server initilization
            self.vllm_client = VLLMClient(
                args.vllm_server_host, args.vllm_server_port, connection_timeout=args.vllm_server_timeout
            )
            self.vllm_client.init_communicator()
            self.generate = self.vllm_generate
        else:
            self.generation_config = GenerationConfig(
                max_new_tokens=self.max_completion_length,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=self.min_p,
                # repetition_penalty=self.repetition_penalty,
                # cache_implementation=args.cache_implementation,
            )
            self.generate = self.regular_generate



    def vllm_generate(self,ordered_set_of_prompts, prompt_ids,prompt_mask):
        completion_ids = self.vllm_client.generate(
            prompts=ordered_set_of_prompts,
            n=self.num_generations,
            # repetition_penalty=self.repetition_penalty,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=-1 if self.top_k is None else self.top_k,
            min_p=0.0 if self.min_p is None else self.min_p,
            max_tokens=self.max_completion_length,
            # guided_decoding_regex=self.guided_decoding_regex,
        )
        return completion_ids


    def regular_generate(self,ordered_set_of_prompts, prompt_ids,prompt_mask,return_all):

        prompt_completion_ids = self.model.generate(
                        prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config, output_scores=True
                    )
        if not return_all:
            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
            return completion_ids
        else:
            return prompt_completion_ids

    def customize_train_model(self):
        """
        :param model: 预训练模型
        :param tokenizer: 模型的 tokenizer
        :param train_data: 包含 prompt 和 label 的 List[Dict]
        :param epochs: 训练轮数
        :param batch_size: 批大小
        :param lr: 学习率
        :param max_length: 最大序列长度
        :param device: "cuda" or "cpu"
        :return: 微调后的模型
        """
        device = self.model.device

        dataset = SFTDataset(self.train_data)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=dict_collate_fn)

        model = self.model.to(device)
        optimizer = AdamW(model.get_input_embeddings(), lr=self.lr)

        num_training_steps = self.epochs * len(dataloader)

        lr_scheduler = get_scheduler(
            "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )

        progress_bar = tqdm(range(num_training_steps))


        model.train()
        for epoch in range(self.epochs):
            for batch in dataloader:
                text = batch['text']
                sem_labels = batch['sem_label']

                ## Encoding
                text_encodings = self.tokenizer(text, add_special_tokens=False)

                ## Sampling
                if self.generate == self.regular_generate:
                    outputs = self.generate(ordered_set_of_prompts=text,
                                                        prompt_ids=text_encodings.input_ids.to(self.model.device),
                                                        prompt_mask=text_encodings.attention_mask.to(self.model.device),
                                                        return_all=True)
                    prompt_completions_ids = outputs['sequences']
                    generated_logits = torch.stack(outputs['scores'], dim=1)  # [batch, gen_len, vocab_size]
                    ## TODO

                else:
                    completions_ids = None
                    raise Exception("Not Implemented")


                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss

                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item()}")

        return model


def dict_collate_fn(batch):
    # batch 是 List[Dict]
    collated = {}
    for key in batch[0].keys():
        collated[key] = [sample[key] for sample in batch]
    return collated
