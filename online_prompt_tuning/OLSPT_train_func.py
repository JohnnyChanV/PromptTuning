import torch
from torch.utils.data import Dataset, DataLoader
from transformers import get_scheduler, GenerationConfig
from torch.optim import AdamW
from tqdm.auto import tqdm
from trl.extras.vllm_client import VLLMClient
import torch.nn.functional as F


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


class OLSPTraining:

    def __init__(self, args, model, tokenizer, train_data, answer_seperator="</think>"):

        self.model = model
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.lr = args.learning_rate
        self.answer_seperator=answer_seperator

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
                stop_strings=self.answer_seperator
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

    def regular_generate(self, ordered_set_of_prompts, prompt_ids, prompt_mask, return_all):
        prompt_completion_out = self.model.generate(
            prompt_ids,
            attention_mask=prompt_mask,
            generation_config=self.generation_config,
            output_scores=True,
            return_dict_in_generate=True,  # <-- important
        )
        if not return_all:
            prompt_length = prompt_ids.size(1)
            # sequences = [B, prompt_len + gen_len]
            completion_ids = prompt_completion_out.sequences[:, prompt_length:]
            return completion_ids
        else:
            return prompt_completion_out  # has .sequences and .scores

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
                    # ===== TODO 开始 =====

                    # 保障右侧补齐与 pad/eos 合理性
                    self.tokenizer.padding_side = "right"
                    if self.tokenizer.pad_token_id is None:
                        if self.tokenizer.eos_token_id is None:
                            raise ValueError("tokenizer.eos_token_id 为空，请先设置 eos token。")
                        # 常见做法：用 eos 作为 pad
                        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                        self.tokenizer.pad_token = self.tokenizer.eos_token

                    eos_id = self.tokenizer.eos_token_id
                    if eos_id is None:
                        raise ValueError("tokenizer.eos_token_id 为空，请先设置 eos token。")

                    device = self.model.device
                    pc_ids = prompt_completions_ids.to(device)  # [B, L_pc]（当前“prompt+生成”序列）
                    B, L_pc = pc_ids.size()

                    # 1) 初始化 golden label_ids，全 -100，与当前序列等长
                    label_ids = torch.full_like(pc_ids, fill_value=-100)

                    # 2) 编码 sem_label，右侧 padding，并在每条序列末尾追加 EOS
                    #    注意：不使用 add_special_tokens，由我们手动加 eos
                    sem_tok = self.tokenizer(sem_labels, add_special_tokens=False)
                    sem_input_ids = [ids + [eos_id] for ids in sem_tok.input_ids]  # List[List[int]]
                    sem_attention_mask = [[1] * len(ids) for ids in sem_input_ids]  # 真实 token 置 1

                    sem_batch = self.tokenizer.pad(
                        {"input_ids": sem_input_ids, "attention_mask": sem_attention_mask},
                        padding=True,
                        return_tensors="pt",
                        # 如需对齐 Tensor Core，可加：pad_to_multiple_of=8
                    )

                    sem_ids = sem_batch.input_ids.to(device)  # [B, L_sem]
                    sem_amask = sem_batch.attention_mask.to(device)  # [B, L_sem]

                    # 将 sem_label 的 token 拼到 label_ids 后，只在这些位置计算损失
                    labels_concat = torch.cat([label_ids, sem_ids], dim=1)  # [B, L_pc + L_sem]

                    # 3) 将 sem_label 的 token 拼到当前的 prompt_completions_ids 后，作为新输入
                    input_ids_concat = torch.cat([pc_ids, sem_ids], dim=1)  # [B, L_pc + L_sem]
                    attn_mask_concat = torch.cat(
                        [torch.ones((B, L_pc), dtype=torch.long, device=device), sem_amask],
                        dim=1
                    )  # [B, L_pc + L_sem]

                    # 4) 前向得到 logits，计算 CE（忽略 -100），标准 LM 左/右移对齐
                    lm_out = model(input_ids=input_ids_concat, attention_mask=attn_mask_concat, use_cache=False)
                    logits = lm_out.logits  # [B, L_total, V]

                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = labels_concat[:, 1:].contiguous()

                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                    )

                    # ===== TODO 结束 =====

                else:
                    completions_ids = None
                    raise Exception("Not Implemented")


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
