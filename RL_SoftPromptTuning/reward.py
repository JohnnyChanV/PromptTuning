import html
import math
import re
from collections import Counter
import random
from typing import List
from urllib import parse

import requests
from transformers import AutoTokenizer

class Rewards:
    def __init__(self, tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")):
        self.config = {
            'best_cot_length': 256,
        }
        self.tokenizer = tokenizer

    def think_before_judge(self, prompts, completions, **reward_kwargs) -> List[float]:
        rewards = []
        # 遍历每个 completion，计算奖励
        for text in completions:
            reward = 0.0
            # 计算 <think> 和 <answer> 标签的数量
            if text.count("</think>") < 1:
                reward = 0.0
            elif len(re.findall(r'<judge>.*?</judge>', text)) != 0 and text.index("</think>") < text.index("<judge>"):
                reward = 1.0

            rewards.append(reward)
        return [r for r in rewards]

    # weight = 0.2
    def cot_length_reward(self, prompts, completions, **reward_kwargs) -> List[float]:
        rewards = []

        # 遍历每个 completion，计算奖励
        for p, c in zip(prompts, completions):
            if "</think>" in c:
                think = c.split("</think>")[0]
                think_length = len(self.tokenizer.tokenize(think))
                reward = math.sin(think_length / (self.config['best_cot_length'] * 2) * math.pi)
                rewards.append(reward)
            else:
                rewards.append(0.0)

        return [r for r in rewards]

    # weight = 0.2
    def repetition_think_panalty(self, prompts, completions, **reward_kwargs):

        def trigram_repetition_rate(sentence):
            # 分词（假设已经有分词工具）
            words = sentence.split()

            # 提取trigrams
            trigrams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]

            # 计算trigrams频率
            trigram_counts = Counter(trigrams)

            # 计算重复的trigrams数量
            num_repeated = sum(count > 1 for count in trigram_counts.values())

            # 计算重复率
            repetition_rate = num_repeated / len(trigrams) if trigrams else 0
            return repetition_rate

        rewards = []

        for p, c in zip(prompts, completions):
            if "</think>" in c:
                think = c.split("</think>")[0]
                repetition_rate = trigram_repetition_rate(think)
                reward = math.sin((repetition_rate-2)/2*math.pi)+1 # math.sin((x-2)/2*math.pi)+1 三角函数
                rewards.append(reward)
            else:
                rewards.append(0.0)

        return [r for r in rewards]

    # weight = 0.4
    def acc_r(self, completions, ground_truth, **reward_kwargs) -> List[float]:
        # print(reward_kwargs.keys())
        rewards = []

        for text, g_t in zip(completions, ground_truth):
            judge = re.findall(r'<judge>(.*?)</judge>', text)
            if len(judge) != 0:
                judge = judge[0]
            else:
                judge = ''

            if judge.lower() == g_t.lower():
                reward = 1.0
            else:
                reward = 0.0
            rewards.append(reward)
        print("acc_r", rewards)
        return [r for r in rewards]
