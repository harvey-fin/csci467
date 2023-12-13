import openai
from datasets import load_dataset
import random
import os
from utils import get_answer_num
import pickle

random.seed(1)

openai.api_key = "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"
prompt = "say helloworld to me."
"""
response = openai.Completion.create(
        engine="text-curie-001",
        prompt=prompt,
        temperature=1,
        max_tokens=10,
        frequency_penalty=0.0,
        logprobs=3
        )
"""

response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
        temperature=1)


import pdb; pdb.set_trace()

print(response)

