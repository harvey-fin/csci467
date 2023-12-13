import numpy as np
from utils import *
import openai
from typing import *
from datasets import load_dataset
import pickle
import os
from tqdm import tqdm

def run_prm_gpt3():
    file_name = "files/prm/gpt-3.pkl"
    if os.path.exists(file_name):
        print("Result exist!")
        return
    prompts, labels = load_prm_all()
    
    accs = []
    probs = []

    for prompt, label in tqdm(zip(prompts, labels)):
        l, p = gpt3_checker(gpt3_inference(prompt))
        accs.append(l==label)
        probs.append(p)
    
    ret_dict = {"accs": accs, "probs": probs}
    with open(file_name, "wb") as f:
        pickle.dump(ret_dict, f)


def gpt3_checker(response) -> Union[int, float]:
    label_space = ["No", "no", "Maybe", "maybe", "Yes", "yes"]
    label_space_grouped = [["No", "no"], ["Maybe", "maybe"], ["Yes", "yes"]]
    probs = response["choices"][0]["logprobs"]

    tokens = probs["tokens"]
    log_probs = probs["top_logprobs"]

    idx = 10
    for t in tokens:
        if t in label_space:
            idx = tokens.index(t)
            label_space_idx = label_space.index(t)//2
            ans = t
    if idx == 10:
        return 0, 0.0

    try:
        log_probs = log_probs[idx]
    except IndexError as e:
        return 0, 0.0
    
    total_sum = 0
    choice_sum = 0
    for tok, p in log_probs.items():
        total_sum += np.exp(p)
        if tok in label_space_grouped[label_space_idx]:
            choice_sum += np.exp(p)

    return label_space_idx-1, choice_sum/total_sum


def gpt3_inference(prompt: str):
    """
    pass the prompt to gpt3 and get the response as a string
    """
    openai.api_key = "" #Enter your API key here

    try:
        response = openai.Completion.create(
                model="text-curie-001",
                prompt=prompt,
                max_tokens=1024,
                temperature=0,
                logprobs=30)
    except openai.error.RateLimitError as e:
        time.sleep(60)
        return "error"
    except openai.error.APIError as e:
        time.sleep(60)
        return "error"
    except openai.error.Timeout as e:
        time.sleep(60)
        return "error"

    return response


if __name__ == "__main__":
    run_prm_gpt3()
