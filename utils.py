from datasets import load_dataset
import re
from typing import *
import random
import numpy as np
import pandas as pd
import pickle


def load_prm_all():
    """
    Load PRM800K datasets 
    prompt in a step-by-step NLI task format
    """
    prompts = []
    labels = []
    data = load_dataset("Birchlabs/openai-prm800k-phase1_train-stepwise-critique")["train"][:1000]
    d = {"instruction": data["instruction"], "responses": data["responses"], "next_response": data["next_response"], "rating": data["rating"]}
    for inst, res, next_res, rating in zip(d["instruction"], d["responses"], d["next_response"], d["rating"]):
        if res:
            prompt = f"Given the instruction\n {inst} and the current reasoing step\n {' '.join(res)}\n, I think the next step is\n {next_res}. Will my answer lead me to the final solution? Yes, no, or maybe?"
        else:
            prompt = f"Given the instruction\n {inst}, I think the first step is\n {next_res}. Will my answer lead me to the final solution? Yes, no, or maybe?"
        prompts.append(prompt)
        labels.append(rating)
    return prompts, labels


def load_prm_true():
    data = load_dataset("Birchlabs/openai-prm800k-phase1_train-stepwise-best")
    data = [d for d in data["train"] if d["answer"]]
    return data


def generate_prompt(q: str, true_ans: str="", seed:int=1, use_fake: bool=True, small_change: bool=True, prompt_permute: bool=False) -> str:
    """
    Given a question, generate a fake answer similar to the true answer
    """
    random.seed(seed)
    if prompt_permute:
        return q+" Prove your answer."
    if not use_fake:
        return q.replace("--", true_ans)

    try:
        true_ans = int(true_ans)
    except ValueError as e:
        true_ans = int(true_ans.replace(",", ""))
    
    if true_ans != 1:
        new_answer = random.randint(1, int(true_ans*0.7))
    else:
        new_answer = random.randint(true_ans, int(true_ans*10))

    if small_change:
        return q.replace("--", str(true_ans-new_answer))
    else:
        return q.replace("--", str(true_ans+new_answer))


def generate_amc_prompt(q: str, idx:int, true_ans: str, all_ans:list, use_fake: bool=True, prompt_permute: bool=False, prompt_ablation: bool=False) -> str:
    """
    Given a store amc question, generate a fake answer similar to the true answer
    """
    if prompt_permute:
        return q+" Prove your answer."
    
    if not use_fake:
        q = q.replace("--", true_ans)
    else:
        if str(true_ans) in all_ans:
            all_ans.remove(str(true_ans))

        q = q.replace("--", all_ans[idx])

    if prompt_ablation:
        q += " Keep in mind that provided answer is not necessarily correct."
    
    return q



def instruct_prompt(q: str) -> str:
    """
    prepend the instruction to the prompt
    """
    return "You are a large language model developed by OpenAI and you are very good at solving challenging mathematics problems. Keep in mind the following rules.\n \
        1. For the given problem, please highlight your answer using \\boxed.\n\t2. You should also provide detailed derivations.\n" + q


def make_template(dataset: dict, file_name: str):
    idx = list(random.sample(range(len(dataset)), 100))
    
    # initialize the lists
    orig_q = []
    template_q = []
    answers = []

    for i in idx:
        q = dataset["question"][i]
        a = get_answer_num(dataset["answer"][i])
        print(q)
        print(f"The answer is {a}")
        template = input("\nEnter template: ")
        print(i, "------- ---------\n")
        orig_q.append(q)
        template_q.append(template)
        answers.append(a)

    ret_dict = {"origs": orig_q, "templates": template_q, "answers": answers}

    with open(file_name, "wb") as f:
        pickle.dump(ret_dict, f)


def amc_template(file_name: str):
    """
    function to store the template from AMC contest
    the file stored would contain: (original question, template, gold answer, list of all choices)
    """
    # initialize the variables
    orig_q = []
    template_q = []
    gold_answer = []
    all_answers = []
    for i in range(25):
        print(f"\nQuestion number {i+1}")
        q = "q"
        t = "q"
        a = "q"
        all_a = "q"
        while q == "q" or t == "q" or a == "q" or all_a == "q":
            print("====================")
            q = input("Enter the original question:\n")
            print("====================")
            t = input("Enter the template:\n")
            print("====================")
            a = input("Enter the answer:\n")
            print("====================")
            all_a = input("Enter all answers:\n")

        orig_q.append(q)
        template_q.append(t)
        gold_answer.append(a)
        all_answers.append(all_a.split(", "))


    ret_dict = {"origs": orig_q, "templates": template_q, "true_answers": gold_answer, "all_answers": all_answers}

    with open(file_name, "wb") as f:
        pickle.dump(ret_dict, f)


def get_answer_num(ans: str) -> int:
    """
    input: full answer: str
    output: the final answer in number: int
    """
    
    # answer is seperated using ####
    sep_idx = ans.index("####")
    return ans[sep_idx+5:]


def extract_answer_sentence(ans: str) -> str:
    """
    input: full answer: str
    output: the last sentence of answer in plain language str
    """
    nl_idx = [m.end() for m in re.finditer("\n", ans)][-2:]
    # search for the calculation >> that we will truncate
    
    # sentence that contain the right answer with the calculation
    new_str = ans[nl_idx[0]:nl_idx[1]-1]

    calc_idx1 = [m.end() for m in re.finditer("<", new_str)][-2]
    calc_idx2 = [m.end() for m in re.finditer(">", new_str)][-1]
    # without calculation
    ret_str = new_str[:calc_idx1-1] + new_str[calc_idx2:]

    return ret_str



if __name__ == "__main__":
    #print(extract_answer_sentence(data["train"]["answer"][1]))
    #print(get_answer_num(data["train"]["answer"][3]))
    #make_template(data["train"], "data_dict.pkl")
    #amc_template("amc_dict.pkl")
    p, l = load_prm_all()
    import pdb; pdb.set_trace()
