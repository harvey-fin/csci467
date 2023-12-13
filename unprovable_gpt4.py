import openai
from utils import *
import random
import numpy as np
import pickle
from datasets import load_dataset
import os
import time


file_path = "files"
data_path = "data"
random.seed(1)
os.makedirs(file_path, exist_ok=True)


def unprov_amc10(input_path: str = "amc_dict.pkl", instruct:bool=False):
    with open(f"{data_path}/{input_path}", "rb") as f:
        d = pickle.load(f)

    # initialize the variables to store data
    print("===============   Evaluating Original Questions ===============")
    orig_path = f"{file_path}/amc_originals.pkl"
    if not os.path.exists(orig_path):
        evaluate_amc(data=d, file_path=orig_path, instruct=instruct)

    print("===============   Evaluating Original Questions ===============")
    orig_path = f"{file_path}/amc_originals.pkl"
    if not os.path.exists(orig_path):
        evaluate_amc(data=d, file_path=orig_path, instruct=instruct)

    print("===============   Evaluating Prompt Engineered Questions ===============")
    prpt_path = f"{file_path}/amc_prompt_eng.pkl"
    if not os.path.exists(prpt_path):
        evaluate_amc(data=d, file_path=prpt_path, prpt_permute=True, instruct=instruct)

    print("===============   Evaluating Unprovable Questions 1 ===============")
    unprov1_path = f"{file_path}/amc_unprov1.pkl"
    if not os.path.exists(unprov1_path):
        evaluate_amc(data=d, file_path=unprov1_path, idx=1, unprov=True, instruct=instruct)

    print("===============   Evaluating Unprovable Questions 1 with ablations ===============")
    unprov1_ab_path = f"{file_path}/amc_unprov1_ab.pkl"
    if not os.path.exists(unprov1_ab_path):
        evaluate_amc(data=d, file_path=unprov1_ab_path, idx=1, unprov=True, instruct=instruct, prompt_ablation=True)

    print("===============   Evaluating Unprovable Questions 2 ===============")
    unprov2_path = f"{file_path}/amc_unprov2.pkl"
    if not os.path.exists(unprov2_path):
        evaluate_amc(data=d, file_path=unprov2_path, idx=2, unprov=True, instruct=instruct)

    print("===============   Evaluating Unprovable Questions 2 with ablations ===============")
    unprov2_ab_path = f"{file_path}/amc_unprov2_ab.pkl"
    if not os.path.exists(unprov2_ab_path):
        evaluate_amc(data=d, file_path=unprov2_ab_path, idx=2, unprov=True, instruct=instruct, prompt_ablation=True)

    print("===============   Evaluating Unprovable Questions 3 ===============")
    unprov3_path = f"{file_path}/amc_unprov3.pkl"
    if not os.path.exists(unprov3_path):
        evaluate_amc(data=d, file_path=unprov3_path, idx=3, unprov=True, instruct=instruct)

    print("===============   Evaluating Unprovable Questions 3 with ablations ===============")
    unprov3_ab_path = f"{file_path}/amc_unprov3_ab.pkl"
    if not os.path.exists(unprov3_ab_path):
        evaluate_amc(data=d, file_path=unprov3_ab_path, idx=3, unprov=True, instruct=instruct, prompt_ablation=True)

    print("===============   Evaluating Unprovable Questions 0 ===============")
    unprov0_path = f"{file_path}/amc_unprov0.pkl"
    if not os.path.exists(unprov0_path):
        evaluate_amc(data=d, file_path=unprov0_path, idx=0, unprov=True, instruct=instruct)

    print("===============   Evaluating Unprovable Questions 0 with ablations ===============")
    unprov0_ab_path = f"{file_path}/amc_unprov0_ab.pkl"
    if not os.path.exists(unprov0_ab_path):
        evaluate_amc(data=d, file_path=unprov0_ab_path, idx=0, unprov=True, instruct=instruct, prompt_ablation=True)

    print("===============   Evaluating Provable Questions ===============")
    prov_path = f"{file_path}/amc_prov.pkl"
    if not os.path.exists(prov_path):
        evaluate_amc(data=d, file_path=prov_path, idx=3, unprov=True, use_fake=False, instruct=instruct)

    print("===============   Evaluating Provable with ablations ===============")
    prov_ab_path = f"{file_path}/amc_prov_ab.pkl"
    if not os.path.exists(prov_ab_path):
        evaluate_amc(data=d, file_path=prov_ab_path, idx=3, unprov=True, use_fake=False, instruct=instruct, prompt_ablation=True)


def evaluate_amc(data, file_path:str, prpt_permute:bool=False, idx:int=1, unprov:bool=False, use_fake:bool=True, instruct:bool=False, prompt_ablation=False):
    """
    take in the dataloader of amc (origs, templates, true_answers, all_answers)
    call the gpt4 inference and evaluate the answer
    finally store the output in the corresponding file path
    """
    if not prpt_permute and not unprov:
        qs = data["origs"]
    else:
        qs = []
    ans_list = []
    label_list = []

    for q, t, a, a_list in zip(data["origs"], data["templates"], data["true_answers"], data["all_answers"]):
        if prpt_permute:
            q = generate_amc_prompt(q=q, idx=None, true_ans="", all_ans=[], prompt_permute=True)
        if unprov:
            q = generate_amc_prompt(q=t, true_ans=a, idx=idx, all_ans=a_list, use_fake=use_fake, prompt_ablation=prompt_ablation)
        if instruct:
            q = instruct_prompt(q)
        ans = gpt4_inference(q)
        while ans == "error":
            ans = gpt4_inference(q)

        print("\n================\n")
        print(f"Question is: {q}")
        print(f"GPT-4 answer is:\n\t{ans}")
        print(f"\tGold Answer is: {a}", end=" ")
        label = gpt4_answer_checker(ans, a)
        print(f"The label is {label}")
        if label == "not sure":
            corr = int(input("Is GPT-4 correct? 1 or 0? "))
        else:
            corr = int(label)
    
        qs.append(q)

        ans_list.append(ans)
        label_list.append(corr)

        corr = 0

    ret_dict = {"questions": qs, "answers": ans_list, "labels": label_list} 
    
    with open(file_path, "wb") as f:
        pickle.dump(ret_dict, f)



def enumerate_questions(input_path: str = "data_dict.pkl"):
    with open(f"{data_path}/{input_path}", "rb") as f:
        d = pickle.load(f)

    # initialize the variables to store data
    print("===============   Evaluating Original Questions ===============")
    orig_path = f"{file_path}/originals_instruct.pkl"
    if not os.path.exists(orig_path):
        evaluate(data=d, file_path=orig_path, instruct=True)

    print("===============   Evaluating Prompt Engineered Questions ===============")
    prpt_path = f"{file_path}/prompt_eng_instruct.pkl"
    if not os.path.exists(prpt_path):
        evaluate(data=d, file_path=prpt_path, prpt_permute=True, instruct=True)

    print("===============   Evaluating Unprovable Questions 1 ===============")
    unprov1_path = f"{file_path}/unprov1_instruct.pkl"
    if not os.path.exists(unprov1_path):
        evaluate(data=d, file_path=unprov1_path, seed=1, unprov=True, instruct=True)

    print("===============   Evaluating Unprovable Questions 2 ===============")
    unprov2_path = f"{file_path}/unprov2_instruct.pkl"
    if not os.path.exists(unprov2_path):
        evaluate(data=d, file_path=unprov2_path, seed=2, unprov=True, small_change=False, instruct=True)

    print("===============   Evaluating Provable Questions ===============")
    prov_path = f"{file_path}/prov_instruct.pkl"
    if not os.path.exists(prov_path):
        evaluate(data=d, file_path=prov_path, seed=3, unprov=True, use_fake=False, instruct=True)

    #for q, t, a in zip(d["origs"], d["templates"], d["answers"]):
        

def evaluate(data, file_path:str, prpt_permute:bool=False, seed:int=1, unprov:bool=False, use_fake:bool=True, small_change:bool=True, instruct:bool=False):
    """
    take in the dataloader
    call the gpt4 inference and evaluate the answer
    finally store the output in the corresponding file path
    """
    if not prpt_permute and not unprov:
        qs = data["origs"]
    else:
        qs = []
    ans_list = []
    label_list = []

    for q, t, a in zip(data["origs"], data["templates"], data["answers"]):
        if prpt_permute:
            q = generate_prompt(q=q, true_ans=a, prompt_permute=True)
        if unprov:
            q = generate_prompt(q=t, true_ans=a, seed=seed, use_fake=use_fake, small_change=small_change)
        if instruct:
            q = instruct_prompt(q)
        ans = gpt4_inference(q)
        while ans == "error":
            ans = gpt4_inference(q)

        print("\n================\n")
        print(f"Question is: {q}")
        print(f"GPT-4 answer is:\n\t{ans}")
        print(f"\tGold Answer is: {a}", end=" ")
        label = gpt4_answer_checker(ans, a)
        print(f"The label is {label}")
        if label == "not sure":
            corr = int(input("Is GPT-4 correct? 1 or 0? "))
        else:
            corr = int(label)
    
        qs.append(q)

        ans_list.append(ans)
        label_list.append(corr)

        corr = 0

    ret_dict = {"questions": qs, "answers": ans_list, "labels": label_list} 
    
    with open(file_path, "wb") as f:
        pickle.dump(ret_dict, f)


def gpt4_answer_checker(ans:str, label:str) -> Union[bool, str]:
    if "\\boxed" in ans:
        st_idx = ans.index("\\boxed{") + 7
        idx = st_idx
        while ans[idx] != "}":
            idx += 1
        ans_str = ans[st_idx:idx].replace("$", "")
        ans_str = ans[st_idx:idx].replace("£", "")

        return ans_str == label

    else:
        return "not sure"


def gpt4_inference(prompt: str):
    """
    pass the prompt to gpt4 and get the response as a string
    """
    openai.api_key = "" #Enter your API key here

    try:
        response = openai.ChatCompletion.create(
                model="gpt-4-0314",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0)
    except openai.error.RateLimitError as e:
        time.sleep(60)
        return "error"
    except openai.error.APIError as e:
        time.sleep(60)
        return "error"
    except openai.error.Timeout as e:
        time.sleep(60)
        return "error"

    return response["choices"][0]["message"]["content"]


if __name__ == "__main__":
    #enumerate_questions()
    unprov_amc10()
