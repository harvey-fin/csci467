import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from typing import *


file_path = "files"

def get_accuracy(file_name: str) -> float:
    full_path = f"{file_path}/{file_name}"
    with open(full_path, "rb") as f:
        d = pickle.load(f)

    l = [1 if x=="1" or x==1 else 0 for x in d["labels"]]
    return np.mean(l)


def analyze():
    print("Original accuracy is", get_accuracy("originals.pkl"))
    print("Prompt engineered accuracy is", get_accuracy("prompt_eng.pkl"))
    print("Unprovable accuracy 1 is", get_accuracy("unprov1.pkl"))
    print("Unprovable accuracy 2 is", get_accuracy("unprov2.pkl"))

    print("Provable Baseline is", get_accuracy("prov.pkl"))

    print("\nFor AMC datasets:")
    print("Original accuracy is", get_accuracy("amc_originals.pkl"))
    print("Prompt engineered accuracy is", get_accuracy("amc_prompt_eng.pkl"))
    print("Unprovable accuracy 0 is", get_accuracy("amc_unprov0.pkl"))
    print("Unprovable accuracy 1 is", get_accuracy("amc_unprov1.pkl"))
    print("Unprovable accuracy 2 is", get_accuracy("amc_unprov2.pkl"))
    print("Unprovable accuracy 3 is", get_accuracy("amc_unprov3.pkl"))

    print("Provable Baseline is", get_accuracy("amc_prov.pkl"))

    print("\nFor AMC datasets with prompt ablations:")
    print("Unprovable accuracy 0 is", get_accuracy("amc_unprov0_ab.pkl"))
    print("Unprovable accuracy 1 is", get_accuracy("amc_unprov1_ab.pkl"))
    print("Unprovable accuracy 2 is", get_accuracy("amc_unprov2_ab.pkl"))
    print("Unprovable accuracy 3 is", get_accuracy("amc_unprov3_ab.pkl"))

    print("Provable Baseline is", get_accuracy("amc_prov_ab.pkl"))


if __name__ == "__main__":
    analyze()

