# train.py - QLoRA training pipeline (skeleton)

import json
from datasets import Dataset

def load_dataset(path):
    with open(path, "r") as f:
        data = json.load(f)
    return Dataset.from_list(data)


def main():
    print("Loading dataset...")

    dataset = load_dataset("data/dataset.json")

    print("Dataset loaded:")
    print(dataset)

    print("Next step: model loading + QLoRA setup")


if __name__ == "__main__":
    main()