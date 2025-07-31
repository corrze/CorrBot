import json
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# This file gives you the statistics of your .jsonl file.
# You can use this to determine your MAX_LENGTH variable.

# === CONFIG ===
JSONL_PATH = "prompt_response.jsonl"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

# === LOAD TOKENIZER ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def analyze_lengths(jsonl_path):
    lengths = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Tokenizing prompts"):
            data = json.loads(line)
            prompt = data["prompt"]
            tokens = tokenizer(prompt, truncation=False)["input_ids"]
            lengths.append(len(tokens))

    # Convert to NumPy for easier math
    lengths_np = np.array(lengths)

    print("\n Token Length Stats:")
    print(f"Total examples: {len(lengths)}")
    print(f"Longest number of tokens: {np.max(lengths_np)}")
    print(f"Mean number of tokens: {np.mean(lengths_np):.2f}")
    print(f"Median number of tokens: {np.median(lengths_np)}")
    print(f"95th percentile: {np.percentile(lengths_np, 95):.0f}")
    print(f"99th percentile: {np.percentile(lengths_np, 99):.0f}")
    print(f"99.9th percentile: {np.percentile(lengths_np, 99.9):.0f}")

if __name__ == "__main__":
    analyze_lengths(JSONL_PATH)
