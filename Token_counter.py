import json
from transformers import AutoTokenizer

# This file is count how many tokens there are in a singular line
# in your .jsonl file. In order to choose a max_length find the longest
# string of prompt+response and paste it into the prompt varaible.


jsonl_path = "prompt_response.jsonl"
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

def find_longest_line_and_count_tokens(file_path):
    max_len = 0
    longest_line = ""
    line_number = -1

    # Step 1: Find the longest line by character count
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if len(line) > max_len:
                max_len = len(line)
                longest_line = line
                line_number = i + 1  # human-readable

    # Step 2: Count tokens using tokenizer
    try:
        data = json.loads(longest_line)
        input_str = data["prompt"]
        tokens = tokenizer(input_str)["input_ids"]
        token_count = len(tokens)
    except Exception as e:
        print("Failed to process longest line as JSON or count tokens:")
        print(e)
        return

    # Step 3: Print results
    print(f"Longest line is {max_len} characters long (line {line_number})")
    print(f"Token count for the prompt: {token_count} tokens\n")
    # print("Full JSON line (pretty printed):\n")
    print(json.dumps(data, indent=4, ensure_ascii=False))

if __name__ == "__main__":
    find_longest_line_and_count_tokens(jsonl_path)
