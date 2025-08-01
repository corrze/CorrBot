import os
import json
import re
from glob import glob
from html import unescape

# THIS FILE IS USED TO EXTRACT PROMPT-RESPONSE PAIRS FROM MESSAGES
# It assumes that the messages are stored in JSON files in a specific folder structure.
# It also assumes that you are using Llama-3.1-8B-Instruct
# If you are using a different model change the "prompt" somewhere around line 121
# to the same structure as your model. You can check what your structure is by running "text_format.py"
# With the your model

YOUR_NAME = "" # Enter the name you have instagram, check any message_1.json file and locat your sender name.
INPUT_FOLDER = "inbox" # You can set the whole path if it's not in the same directory
OUTPUT_FILE = "prompt_response.jsonl" # You can change to this whatever you want.

# Added 7/29/25
def reduce_repetition(text):
    text = re.sub(r'(.)\1{3,}', r'\1\1', text) # Reduce character repetition
    text = re.sub(r'\b(\w+)( \1){2,}', r'\1 \1', text) # Reduce word repetition
    text = re.sub(r'([\U00010000-\U0010ffff])\1{3,}', r'\1\1\1', text) # Reduce emoji repetition
    return text.strip()

# Cleaning the raw text messages To prepare for formatting
def clean(text):
    try:
        text = text.encode('latin1').decode('utf-8')
    except UnicodeEncodeError:
        pass
    except UnicodeDecodeError:
        pass
    text = unescape(text.strip().replace('\u200e', '').replace('\n', ' '))
    text = re.sub(r"\s*\(edited\)$", "", text, flags=re.IGNORECASE) # Remove "(edited)" at the en
    text = reduce_repetition(text) # Reduce repetition
    return text.strip()

# Check if the conversation is a group chat or one-on-one
def is_one_on_one(participants):
    others = [p for p in participants if p['name'] != YOUR_NAME]
    return len(others) == 1

# filters out URL or names with regex
def contains_link(text: str) -> bool:
    url_pattern = re.compile(
        r"(https?://|www\.|[a-zA-Z0-9\-]+\.(com|net|org|edu|gov|co|io|gg|me|ly))"
    )
    return bool(url_pattern.search(text))

# remove phrases generated by instagram
def is_valid_prompt(prompt: str) -> bool:
        system_phrases = [ # You can add more phrases to remove from your text messages
            "Reacted",
            "Liked a message",
            "sent an attachment",
            "sent a photo",
            "sent a video",
            "sent a file",
            "sent a sticker",
            "sent a voice message",
            "sent a GIF",
            "sent a location",
            "sent a contact",
            "sent a poll",
            "sent a link",
            "This poll is no longer available."
        ]
        return not any(prompt.startswith(p) for p in system_phrases)

# This looks at all your message_* files and then sorts it by chronogloical order.
# Check the messagse are send by you and looks backwards in the message history
# to gather more context. Then it constructs a prompt using the preceeding
# N messages and attaches your response
# I am using a context turn of 3 so it checks the last 3 messages.
def extract_prompt_response_pair(convo_path, context_turns = 3, min_response_len = 4):
    json_files = sorted(glob(os.path.join(convo_path, "message_*.json")))
    messages = []
    participants = []

    # Loading and opening all JSON files in convo folder
    for file in json_files:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            participants = data.get("participants", participants)
            messages.extend(data.get("messages", []))
    
    messages.sort(key=lambda x: x["timestamp_ms"])
    one_on_one = is_one_on_one(participants)
    pairs = []
    
    for i in range(1, len(messages)):
        curr = messages[i]
        if curr.get("sender_name") != YOUR_NAME or not curr.get("content"):
            continue
        
        response = clean(curr["content"])
        if (len(response) < min_response_len or contains_link(response) or not is_valid_prompt(response)):
            continue

        # Collect N context turns before the response
        prompt_lines = []
        turns_collected = 0
        j = i - 1

        while j >= 0 and turns_collected < context_turns:
            msg = messages[j]
            if not msg.get("content"):
                j -= 1
                continue
            
            content = clean(msg["content"])
            sender = msg.get("sender_name")

            # Skip earlier messages from the same sender
            if sender == YOUR_NAME or contains_link(content) or not is_valid_prompt(content) or len(content) < min_response_len:
                j -= 1
                continue

            if one_on_one or YOUR_NAME.lower() in content.lower():
                prompt_lines.insert(0, content)
                turns_collected += 1

            j -= 1

        if prompt_lines:
            prompt = "\n".join(prompt_lines)
            response = response.strip()
            pairs.append({
                "prompt": ( # HERE change this to whatever format your model uses. LLaMA3.1 uses this instruction-tuned prompt style.
                    "<|start_header_id|>user<|end_header_id|>\n\n"
                    + prompt.strip()
                    + "<|eot_id|>\n"
                    + "<|start_header_id|>assistant<|end_header_id|>\n\n"
                    + response.strip()
                    + "<|eot_id|>"
                ),
                "raw_prompt": prompt,
                "raw_response": response
            })
    return pairs
    
# Loops through all the folders and extracts each convo and puts it into a jsonl file.
def main():
    all_pairs = []
    
    for folder in os.listdir(INPUT_FOLDER):
        convo_path = os.path.join(INPUT_FOLDER, folder)
        if os.path.isdir(convo_path):
            pairs = extract_prompt_response_pair(convo_path, context_turns=3)
            all_pairs.extend(pairs)

    # Check if extracted
    print(f"Extracted {len(all_pairs)} prompt-response pairs.")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            json.dump(pair, f, ensure_ascii=False)
            f.write("\n") # New Line for each json object

if __name__ == "__main__":
    main()