from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re

# === CONFIG ===
MODEL_PATH = r"" # Change to path of checkpoint. There is an exameple in test_bots.py
MAX_HISTORY_TURNS = 3 # How many preceeding chats it will remember.

# Filtering out replies for repeated emojis or characters
def clean_reply(text):
    text = re.sub(r'(.)\1{3,}', r'\1\1', text)
    text = re.sub(r'\b(\w+)( \1){2,}', r'\1 \1', text)
    text = re.sub(r'([\U00010000-\U0010ffff])\1{3,}', r'\1\1\1', text)
    return text.strip()

# NOTICE, this prompt is built off the Llama3.1 chat format, if you are using a different model
# You would have to change the prompt building process
def build_prompt(history):
    history = history[-MAX_HISTORY_TURNS:]
    prompt = ""
    for role, message in history:
        prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n{message}<|eot_id|>\n"
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return prompt

# === LOAD MODEL + TOKENIZER ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype="auto"
)
model.eval()

# === SPECIAL STOP TOKEN ===
# Defines when the model stops generating tokens
eot_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

print("Chat with CorrBot (type 'exit' to quit)")
chat_history = []

# Chat loop
while True:
    user_input = input("You: ")
    if user_input.strip().lower() in ["exit", "quit"]:
        print("Exiting chat.")
        break

    chat_history.append(("user", user_input.strip()))
    prompt = build_prompt(chat_history)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=160, # You can change this if you want your model to talk less or more
        do_sample=True,
        temperature=0.65, # Increasing this would make your model more chaotic and random.
        top_p=0.9, # probability threshold | no need to change unlesss you want to experiment
        top_k=50, # cap for number of tokens to sample one | no need to change unlesss you want to experiment
        no_repeat_ngram_size=3,
        eos_token_id=eot_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)

    if "<|start_header_id|>assistant<|end_header_id|>\n\n" in decoded:
        reply = decoded.split("<|start_header_id|>assistant<|end_header_id|>\n\n")[-1]
        reply = reply.split("<|eot_id|>")[0]
    else:
        reply = decoded

    reply = clean_reply(reply.strip())
    chat_history.append(("assistant", reply))
    print(f"CorrBot: {reply}\n") # You can change this name to whatever you want.
