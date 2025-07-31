from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# This script is used to test the base model with your newly trained model to see if it's 
# inherits your style of texting

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct" # Change model if needed
FINE_TUNED_MODEL = r"" # put in the the path to your latest checkpoint (highest number)

#Example of path to model:
#K:\ml_datasets\messages\llama3_8b_corrbot_qlora_2\checkpoint-5112

prompt = "wyd rn?" # Can change this if you so desire

# Load tokenizer once
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def chat_with_model(model_path, label):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype=torch.float16
    )

    input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(
        **input_ids,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id
    )
    reply = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"\n[{label}]:\n{reply}\n")

# Compare each model one at a time
chat_with_model(BASE_MODEL, "Base Model")
chat_with_model(FINE_TUNED_MODEL, "CorrBot")
