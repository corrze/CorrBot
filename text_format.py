from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct") # Change model if needed
print(tokenizer.chat_template)

# Use this script to check the chat template of any model
# It should print the chat template used by the tokenizer, which is useful for formatting prompts correctly

# It will print out a very long and confusing log but for example this is how Llama3.1-Instruct does it.

#{%- for message in messages %}
#    ...
#    {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' }}
#    ...
#{%- endfor %}

# Which can be translated to

# -----------------------------------------------
#<|start_header_id|>user<|end_header_id|>
#
#[Prompt text from context]
#
#<|eot_id|>
#<|start_header_id|>assistant<|end_header_id|>
#
#[Your response]
#
#<|eot_id|>
#-------------------------------------------------