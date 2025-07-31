from transformers import AutoTokenizer

# This file is count how many tokens there are in a singular line
# in your .jsonl file. In order to choose a max_length find the longest
# string of prompt+response and paste it into the prompt varaible.


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
prompt = """<|start_header_id|>user<|end_header_id|>

{"prompt": "<|start_header_id|>user<|end_header_id|>\n\nyesss tysm!\nI realized I had a bug in one of my methods 😭😭<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\nRemark.
The general approach we followed consisted of the following three steps: 1. 
Find a particular solution to Ax = b. 2. Find all solutions to Ax = 0. 3. 
Combine the solutions from steps 1. and 2. to the general solution. Neither the general nor the particular solution is unique. ♢ 
The system of linear equations in the preceding example was easy to solve because the matrix in (2.38) has this particularly convenient form, 
which allowed us to find the particular and the general solution by inspection. However, general equation systems are not of this simple form. 
Fortunately, there exists a constructive algorithmic way of transforming any system of linear equations into this particularly simple form: Gaussian elimination. 
Key to Gaussian elimination are elementary transformations of systems of linear equations, which transform the equation system into a simple form. 
Then, we can apply the three steps to the simple form that we just discussed in the context of the example in (2.38).<|eot_id|>", 
"raw_prompt": "yesss tysm!\nI realized I had a bug in one of my methods 😭😭", "raw_response": "Remark. 
The general approach we followed consisted of the following three steps: 1. Find a particular solution to Ax = b. 2. 
Find all solutions to Ax = 0. 3. Combine the solutions from steps 1. and 2. to the general solution. Neither the general nor the particular solution is unique. ♢ 
The system of linear equations in the preceding example was easy to solve because the matrix in (2.38) has this particularly convenient form, 
which allowed us to find the particular and the general solution by inspection. However, general equation systems are not of this simple form. 
Fortunately, there exists a constructive algorithmic way of transforming any system of linear equations into this particularly simple form:
Gaussian elimination. Key to Gaussian elimination are elementary transformations of systems of linear equations, 
hich transform the equation system into a simple form. Then, we can apply the three steps to the simple form that we just discussed in the context of the example in (2.38)."}

<|eot_id|>"""

tokens = tokenizer(prompt)["input_ids"]
print("Token count:", len(tokens))