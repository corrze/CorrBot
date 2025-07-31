# CorrBot - Personalized Chat Bot

As of 7/30/2025...

CorrBot is going to be an LLM trained on my very own messages! You can train your very own [insert name]-Bot as well! Here is an easy guide to creating your own persoanlized chatbot. This project is me documenting my journey as I explore how to fine tune as well as train my own model.

# Motivation

I wanted work on this project to compare how big the difference is, between a larger model and a smaller model. This is because I got my hands on an Nvidia 3090, a consumer gpu capabable of training AI models,and I wanted to see how well of a model it can train, compared to a company trained model. For context, they train using many AI gpus for days to months. I think my smaller made from scratch model will not be as accurate as the pretrained model but I want to see how bad it is in comparison.


# Turtorial (Fine-Tuning a pretrained model) !!!This guide is only for *instagram* messages.!!!

NOTE: So far, I have only fine-tuned a model on a dataset, I will be trainig my own model and writing about that at a later date.

Also I am using Python 3.10.11 for this. It is more stable then the newer version. Other package versions I am using. Below:
Name: numpy | Version: 1.26.4 | Summary: Fundamental package for array computing in Python
Name: torch | Version: 2.2.2+cu121 | Summary: Tensors and Dynamic neural networks in Python with strong GPU acceleration
Name: transformers | Version: 4.54.0 | Summary: State-of-the-art Machine Learning for JAX, PyTorch and TensorFlow
Name: peft | Version: 0.16.0 | Summary: Parameter-Efficient Fine-Tuning (PEFT)
Name: accelerate | Version: 1.9.0 | Summary: Accelerate
Name: bitsandbytes | Version: 0.46.1 | Summary: k-bit optimizers and matrix multiplication routines.


# 1. Make sure your computer can run the fine tuning.

If not then I recommend using Google Collab in order to get computing power. I am using an NVIDIA RTX 3090 for this as it has enough VRAM to finetune a larger model. I recommend having at least 16 Gbs of VRAM.
I am using an 8 Billion parameter model and it uses 13 Gbs of VRAM usually. However when I trained on a larger dataset it used up to 23 Gbs so be aware if you are training it locally.

# 2. Creating a hugging face account and get permission to use your model of choice.
For this project I used Llama3.1-8B-Instruct by Meta. I haven't used other models just yet but make sure your model is capable of generating text.

# 3. Clone the repository and download the needed packages
pip install torch transformers datasets peft accelerate bitsandbytes xformers huggingface-cli


# What I am planning on so far

1. !!!!! DONEEEE !!!!!!   Fine-Tune on 200k lines of prompt+respone data using Llama3.1-8B-Instruct with LoRA
2. Training and creating a model from scratch ~350M parameters and fine-tune on that
4. Compare the models together to see the differences.
5. Put on my website and create a video to show findings.
