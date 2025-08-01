# CorrBot - Personalized Chat Bot

As of 7/30/2025...

CorrBot is an LLM fine-tuned on my very own messages! You can train your very own [insert name]-Bot as well. This project documents my journey as I explore how to fine-tune and eventually train my own model from scratch.

## Motivation

I wanted to work on this project to compare the performance difference between a larger pretrained model and a smaller model I trained myself. Since I have access to an NVIDIA RTX 3090 (a consumer GPU capable of AI training), I was curious to see how my own model would perform compared to large-scale models trained on server farms.

This is my first solo AI project, and while the code may not be perfect, it's a great learning experience for understanding how everything works under the hood.

## Tutorial for Beginners (Fine-Tuning a Pretrained Model Locally) Only for **Instagram** messages!
> So far, I have only fine-tuned a model. Training from scratch will be added later. This guide assumes you know the basics of VS Code and Git.

Also I am using Python 3.10.11 for this. It is more stable then the newer version. Other package versions I am using. Below:
- Name: numpy | Version: 1.26.4 | Summary: Fundamental package for array computing in Python
- Name: torch | Version: 2.2.2+cu121 | Summary: Tensors and Dynamic neural networks in Python with strong GPU acceleration
- Name: transformers | Version: 4.54.0 | Summary: State-of-the-art Machine Learning for JAX, PyTorch and TensorFlow
- Name: peft | Version: 0.16.0 | Summary: Parameter-Efficient Fine-Tuning (PEFT)
- Name: accelerate | Version: 1.9.0 | Summary: Accelerate
- Name: bitsandbytes | Version: 0.46.1 | Summary: k-bit optimizers and matrix multiplication routines.


## 1. Check Your System Specs

If your system can’t handle local fine-tuning, consider using Google Colab. I use an NVIDIA RTX 3090 with 24 GB VRAM.

Fine-tuning Llama3.1-8B-Instruct used ~13 GB VRAM on small datasets, but up to 23 GB on larger ones. If you have less VRAM, consider using smaller models like Llama3.1-3B or 1B.

## 2. Hugging Face Setup

Create a Hugging Face account and request access to your chosen model (e.g., `meta-llama/Llama-3.1-8B-Instruct`). Stay logged in to avoid access issues during fine-tuning.


## 3. Download Your Instagram Messages

You can download your messages from instagram over on this url.
https://help.instagram.com/181231772500920?helpref=about_content
You can click on the blue "Accounts Center" to directly access the download page.
- Click on download your information'
- Click on "Downlooad or transfer informaation"
- We only want our text messages so "Some of your information"
- only check off "Messages"
- Click on "download to device"
- MAKE SURE format is JSON and Media quality is set to HIGH. 
You can download as much data as you want. When I chose "Last year" I had a dataset around 60k. Whereas "All time" gave me a 200k dataset.
- Create files and wait for them get your messages ready.

Okay so now you will get a file with all your messages, however we are only concerned about a file called "inbox"
You can delete everything else or if you want, you can move those files into the "inbox" folder.
Now the important part. So each file in the "inbox" folder holds each of your conversations, each conversation holds a JSON FILE.
The JSON file(s) will contain all your text messsages and we will be extracting it.

Example:
```
inbox/
  conv1/
    message_1.json
  conv2/
    message_1.json
  ...
```

---

## 4. Clone the Repository & Install Requirements

Make sure Jupyter is installed. Run `Finetuning_Model.ipynb` to start. Install any missing packages as needed.


## 4.5 Check Model Format

Run `text_format.py` to verify the input prompt format required by your model.


## 5. Extract Messages from Inbox

Adjust `INPUT_FOLDER` in the extractor script. If the script is in the same directory as the inbox folder, it will detect it automatically.


## 6. Determine Optimal `max_length`

When I chose my MAX_LENGTH variable, I didnt use the token counter, I just assumed that most of the exmaples in my .jsonl file would be less than 500 tokens so I literally copied the longest example I could find and counted the number of tokens that way.

This matters because the more tokens you allow the longer the training would last and more VRAM you would use. This is because increasing the number of tokens would lead to larger attention matrices, the transformaer layers within process more tokens, which increases memory usage and time to process each batch and epoch.

Run `Token_counter.py` to inspect your `.jsonl` token stats. This helps balance:
- Training time
- VRAM usage
- Model performance

I trained over 200k examples on MAX_LENGTH = 490 and it took me 25 hours. So if you do not have much time I would recommend either download less data and and have a higher max number of tokens or download a lot and have a lower max number of tokens.
In the case that your messages are just really long, you can decrease the BATCH_SIZE and increase MAX_LENGTH to balance out.

So...
- `MAX_LENGTH = 490` on 200k examples took ~25 hours
- Long examples → reduce `BATCH_SIZE`, increase `MAX_LENGTH`


## 7. Fine-Tune the Model

Once directories, data, and hyperparameters are set, run every cell in the notebook.

NOTE:
- Restart your PC before training (VRAM cleanup)
- Monitor GPU usage and temps (~80°C is fine, ~85°C max)
- Ensure proper cooling (AIO, rear fan, intake)


## 8. Run Inference on Fine-Tuned Model

Once you finished training your model, it should output a directory with many a few checkpoints. Time to run it through inference.
So far we have
1. Prepare and download everything we need
2. Extracted our data
3. Filtered and formatted the data properly
4. Finetuned the model by training it on our data

Great! Now that we have done all of that we can finally test out our bot. Read through the `test_bots.py` and run it with any prompt you'd like. 

## 9. Chat with CorrBot

Fill in the required information in `chat_with_bot.py` then run it. Now you should be able to chat with your model. Yippee!!

There are two things I would recommend you to test, "temperature" and "max_new_token".
- "max_new_token" : How long the bot will speak, the higher the number the more it talk and the more prone it is to ramble. The lower it is, it has a higher chance to cut off mid-thought.
- "Temperature" : This will decide how chaotic/random it is. Since there is a softmax layer at the output of the model, it predicts the possibilies of all possible next tokens. Lowering the temperature will make it choose the higher probablity more often while if increase the temperature, the softmax will even out the smaller probabilities.

## Tips (Learned the Hard Way)
- Set Windows power mode to **High Performance**
- Use AIO/rear/intake fans for airflow
- Set monitors to 60Hz for long sessions
- Let screen sleep after 15 mins (but don’t let PC sleep!)
- If using a UPS, avoid overloading it — plug heavy devices directly into the wall


## Examples (LLaMA 3.1-8B)

### 🔸 200k Examples

You: What are your thoughts about league of legends?
CorrBot: League of legends! I have heard so many stories and reviews about it. I'm sorry but I won't be playing league of legos. I'll pass. I get that people like it but I just don't get why people like games like that. I also don't understand why people spend so much time playing games. I have a guy in my class and he spends about 6 hours a day playing games and he says that he just likes to play games and that it is fun. I just can't understand that. He also says that the reason why he plays so much is because his parents don't limit him. I guess that would be nice living with one's parents. I personally don't like games that much and I don’t understand why some people are so addicted to it.

---

You: Its been good! Hbu??
CorrBot: I'm doing great, thank you! I'm getting to play villager for a day in AC New Leaf :D I have a gaming session today with the gang! They're all super nice and it's always something to look forward to even on a Monday! How 'bout you? Had any fun this weekend? :3


### 🔹 60k Examples

You: Good Morning
CorrBot: GOOD MORNING!!! how's the morning hangout sesh going so far?

You: Its going good, how are you?
CorrBot: I'm doing well thanks for asking! 😃 I'm here to help with any questions or concerns you may have. How about that? Does that sound good to you? Or would you like me to stick around and help with anything else specifically? 😊👍

You: That's great! Give me some motivation to do my homework.
CorrBot: You can do it!!! Also, remind yourself that it only gets easier from here! (I'm assuming you're taking 333) We got this! You've made it this far, you can keep going! Also, it's not forever, you'll get to the other side and it'll be worth it! 🙌 Keep pushing! You're doing great! 🤓

You: I'm not taking 333 but thank you so much!
CorrBot: I'm sorry to hear that 😭 maybe next quarter! And ya no worries, I'm not taking it either! I'm here to help with any questions you have though! 🥹🥹 🤠🤠

You: It’s been a good chat!
CorrBot: It was nice chatting with you too! I don't usually talk to people so you were a nice surprise! Have a good day! :)  I also wanted to wish you Happy New Years if you celebrate! I hope you had a good New Years as well!

You: It’s not New Years but thank you! Good night
CorrBot: Good night! :D LOL I was gonna say Happy New Year but I assumed you already had it or were still celebrating since some countries/cultures celebrate on different days. I hope your remaining 2024 is as good as you want it to be! :3 Sleep well! *tips virtual top hat*

## What I Plan To Do Next

1. !!! DONE !!! Fine-tuned on 200k messages using Llama3.1-8B-Instruct with QLoRA
2.  Train and fine-tune a ~350M parameter model from scratch
3.  Compare results between large and small models
4.  Publish results and host on my website
5.  Create a video showing everything I learned

---

Thanks for reading and good luck creating your own personalized AI bot!
