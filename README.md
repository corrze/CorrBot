# CorrBot - Personalized Chat Bot

As of 7/30/2025...

CorrBot is an LLM fine-tuned on my very own messages! You can train your very own [insert name]-Bot as well. This project documents my journey as I explore how to fine-tune and eventually train my own model from scratch.

## Motivation

I wanted to work on this project to compare the performance difference between a larger pretrained model and a smaller model I trained myself. Since I have access to an NVIDIA RTX 3090 (a consumer GPU capable of AI training), I was curious to see how my own model would perform compared to large-scale models trained on server farms.

This is my first solo AI project, and while the code may not be perfect, it's a great learning experience for understanding how everything works under the hood.


## Tutorial for Beginners (Fine-Tuning a Pretrained Model Locally)
> Only for **Instagram** messages!

> ⚠️ So far, I have only fine-tuned a model. Training from scratch will be added later. This guide assumes you know the basics of VS Code and Git.

### 🛠 Python & Package Versions
- Python: 3.10.11 (More stable than newer versions)
- numpy: 1.26.4
- torch: 2.2.2+cu121
- transformers: 4.54.0
- peft: 0.16.0
- accelerate: 1.9.0
- bitsandbytes: 0.46.1


## 1. Check Your System Specs

If your system can’t handle local fine-tuning, consider using Google Colab. I use an NVIDIA RTX 3090 with 24 GB VRAM.

Fine-tuning Llama3.1-8B-Instruct used ~13 GB VRAM on small datasets, but up to 23 GB on larger ones. If you have less VRAM, consider using smaller models like Llama3.1-3B or 1B.

## 2. Hugging Face Setup

Create a Hugging Face account and request access to your chosen model (e.g., `meta-llama/Llama-3.1-8B-Instruct`). Stay logged in to avoid access issues during fine-tuning.


## 3. Download Your Instagram Messages

Go to: https://help.instagram.com/181231772500920
- Click "Accounts Center"
- Choose **Download Your Information** > "Some of Your Information"
- Select **Messages**, format as **JSON**, media quality as **High**
- Choose "All Time" for more data (I got ~200k messages)

This will download an `inbox` folder with subfolders for each conversation.

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

Run `Token_counter.py` to inspect your `.jsonl` token stats. This helps balance:
- Training time
- VRAM usage
- Model performance

For example:
- `MAX_LENGTH = 490` on 200k examples took ~25 hours
- Long examples → reduce `BATCH_SIZE`, increase `MAX_LENGTH`


## 7. Fine-Tune the Model

Once directories, data, and hyperparameters are set, run every cell in the notebook.

⚠️ NOTE:
- Restart your PC before training (VRAM cleanup)
- Monitor GPU usage and temps (~80°C is fine, ~85°C max)
- Ensure proper cooling (AIO, rear fan, intake)


## 8. Run Inference on Fine-Tuned Model

You’ll get checkpoint directories after training. The highest-numbered checkpoint usually has the most training steps.

Use `test_bots.py` to try it out with any prompt.

## 9. Chat with CorrBot

Edit and run `chat_with_bot.py`. Customize:
- `temperature`: randomness (lower = more deterministic)
- `max_new_tokens`: length of response (higher = longer replies, more rambling)

## Tips (Learned the Hard Way)
- Set Windows power mode to **High Performance**
- Use AIO/rear/intake fans for airflow
- Set monitors to 60Hz for long sessions
- Let screen sleep after 15 mins (but don’t let PC sleep!)
- If using a UPS, avoid overloading it — plug heavy devices directly into the wall


## Examples (LLaMA 3.1-8B)

### 🔸 200k Examples
```
You: What are your thoughts about league of legends?
CorrBot: League of legends! I have heard so many stories and reviews about it. I'm sorry but I won't be playing league of legos. I'll pass. I get that people like it but I just don't get why people like games like that. I also don't understand why people spend so much time playing games. I have a guy in my class and he spends about 6 hours a day playing games and he says that he just likes to play games and that it is fun. I just can't understand that. He also says that the reason why he plays so much is because his parents don't limit him. I guess that would be nice living with one's parents. I personally don't like games that much and I don’t understand why some people are so addicted to it.
```

```
You: Its been good! Hbu??
CorrBot: I'm doing great, thank you! I'm getting to play villager for a day in AC New Leaf :D I have a gaming session today with the gang! They're all super nice and it's always something to look forward to even on a Monday! How 'bout you? Had any fun this weekend? :3
```

### 🔹 60k Examples
```
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
```

---

## What I Plan To Do Next

1. ✅ Fine-tuned on 200k messages using Llama3.1-8B-Instruct with QLoRA
2.  Train and fine-tune a ~350M parameter model from scratch
3.  Compare results between large and small models
4.  Publish results and host on my website
5.  Create a video showing everything I learned

---

Thanks for reading and good luck creating your own personalized AI bot!
