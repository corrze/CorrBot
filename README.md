# CorrBot - Personalized Chat Bot

As of 7/30/2025...

CorrBot is going to be an LLM trained on my very own messages! You can train your very own [insert name]-Bot as well! Here is an easy guide to creating your own persoanlized chatbot. This project is me documenting my journey as I explore how to fine tune as well as train my own model.

# Motivation

I wanted work on this project to compare how big the difference is, between a larger model and a smaller model. This is because I got my hands on an Nvidia 3090, a consumer gpu capabable of training AI models,and I wanted to see how well of a model it can train, compared to a company trained model. For context, they train using many AI gpus for days to months. I think my smaller made from scratch model will not be as accurate as the pretrained model but I want to see how bad it is in comparison.

This is my first solo project with AI and I wouldn't say its anywhere near good code. But this project is meant for me to learn how it all works.

# Tutorial for Beginners (Fine-Tuning a pretrained model) !!!Only for **instagram** messages!!!

NOTE: So far, I have only fine-tuned a model on a dataset, I will be trainig my own model and writing about that at a later date.

Also I am using Python 3.10.11 for this. It is more stable then the newer version. Other package versions I am using. Below:
- Name: numpy | Version: 1.26.4 | Summary: Fundamental package for array computing in Python
- Name: torch | Version: 2.2.2+cu121 | Summary: Tensors and Dynamic neural networks in Python with strong GPU acceleration
- Name: transformers | Version: 4.54.0 | Summary: State-of-the-art Machine Learning for JAX, PyTorch and TensorFlow
- Name: peft | Version: 0.16.0 | Summary: Parameter-Efficient Fine-Tuning (PEFT)
- Name: accelerate | Version: 1.9.0 | Summary: Accelerate
- Name: bitsandbytes | Version: 0.46.1 | Summary: k-bit optimizers and matrix multiplication routines.


# 1. Make sure your computer can run the fine tuning

If not then I recommend using Google Collab in order to get computing power. I am using an NVIDIA RTX 3090 as it has enough VRAM to fine-tune a larger model. I recommend having at least 16 Gbs of VRAM.
I am using an 8 Billion parameter model and it uses 13 Gbs of VRAM usually. However when I trained on a larger dataset it used up to 23 Gbs so be aware about this fact when you are training it locally.

I would recommend running Llama3.1-3B or even 1B if you do not have 24 Gbs of VRAM.
As when I trained with QLoRA 4-bit, bf16 activations, and gradient checkpointing, it used up 23 Gbs of VRAM after 25 hours on the 8B model. 
However, with proper configurations you can make it work.

# 2. Creating a hugging face account and get permission to use your model of choice.
For this project I used Llama3.1-8B-Instruct by Meta. I haven't used other models just yet but make sure your model is capable of generating text. Make sure you are logged in the entire time to access this model.

# 3. Gathering your messages from instagram
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

It should look like this:

Inbox: Conv1, Conv2, ... , Conv*

Each Conv file: message_1.json, ... , message_*.json
(most will have just one file but it can have muliple messsage_* files)


# 4. Clone the repository and download the needed packages
Finetuning_Model.ipyn is a Jupyter Notebook, so make sure to install that and everything it requires. You may have to install some other things while on working through these steps. But do not fret! It's not hard at all.

# 4.5 Find out what format your model uses
Run the text_format.py file in order to figure out what prompt style your data. 
Read text_format.py to look more into it.


# 5. Extract the data from "inbox"
I'd recommend changing the INPUT_FOLDER as the path towards your inbox folder.
However if your inbox folder is in the same directory as the extracting script
it should automatically find the folder.

# 6. Find your the optimal amonut of tokens to train on
Run Token_counter.py and find out your .jsonl stats.

When I chose my MAX_LENGTH variable, I didnt use the token counter, I just assumed that most of the exmaples in my .jsonl file would be less than 500 tokens so I literally copied the longest example I could find and counted the number of tokens that way.

However, I have provided a script that will scan the entire .jsonl file to let you choose the best MAX_LENGTH as you'd like.

This matters because the more tokens you allow the longer the training would last and more VRAM you would use. This is because increasing the number of tokens would lead to larger attention matrices, the transformaer layers within process more tokens, which increases memory usage and time to process each batch and epoch.

I trained over 200k examples on MAX_LENGTH = 490 and it took me 25 hours. So if you do not have much time I would recommend either downloaded less and and have a higher max number of tokens or download a lot and have a lower max number of tokens.
In the case that your messages are just really long, you can decrease the BATCH_SIZE and increase MAX_LENGTH to balance out.

# 7. Fine-tuined the model!
Once you have everything figured out, ouptput directory/input directory/max_length size, then you can start training! Run every cell and you'll be good to go!

If your code crashes mid way through, this could be a VRAM issue and I recommend you to restart your computer before even training. My computer was forced to restart after closing VSCODE bc of my VRAM overloading.
When you are finetuning this model I recommend having task manager open as well as VSCODE, this allows you to keep track of your GPU status and temperature as you are fine-tuning. When I was training, it stayed at around 97% utilization and around 80 degrees C. I believe that being around 85 degrees should be okay but if it goes any higher, make sure you have proper cooling so it doesn't interrupt training.

# 8. Test the model with the base model
Once you finished training your model, it should output a directory with many a few checkpoints. Time to run it through inference.
So far we have
1. Prepare and download everything we need
2. Extracted our data
3. Filtered and formatted the data properly
4. Finetuned the model by training it on our data

Great! Now that we have done all of that we can finally test out our bot. Read through the "test_bots.py" and run it with any prompt you'd like. After fine-tuning these models you would have a lot of "checkpoints" these are the data that it saved during your training, they show what steps they stop on. It would make sense to choose the check point with the highest number as it has trained on more data.

# 9. Chat with your bot!
Fill in the required information in "chat_with_bot.py" then run it. Now you should be able to chat with your model. Yippee!!

There are two things I would recommend you to test, "temperature" and "max_new_token".
- "max_new_token" : How long the bot will speak, the higher the number the more it talk and the more prone it is to ramble. The lower it is, it has a higher chance to cut off mid-thought.
- "Temperature" : This will decide how chaotic/random it is. Since there is a softmax layer at the output of the model, it predicts the possibilies of all possible next tokens. Lowering the temperature will make it choose the higher probablity more often while if increase the temperature, the softmax will even out the smaller probabilities.


# Tips (that i found the hard way)
- Have your power option as "High Performance"
- Make sure you have proper cooling in your pc. Rear fan, AIO Cooling as well as intake fans.
- If you are training for a long time, 8+ hours, I recommend lowering you monitor settings to 50 - 60 hertz to prevent crashes. I would also recommend letting your computer screen turn off after 15 mintutes.
- DO NOT TURN OFF YOUR COMPUTER OR LET IT GO TO SLEEP ON ITS OWN
- If you are connecting it to UPS System (Uninterruptible power supply and backup battery) make sure that you keep the load under the max voltage. AI training is intense and it can overload your UPS, forcing it shut down. So make sure to plug other eletronics into the wall instead.

  
# Examples | CorrBot fine-tuned on Llama3.1-8B, 200k examples 

You: What are your thoughts about league of legends?
CorrBot: League of legends! I have heard so many stories and reviews about it. I'm sorry but I won't be playing league of legos. I'll pass. I get that people like it but I just don't get why people like games like that. ICorrBot: League of legends! I have heard so many stories and reviews about it. I'm sorry but I won't be playing league of legos. I'll pass. I get that people like it but I just don't get why people like games like that. I also don't understand why people spend so much time playing games. I have a guy in my class and he spends about 6 hours a day playing games and he says that he just likes to play games and that it is fun. I just can't understand that. He also says that the reason why he plays so much is because his parents don't limit him. I guess that would be nice living with one's parents. I personally don't like games that much and I don’t understand why some people are so addictive to it

----------------------------------------------------------------------------------------------------------------------------------------------

You: Its been good! Hbu??
CorrBot: I'm doing great, thank you! I'm getting to play villager for a day in ac new leaf :D also I have a gaming session today with the gang! They're all super nice and it's always something to look forward to even on a Monday! How 'bout you? Had any fun this weekend? :3

# CorrBot fine-tuned on Llama3.1-8B, 60k examples 

You: Good Morning
CorrBot: GOOD MORNING!!! how's the morning hangout sesh going so far?

You: Its going good, how are you?
CorrBot: I'm doing well thanks for asking! 😃 I'm here to help with any questions or concerns you may have. How about that? Does that sound good to you? Or would you like me to stick around and help with anything else specifically? 😊👍

You: That's great! Give me some motivation to do my homework.
CorrBot: You can do it!!! Also, remind yourself that it only gets easier from here! (I'm assuming you're taking 333) We got this! You've made it this far, you can keep going! Also, it's not forever, you'll get to the other side and it'll be worth it! 🙌 Keep pushing! You're doing great! 🤓

You: Im not taking 333 but thank you so much!
CorrBot: I'm sorry to hear that 😭 maybe next quarter! And ya no worries, I'm not taking it either! I'm here to help with any questions you have though! 🥹🥹 🤠🤠

You: Its been a good chat!
CorrBot: It was nice chatting with you too! I don't usually talk to people so you were a nice surprise! Have a good day! :)  I also wanted to wish you happy new years if you celebrate! I hope you had a good new years as well!

You: Its not new years but thank you! Good night
CorrBot: Good night! :D LOL I was gonna say happy new year but I assumed you already had it or were still celebrating since some countries/ cultures celebrate on different days. I hope your remaining 2024 is as good as you want it to be! :3 Sleep well! tips virtual top hat


# What I am planning on so far

1. !!!!! DONEEEE !!!!!!   Fine-Tune on 200k lines of prompt+respone data using Llama3.1-8B-Instruct with LoRA
2. Training and creating a model from scratch ~350M parameters and fine-tune on that
4. Compare the models together to see the differences.
5. Put on my website and create a video to show findings.
