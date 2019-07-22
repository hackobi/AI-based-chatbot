# Plain Vanilla Tensorflow Chatbot

This code is the full code to make a chatbot in a very easy way implemented with a sequence to sequence model to train a chatbot in Tensorflow that will be able to hold a conversation once done.

The dataset used to train this version of the bot is the [Cornell Movie Dialogue Dataset](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) but feel free to train it with any corpus you want.

## Dependencies
* Numpy
* Six
* Tensorflow


## Setup

Create temporary working directory prior to training
`mkdir working_dir`

Download test/train data from Cornell Movie Dialog Corpus

`cd data/
bash pull_data.sh`

## Training

edit seq2seq.ini file to set 
``mode = train``

Once you've set the the model in training mode run the code:

```python execute.py```

afterwards, you can go back and edit the seq2seq.ini file and set it to 

```mode = test```

in order to test the bot. Note that you don't have to wait for the training to end in order to start testing it.

## UI

In order to run the bot from a browser run 

```python ui/app.py```

you'll then see the message
``    "Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)"``


copy the address and open it in a browser.

You can change the port in app.py in the folder [ui](https://github.com/hackobi/AI-based-chatbot/tree/master/ui)


Below you'll find a compilation of other great chatbot implementations for reference.

## Chatbot Implementations

### [ai-chatbot-framework](https://github.com/alfredfrancis/ai-chatbot-framework)

A python chatbot framework with Natural Language Understanding and Artificial Intelligence.

### [Chatbot](http://chatbot.sohelamin.com/)

An AI Based Chatbot

### [ChatterBot](http://chatterbot.readthedocs.io/)

ChatterBot is a machine learning, conversational dialog engine for creating chat bots

### DeepChatModels

[Conversation Models in Tensorflow](https://github.com/mckinziebrandon/DeepChatModels)

### [DeepQA](https://github.com/Conchylicultor/DeepQA)

My tensorflow implementation of "A neural conversational model", a Deep learning based chatbot

### [chatbot-rnn](https://github.com/pender/chatbot-rnn)

A toy chatbot powered by deep learning and trained on data from Reddit

### [Mybluemix Chatbot](https://webchatbot.mybluemix.net/)

Build your own chatbot base on IBM Watson

### [neural-chatbot](https://github.com/inikdom/neural-chatbot)

A chatbot based on seq2seq architecture done with tensorflow.

### [NeuralConvo](https://github.com/macournoyer/neuralconvo)

Neural conversational model in Torch

### [ParlAI](https://github.com/facebookresearch/ParlAI)

A framework for training and evaluating AI models on a variety of openly available dialog datasets.

### [tf_seq2seq_chatbot]([https://github.com/nicolas-ivanov/tf_seq2seq_chatbot)]

tensorflow seq2seq chatbot

### [stanford-tensorflow-tutorials](https://github.com/chiphuyen/stanford-tensorflow-tutorials/tree/master/assignments/chatbot)

A neural chatbot using sequence to sequence model with attentional decoder.


## Chinese_Chatbot

### Seq2Seq_Chatbot_QA

使用TensorFlow实现的Sequence to Sequence的聊天机器人模型

https://github.com/qhduan/Seq2Seq_Chatbot_QA

### Chatbot

基於向量匹配的情境式聊天機器人

https://github.com/zake7749/Chatbot

### chatbot-zh-torch7

中文Neural conversational model in Torch

https://github.com/majoressense/chatbot-zh-torch7


## Corpus

### Cornell Movie-Dialogs Corpus

http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html

### Dialog_Corpus

Datasets for Training Chatbot System

https://github.com/candlewill/Dialog_Corpus

### OpenSubtitles

A series of scripts to download and parse the OpenSubtitles corpus.

https://github.com/AlJohri/OpenSubtitles

### insuranceqa-corpus-zh

OpenData in insurance area for Machine Learning Tasks

https://github.com/Samurais/insuranceqa-corpus-zh

### dgk_lost_conv

dgk_lost_conv 中文对白语料 chinese conversation corpus

https://github.com/majoressense/dgk_lost_conv


## Papers

### Sequence to Sequence Learning with Neural Networks

http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf

### A Neural Conversational Model

http://arxiv.org/pdf/1506.05869v1.pdf

## Tutorials

### Research Blog: Computer, respond to this email.

https://research.googleblog.com/2015/11/computer-respond-to-this-email.html

### Deep Learning for Chatbots, Part 1 – Introduction

http://www.wildml.com/2016/04/deep-learning-for-chatbots-part-1-introduction/

### Deep Learning for Chatbots, Part 2 – Implementing a Retrieval-Based Model in Tensorflow

http://www.wildml.com/2016/07/deep-learning-for-chatbots-2-retrieval-based-model-tensorflow/


## Credits

Shout out to [suriyadeepan(https://github.com/suriyadeepan)  who put together most of the code as well as all the guys that have encountered issues and improved the base code.

See different issues people encountered and how they worked them around.

|Tensorflow Chatbot issues](https://github.com/llSourcell/tensorflow_chatbot/issues/3)
