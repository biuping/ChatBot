# ChatBot
对话机器人，包含了看图说话，单轮对话和多轮对话，tensorflow 2.0 pytorch 1.3.1 GPT-2
### 开发环境
    - flask==1.0.2
    - tensorflow==2.0.0
    - pytorch==1.3.1
    - sklearn==0.19.2
    - scipy==1.4.1
    - numpy==1.18.5
    - jieba==0.42.1
    - pandas==0.23.4
    - torchvision==0.5.0
    - transformers==2.1.1  

### 使用
- 启动前端，可以在pycharm中直接启动
- 启动app.py
- 回车或点击左爪发送消息，点击右爪发送图片，点击左耳切换图片描述和图片描述注入对话模型两种模式，点击右耳切换多轮对话和单轮对话模式  
![Alt 前端使用](/result/use.jpg)  
---

### 单轮对话  
1. 采用小黄鸡作为对话语料，总共450000对话数据，不过有部分噪音和特殊符号
2. 利用pandas从xhj.csv中读入对话，第一列为问题，第二列为回答，分别将其分离到question和answer数组，并进行预处理，在每个句子前加上'start ',句子后加上' end'。xhj.csv已经用jieba分词处理过，但在预测时输入句子要进行jieba分词和预处理。
3. 模型采用seq2seq，encoder和decoder采用GRU网络，利用BahdanauAttention实现注意力机制。将input注入encoder获得encoder-output和encoder-hidden（decoder-hidden），然后将['start']作为decoder第一个input，和decoder-hidden，encoder-output注入decoder，将encoder-output和decoder-hidden注入BahdanauAttention获得注意力权重和context-vector；把input嵌入到对应维度，与context-vector连接起来然后输入到GRU，最后通过Dense层输出output和decoder-hidden，利用output和target(目标，即回答)，此时decoder的input为target[:, t]，重复decoder操作。
4. 预测：用jieba将句子分词后预处理，然后通过tokenizer获取编码，然后转化成张量。初始hidden为0矩阵，shape为[1, units]，同样以['start']作为第一个input注入decoder，获得output，取第一行最大值通过tokenizer转换为词，重复直到句子的最大length结束，将这些词拼起来即为回答句子。  
![Alt 模型图](/result/single_model.jpg)
---
<center>蓝色部分为encoder，红色为decoder</center>


### 文件目录  
```
│  .gitignore
│  app.py
│  README.md
│               
├─gpt_2
│  │  dataset.py
│  │  gpt_train.py
│  │  predict.py
│  │  
│  ├─config
│  │      model_config_dialogue_small.json
│  │      
│  ├─data
│  │      interacting_mmi.log
│  │      train.txt
│  │      training.log
│  │      train_tokenized.txt
│  │      
│  ├─model
│  │      config.json
│  │      pytorch_model.bin
│  │      
│  ├─sample
│  │      samples.txt
│  │      
│  ├─tensorboard_summary
│  │      
│  ├─vocabulary
│  │      vocab_small.txt
│  │      
│  └─__pycache__
│          predict.cpython-37.pyc
│          
├─see_speak
│  │  load_func.py
│  │  preprocess_img.py
│  │  see_evaluate.py
│  │  see_hparam.py
│  │  see_train.py
│  │  
│  ├─data
│  │      features.pkl
│  │      Flickr_8k.testImages.txt
│  │      Flickr_8k.trainImages.txt
│  │      jieba_description.txt
│  │      tokenizer.pkl
│  │      
│  │      
│  └─_model
│  │      see_and_speak.h5
│          
├─single_chat
│  │  data.py
│  │  dialog.py
│  │  hparam.py
│  │  s2sModel.py
│  │  __init__.py
│  │  
│  ├─data
│  │      xhj.csv
│  │      
│  │      
│  └─model
│          
├─static
│  │  index.html
│  │  
│  ├─assets
│  │      autoload.js
│  │      flat-ui-icons-regular.eot
│  │      flat-ui-icons-regular.svg
│  │      flat-ui-icons-regular.ttf
│  │      flat-ui-icons-regular.woff
│  │      live2d.js
│  │      waifu-tips.js
│  │      waifu-tips.json
│  │      waifu.css
│  │      
│  ├─css
│  │      index.css
│  │      
│  ├─img
│  │      5.jpg
│  │      bg.jpg
│  │      catbtn.png
│  │      chatbg.jpg
│  │      ear.png
│  │      left.png
│  │      right.png
│  │      
│  ├─js
│  │      axios.js
│  │      jquery.js
│  │      main.js
│  │      vue.js
│  │      
│  └─layui
│                  
└─temp

```
