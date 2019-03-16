# sentimentanalysis
对微博评论进行情感三分类(正面,中性,负面)

[![license](https://img.shields.io/github/license/go88/fer2013-recognition.svg?style=for-the-badge)](https://choosealicense.com/licenses/mit/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=for-the-badge)](https://github.com/go88/fer2013-recognition/pulls)
[![GitHub (pre-)release](https://img.shields.io/github/release/go88/fer2013-recognition/all.svg?style=for-the-badge)](https://github.com/faker2cumtb/sentimentanalysis/releases)

---

## 目录

```text
data/       存放训练语料测试语料
    cache/    词向量
    dict/     停用词表
    config.properties    sql相关配置信息
experiment_record       实验记录
    dict/       语料词典
    model/      保存的模型
    sql/        sql语句
models/         各种模型
tools/          nlp相关工具
utils/          一些工具
data_process.py     数据预处理
funcs_test.py       功能测试
sentiment_classify.py       训练和测试

```

---
## 模型介绍
### 1. lstm_cnn 模型

先用lstm提取文本特征,再对文本特征进行卷积池化,多种大小的卷积核进行拼接,接全连接层得出分类结果

### 2. cnn_lstm 模型

先用cnn卷积出N_gram特征,多种大小的卷积核进行拼接,再输入lstm,接全连接层得出分类结果

### 3. lstm模型
多层lstm模型

### 4. cnn模型
多卷积核大小拼接的cnn模型
## 主程序介绍
python sentiment_classify.py    
终端运行 sentiment_classify.py文件需要输入的参数

| Parameter | Introduce | Demo |
| ------ | ------ | ------ |
|--model_path|模型路径|
|--mode|模式|train|
|--data_file_path|训练文件路径|
|--epochs|迭代次数|55|
|--model_type|模型名称|lstm_cnn|
|--use_embedding|词向量生成方法|train_in_layer|
|--vocabulary_size|限制词典大小|10000|
|--sentence_length|限制句子长度|40|
|--hidden_dim|隐层数|512|
|--embedding_dim|词嵌入层数|300|
|--batch_size|batch_size|512|
|--dropout|dropout|0.5|
|--lr|学习率|0.001|
|--word_dict_path|词典路径|
