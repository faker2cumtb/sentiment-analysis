# -*- coding: utf-8 -*-

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from tensorflow import keras


# import keras


class MLModel():
    """机器学习的模型的基本类"""

    def __init__(self, emb_dim, embedding_vectors, num_words, sentence_length,
                 hid_dim=None,
                 class_dim=None,
                 dropout_rate=None):
        """
		  emb_dim: The dimension of the Embedding layer.
		  num_words: 所使用的词向量的个数
		  embedding_vectors: 预训练的词向量
		"""
        self.emb_layer = self.create_embedding_layer(embedding_vectors, num_words, emb_dim, sentence_length)

    def create_embedding_layer(self, embedding_vectors, num_words, emb_dim, input_length):
        if embedding_vectors.any():
            emb_dim = embedding_vectors.shape[1]

            layer = keras.layers.Embedding(input_dim=num_words,
                                           weights=[embedding_vectors],
                                           trainable=False,
                                           output_dim=emb_dim,
                                           input_length=input_length)
        else:
            layer = keras.layers.Embedding(input_dim=num_words,
                                           output_dim=emb_dim,
                                           input_length=input_length)

        return layer


def batch(reader, batch_size, drop_last=True):
    """
	参考百度Senta项目
	Create a batched reader.

	:param reader: the data reader to read from.
	:type reader: callable
	:param batch_size: size of each mini-batch
	:type batch_size: int
	:param drop_last: drop the last batch, if the size of last batch is not equal to batch_size.
						注意当batch_size > 全部样本个数的时候，一定要把drop_last设为 False，否则无数据返回
	:type drop_last: bool
	:return: the batched reader.
	:rtype: callable
	"""

    def batch_reader():
        r = reader()
        b = []
        for instance in r:
            b.append(instance)
            if len(b) == batch_size:
                yield b
                b = []
        if drop_last == False and len(b) != 0:
            yield b

    return batch_reader


def output_model_performance(preds, labels):
    """
	输出总体正确率，每个类的精准率和召回率，f1，以及混淆矩阵
	:param preds: 预测的分类结果
	:param labels: 正确的分类标签
	:return:
	"""
    preds = np.argmax(preds, axis=1)
    labels = np.argmax(labels, axis=1)

    print('#############   总体正确率=%f   ###################' % accuracy_score(labels, preds))
    print('\n#############   每个类的精准率，召回率，F1   ###################')
    print(classification_report(labels, preds))
    print('#################       混淆矩阵       #######################')

    cm = confusion_matrix(labels, preds)
    print(cm)

    pred_data = np.sum(cm, axis=0)
    real_data = np.sum(cm, axis=1)
    real_good_ratio = real_data[2] * 1.0 / (real_data[0] + real_data[2])
    pred_good_ratio = pred_data[2] * 1.0 / (pred_data[0] + pred_data[2])
    print('\n#################       好评率       #######################')
    print('实际好评率=%f   预测好评率=%f' % (real_good_ratio, pred_good_ratio))
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')
