# -*- coding: utf-8 -*-
#
# 查看parse_args方法的实现可以了解命令执行的参数
# models基于keras实现
#
# ==============================================================================

import logging
import argparse
import ast
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from utils import text_utils as tu
from utils import ml_utils as mu

from models import CNN_1_layers
from models import CNN_2_Layers
from models import SimpleNN
from models import SimpleLSTM
from models import BiLSTM
from models import CNN_LSTM
from models import StackedLSTM
from models import FastText
from models import GRU
from models import CNN_LSTM_2
from models import LSTM_CNN_2

logger = logging.getLogger("sentiment_classify")
logger.setLevel(logging.INFO)

EMB_TYPE_FROM_EXISTED = 'from_existed'  # 读取已有的向量文件
EMB_TYPE_TRAIN_FROM_CORPUS = 'train_from_corpus'  # 从语料中单独训练词向量
EMB_TYPE_TRAIN_IN_LAYER = 'train_in_layer'  # 在模型训练过程中同时生成词向量

# 验证集的分裂比例
VALIDATION_SPLIT = 0.2

model_pools = {
    'cnn_1_layer': CNN_1_layers,
    'cnn_2_layers': CNN_2_Layers,
    'simple_nn': SimpleNN,
    'simple_lstm': SimpleLSTM,
    'bi_lstm': BiLSTM,
    'cnn_lstm': CNN_LSTM,
    'stacked_lstm': StackedLSTM,
    'fasttext': FastText,
    'gru': GRU,
    'cnn_lstm_2':CNN_LSTM_2,
    'lstm_cnn_2':LSTM_CNN_2
}


def parse_args():
    parser = argparse.ArgumentParser("Sentiment Classification.")
    """文件格式的说明见text_util
    """

    # sql-file path
    parser.add_argument(
        "--sql_file_path",
        type=str,
        required=False,
        help="The path of sql-file.")
    # training data path
    parser.add_argument(
        "--data_file_path",
        type=str,
        required=False,
        help="The path of data. ")

    # for validate set: sql-file path when mode = train
    parser.add_argument(
        "--sql_file_path_eval_when_train",
        type=str,
        required=False,
        help="The path of eval sql-file when mode = train")

    # for validate set data path
    parser.add_argument(
        "--data_file_path_eval_when_train",
        type=str,
        required=False,
        help="The path of  eval data when mode = train ")

    # word_dict path
    parser.add_argument(
        "--word_dict_path",
        type=str,
        required=True,
        default=None,
        help="The path of word dictionary. If mode is train then this argument is the path of Generated dictionary. If not ,this argument is the path of the existed dictionaray ")
    # current mode
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=['train', 'eval', 'infer'],
        help="train/eval/infer mode")
    # model type
    parser.add_argument(
        "--model_type",
        type=str,
        choices=model_pools.keys(),
        default="cnn_lstm",
        help="type of model")
    # model save path
    parser.add_argument(
        "--model_path",
        type=str,
        default="models",
        required=True,
        help="The path to saved the trained models.")
    # lr value for training
    parser.add_argument(
        "--lr",
        type=float,
        default=0.002,
        help="The lr value for training.")
    # Whether to use gpu
    parser.add_argument(
        "--use_gpu",
        type=ast.literal_eval,
        default=False,
        help="Whether to use gpu to train the model.")
    # parallel train
    parser.add_argument(
        "--is_parallel",
        type=ast.literal_eval,
        default=False,
        help="Whether to train the model in parallel.")
    # dim of embedding layer
    parser.add_argument("-e", "--embedding_dim",
                        help="The dimension of the Embedding layer.",
                        type=int, default=128)
    # size of dictionary used
    parser.add_argument("-v", "--vocabulary_size",
                        help="The number of the words to be considered "
                             "in the dataset corpus.",
                        type=int, default=6000)
    # the max length of sentence trained
    parser.add_argument("-s", "--sentence_length",
                        help="The number of words in a data point."
                             "Entries of smaller length are padded.",
                        type=int, default=60)
    # dim of hidden CNN layer
    parser.add_argument("-c", "--hidden_dim",
                        help="The number of the CNN layer filters. / lstm layer dim/ NN dense dim",
                        type=int, default=128)
    # batch size
    parser.add_argument("-b", "--batch_size",
                        help="The size of each batch for training.",
                        type=int, default=500)
    # epoch number
    parser.add_argument("-p", "--epochs",
                        help="The number of epochs for training.",
                        type=int, default=55)
    # dropout rate
    parser.add_argument("-d", "--dropout",
                        help="dropout ratio training.",
                        type=float, default=0.5)
    # whether to gen embedding vectors from training corpus
    parser.add_argument("-g", "--use_embedding",
                        help="how to use embedding vectors ",
                        type=str,
                        choices=['from_existed', 'train_from_corpus', 'train_in_layer'],
                        default=EMB_TYPE_TRAIN_IN_LAYER)
    # model save path
    parser.add_argument(
        "-ep", "--embedding_path",
        type=str,
        default='/home/great/data/word2vec/sgns.weibo.word',
        help="The path to word vectors file.")
    args = parser.parse_args()
    return args


def eval_model(corpus, loaded_model=None, model_path=None):
    """
    验证函数
    """

    # score = new_model.evaluate(x_train, y_train, batch_size=batch_size)
    # tf.logging.info("Score: {}".format(score))
    # print(score)
    print('\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    print('#############  测试文件 %s  ###################' % corpus.corpus_path)
    print('#############  测试模型 %s  ###################\n' % model_path)

    if loaded_model is None:
        loaded_model = keras.models.load_model(model_path)

    preds = infer_model(corpus, loaded_model)
    y_test = corpus.labels

    mu.output_model_performance(preds, y_test)


def infer_model(corpus, loaded_model=None, model_path=None):
    """
    预测函数
    """

    x_test = corpus.data

    if loaded_model is None:
        loaded_model = keras.models.load_model(model_path)
    # score = new_model.evaluate(x_train, y_train, batch_size=batch_size)
    # tf.logging.info("Score: {}".format(score))
    # print(score)

    preds = loaded_model.predict(x_test)

    return preds


def train_model(model_type, emb_dim, hid_dim, batch_size, epochs, corpus, corpus_eval_when_train, dropout,
                embedding_type, voc_size, embedding_path=None, model_path=None):
    """训练和保存模型
    """

    assert model_path is not None, str(model_path) + "can not be found"
    print('\n#############  训练文件 %s  ###################' % corpus.corpus_path)
    print('#############  训练模型 %s  ###################\n' % model_path)

    embedding_vectors = np.array([])
    if embedding_type == EMB_TYPE_FROM_EXISTED:
        embedding_vectors = tu.load_word_embedding_vectors(corpus.word_dict, embedding_path)
    elif embedding_type == EMB_TYPE_TRAIN_FROM_CORPUS:
        embedding_vectors = tu.train_word_embedding_vectors(corpus, emb_dim)

    # voc_size = len(corpus.word_dict)
    sen_len = corpus.sentence_len
    num_class = corpus.num_class

    x_train = corpus.data
    y_train = corpus.labels

    model_class = model_pools[model_type]
    model = model_class(emb_dim=emb_dim,
                        num_words=voc_size,
                        sentence_length=sen_len,
                        hid_dim=hid_dim,
                        class_dim=num_class,
                        dropout_rate=dropout,
                        embedding_vectors=embedding_vectors,
                        # input_length=sen_len
                        ).model
    model.summary()

    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_acc', verbose=1, save_best_only=True,
                                                    save_weights_only=False, mode='auto', period=1)
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.05, patience=10, verbose=1, mode='auto')

    ################### for debug ###############################
    # from tensorflow.python import debug as tf_debug
    # sess = tf_debug.LocalCLIDebugWrapperSession(tf.Session(), dump_root='/home/great/temp/tmp')
    # # sess.add_tensor_filter('my_filter', my_filter_callable)
    # tf.keras.backend.set_session(sess)
    ##############################################################

    if corpus_eval_when_train:
        model.fit(x_train, y_train, batch_size=batch_size,
                  validation_data=(corpus_eval_when_train.data, corpus_eval_when_train.labels), epochs=epochs,
                  callbacks=[checkpoint, earlystop],class_weight="auto")
    else:
        model.fit(x_train, y_train, batch_size=batch_size,
                  validation_split=VALIDATION_SPLIT, epochs=epochs,
                  callbacks=[checkpoint, earlystop],class_weight="auto")

    # preds = model.predict(x_train)
    # y_test = corpus.labels
    #
    # # score = model.evaluate(x_train, y_test)
    # # print(score)
    #
    # mu.output_model_performance(preds, y_test)
    #
    # loaded_model = keras.models.load_model(model_path)
    # x_test = corpus_eval_when_train.data
    # y_test = corpus_eval_when_train.labels
    # data_file_path = 'data/test/db_sentiment_corpus_3.txt'
    # test_corpus = tu.Corpus(data_file_path, data_from_db=False)
    # x_test = test_corpus.data
    # y_test = test_corpus.labels
    # preds = loaded_model.predict(x_test)
    # mu.output_model_performance(preds, y_test)


def main(args):
    if args.sql_file_path:
        data_file_path = args.sql_file_path
        data_from_db = True
    else:
        data_file_path = args.data_file_path
        data_from_db = False

    fasttext = args.model_type == 'fasttext'

    # train mode
    if args.mode == "train":
        # 训练集的语料
        train_corpus = tu.Corpus(file_path=data_file_path, data_from_db=data_from_db,sentence_len=args.sentence_length,vocabulary_size=args.vocabulary_size,
                                 word_dict_path=args.word_dict_path, fasttext=fasttext)

        # 验证集的语料, 如为None则从训练集split
        corpus_eval_when_train = None
        if args.sql_file_path_eval_when_train:
            corpus_eval_when_train = tu.Corpus(file_path=args.sql_file_path_eval_when_train, data_from_db=True,
                                               word_dict_path=args.word_dict_path, fasttext=fasttext)
        elif args.data_file_path_eval_when_train:
            corpus_eval_when_train = tu.Corpus(file_path=args.data_file_path_eval_when_train, data_from_db=False,
                                               word_dict_path=args.word_dict_path, fasttext=fasttext)

        train_model(model_type=args.model_type,
                    emb_dim=args.embedding_dim,
                    hid_dim=args.hidden_dim,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    corpus=train_corpus,
                    corpus_eval_when_train=corpus_eval_when_train,
                    dropout=args.dropout,
                    voc_size=train_corpus.voc_size,
                    model_path=args.model_path,
                    embedding_type=args.use_embedding,
                    embedding_path=args.embedding_path)

    # eval_model(corpus=corpus, model_path=args.model_path)

    # paddle 作为参考
    # train_net(
    #     train_reader,
    #     word_dict,
    #     args.model_type,
    #     args.use_gpu,
    #     args.is_parallel,
    #     args.model_path,
    #     args.lr,
    #     args.batch_size,
    #     args.num_passes)

    # eval mode
    elif args.mode == "eval":
        assert args.model_path is not None, str(args.model_path) + "can not be found"
        loaded_model = keras.models.load_model(args.model_path)
        corpus = tu.Corpus(file_path=data_file_path, data_from_db=data_from_db,
                           fasttext=fasttext, word_dict_path=args.word_dict_path,
                           for_test=True)
        eval_model(corpus=corpus, loaded_model=loaded_model, model_path=args.model_path)

    # infer mode
    elif args.mode == "infer":
        assert args.model_path is not None, str(args.model_path) + "can not be found"
        loaded_model = keras.models.load_model(args.model_path)
        corpus = tu.Corpus(file_path=data_file_path, data_from_db=data_from_db,
                           fasttext=fasttext, word_dict_path=args.word_dict_path,
                           for_test=True)
        return infer_model(corpus=corpus, loaded_model=loaded_model, model_path=args.model_path)


if __name__ == "__main__":
    """
    readme: 
        1.如果是从mysql中读取语料, 那么sql语句最后返回的结果一定要有label和text两个字段, 也就是说需要对语料字段进行重命名
          e.g. select label, sentence_now as text from sentiment_label_no_aspect where corpus_type = 3;
        2.生成的字典文件和model文件在项目的experiment_record中
    """
    # 屏蔽日志
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    argments = parse_args()
    main(argments)

# train parameters
# --model_path models/save/cnn2layers.h5 --mode train --train_data_path data/3train.txt --word_dict_path data/3train.dict --epochs 2 --model_type simple_nn
# --model_path models/save/lstm_age.h5 --mode train --train_data_path /home/great/projects/age/data/age_20w.txt --epochs 20 --model_type lstm

# python sentiment_classify.py  --model_path models/save/cnn_lstm_all_corpus_2_v1.h5 --mode train --data_file_path C:/Users\Administrator\Desktop/ALL_TRAIN_DATA_drop_dup.txt --epochs 2 --model_type cnn_lstm_2 -g train_in_layer --data_file_path_eval_when_train C:/Users\Administrator\Desktop/ALL_VAL_DATA_drop_dup.txt --vocabulary_size 4000 --sentence_length 40 --hidden_dim 10 --embedding_dim 10 --batch_size 1000 --dropout 0.8 --lr 0.001 --word_dict_path my_test_dict.txt

# eval parameters
# --model_path models/save/cnn2layers.h5 --mode eval --test_data_path data/3test.txt --word_dict_path data/3train.dict
# --model_path models/save/lstm_age.h5 --mode eval --test_data_path data/3test.txt --word_dict_path data/3train.dict

## 服务器上
# python sentiment_classify.py  --model_path /home/dminer/luoxinyu/╟щ╕╨╖╓╬Ў╩╡╤щ19_2_14/models/save/cnn_lstm_2_all_corpus_2_v2.h5 --mode train --data_file_path /home/dminer/luoxinyu/╟щ╕╨╖╓╬Ў╩╡╤щ19_2_14/data/ALL_TRAIN_DATA_drop_dup.txt --epochs 55 --model_type cnn_lstm_2 -g train_in_layer --data_file_path_eval_when_train /home/dminer/luoxinyu/╟щ╕╨╖╓╬Ў╩╡╤щ19_2_14/data/ALL_VAL_DATA_drop_dup.txt --vocabulary_size 10000 --sentence_length 40 --hidden_dim 512 --embedding_dim 512 --batch_size 1024 --dropout 0.5 --lr 0.01 --word_dict_path /home/dminer/luoxinyu/╟щ╕╨╖╓╬Ў╩╡╤щ19_2_14/experiment_record/dict/cnn_lstm_2_dict_2.txt
