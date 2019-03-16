# -*- coding: utf-8 -*-

from utils import text_utils as tu
from utils import calculate_utils as cu
import os
import pandas as pd

def cat():
    y = [1, 2, 3, 6]
    print(cu.to_categorical(y))


def file_test():
    # xls_file = 'data/jiangjinfu_data_seg.xlsx'
    xls_file = 'data/50000_data.xlsx'
    xls_file = 'data/lidan_data_seg.xlsx'
    xls_file = '/home/great/projects/rbfans/classifier/data/艺人极性标注-20181229-7-yamy-805条.xls'

    xls_file = '/home/great/projects/rbfans/classifier/data/艺人极性标注-20181229-8-付辛博-823条.xls'
    # xls_file = '/home/great/projects/rbfans/classifier/data/艺人极性标注-20181229-5-孟美岐-808条.xls'
    # tu.read_xls_to_txt(xls_file, label_col=1, text_col=2)
    xls_file = '/home/great/projects/rbfans/classifier/data/20000_3_data.xlsx'
    tu.read_xls_to_txt(xls_file, label_col=1, text_col=2, label_format='common', need_seg=True)


def vec_test():
    word_dict = tu.load_dict('data/3train.dict')
    vector_file = '/home/great/data/word2vec/sgns.weibo.bigram'
    vector_file = '/home/great/data/word2vec/vector.txt'
    vec = tu.load_word_embedding_vectors(word_dict, vector_file)
    pass


def vec_train():
    train_data_path = 'data/20000_3_data.txt'
    word_dict_path = 'data/20000_3_data.dict'
    corpus = tu.Corpus(train_data_path, word_dict_path=word_dict_path, padding=False)

    tu.train_word_embedding_vectors(corpus, 100)


def batch_eval_performance(batch_file_dir, model_name, batch=True):
    # --model_path models/save/cnn2_20000_3_train.h5 --mode train --train_data_path data/20000_3_data.txt  --epochs 20 --model_type cnn_2_layers
    # dict = '/home/great/projects/experiment/data/3train.dict'
    # dict = '/home/great/projects/experiment/data/20000_3_data.dict'
    dict = '/home/great/projects/experiment/experiment_record/dict/db_sentiment_corpus_2.dict'

    # model = 'models/save/bilstm_3train.h5'
    # model = 'models/save/bi_lstm.h5'
    # model = 'models/save/cnn2_20000_3_train.h5'
    # model = 'models/save/cnn1_20000_3_train.h5'
    # model = 'models/save/cnnlstm_20000_3_layer.h5'
    # model = 'models/save/stacked_lstm_20000_3_layer.h5'
    # model = 'models/save/corpus_2_nn_layer.h5'
    # model = 'models/save/corpus_2_stacked_lstm_layer.h5'
    # model = 'models/save/corpus_2_cnn2_layer.h5'
    # model = 'models/save/corpus_2_bilstm_layer_8.h5'
    # model = 'models/save/corpus_2_cnn2_layer_3.h5'
    # model = 'models/save/corpus_2_simplenn_layer_1.h5'

    # model = 'models/save/corpus_2_stacked_lstm_layer_2.h5'
    # model = 'models/save/corpus_2_cnn_lstm_layer_2.h5'
    # model = 'models/save/corpus_2_simple_nn_layer_1.h5'
    # model = 'models/save/corpus_2_fasttext_layer_1.h5'
    model = 'models/save/%s.h5' % model_name

    # bash; source activate tf;
    cmd_line_base = 'python sentiment_classify.py --word_dict_path %s --mode  eval --model_path %s --use_gpu True --data_file_path ' \
                    % (dict, model)

    if os.path.exists(batch_file_dir) and batch:
        for dirnames, paths, filenames in os.walk(batch_file_dir):
            for file in filenames:
                (file_name, extension) = os.path.splitext(file)
                if extension.lower() == '.txt':
                    test_file = os.path.join(dirnames, file)
                    cmd_line = cmd_line_base + " " + test_file
                    # print(cmd_line)
                    os.system(cmd_line)

    else:
        test_file = 'data/db_sentiment_corpus_3.txt'
        cmd_line = cmd_line_base + " " + test_file
        os.system(cmd_line)


def process_mysql_corpus():
    corpus_type = 2

    tu.read_mysql_to_txt(corpus_type)
    corpus_type = 3

    tu.read_mysql_to_txt(corpus_type)

def xlsx_txt(old_path,new_path):
    """
    数据库导出的xlsx文件转化为 label+"\t"+text 的txt语料格式
    :param old_path:
    :param new_path:
    :return:
    """
    # pd.read_excel(old_path)
    df = pd.read_excel(old_path)
    # df = df.reset_index(drop=True)
    # df.to_excel(old_path)
    df = df.loc[:,["label","target文本"]]
    print (df.head())
    with open(new_path,'w',encoding="utf8") as f:
        for i in range(df.shape[0]):
            # print(i)
            # print (df.loc[i,"target文本"])
            # print (str(df.loc[i,"label"])+"\t"+df.loc[i,"target文本"]+"\n")
            text = df.loc[i,"target文本"].replace('\n',"")
            f.write(str(df.loc[i,"label"])+"\t"+text.strip("//").strip()+"\n")

if __name__ == '__main__':
    # import sys
    #
    # # file_test()
    # # vec_test()
    # # vec_train()
    #
    # path = '/home/great/projects/rbfans/classifier/data/'
    # # path = 'data/test'
    # if len(sys.argv) > 1 and sys.argv[1] == 'eval':
    #     batch_eval_performance(path, sys.argv[2])
    # else:
    #     process_mysql_corpus()
    xlsx_txt("C:/Users\Administrator\Desktop/ALL_VAL_DATA_drop_dup.xlsx","C:/Users\Administrator\Desktop/ALL_VAL_DATA_drop_dup.txt")

# --model_path models/save/test.h5 --mode train --data_file_path data/db_sentiment_corpus_2.txt  --epochs 20 --model_type cnn_lstm -g train_in_layer

# --model_path models/save/test.h5 --mode train --sql_file_path experiment_record/sql/db_sentiment_corpus_2.sql  --epochs 20 --model_type cnn_lstm -g train_in_layer --sql_file_path_eval_when_train experiment_record/sql/db_sentiment_corpus_3.sql