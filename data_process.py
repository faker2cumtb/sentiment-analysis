# -*- coding: utf-8 -*-
#
# 查看parse_args方法的实现可以了解命令执行的参数
# ==============================================================================
import logging
import argparse
import ast
from utils import text_utils as tu

logger = logging.getLogger("paddle-fluid")
logger.setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser("Sentiment Classification.")
    # training data path
    parser.add_argument(
        "--train_data_path",
        type=str,
        required=False,
        help="The path of trainning data. Should be given in train mode!")
    # test data path
    parser.add_argument(
        "--test_data_path",
        type=str,
        required=False,
        help="The path of test data. Should be given in eval or infer mode!")
    # word_dict path
    parser.add_argument(
        "--word_dict_path",
        type=str,
        required=True,
        help="The path of word dictionary.")
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
        default="bilstm_net",
        help="type of model")
    # model save path
    parser.add_argument(
        "--model_path",
        type=str,
        default="models",
        required=True,
        help="The path to saved the trained models.")
    # Number of passes for the training task.
    parser.add_argument(
        "--num_passes",
        type=int,
        default=10,
        help="Number of passes for the training task.")
    # Batch size
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="The number of training examples in one forward/backward pass.")
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
    args = parser.parse_args()
    return args


def process_xls_file():
    # xls_file = 'data/jiangjinfu_data_seg.xlsx'
    xls_file = 'data/50000_data.xlsx'
    xls_file = 'data/lidan_data_seg.xlsx'
    xls_file = '/home/great/projects/rbfans/classifier/data/艺人极性标注-20181229-7-yamy-805条.xlsx'
    
    # xls_file = '/home/great/projects/rbfans/classifier/data/艺人极性标注-20181229-8-付辛博-823条.xlsx'
    # xls_file = '/home/great/projects/rbfans/classifier/data/艺人极性标注-20181229-5-孟美岐-808条.xlsx'
    # xls_file = '/home/great/projects/rbfans/classifier/data/艺人极性标注-20181229-9-陈羽凡-854条.xlsx'
    # xls_file = '/home/great/projects/rbfans/classifier/data/艺人极性标注-20181229-6-薛之谦-712条.xlsx'
    xls_file = '/home/great/projects/rbfans/classifier/data/艺人极性标注-20181229-14-吴宣仪-748条.xlsx'
    xls_file = '/home/great/projects/rbfans/classifier/data/艺人极性标注-20181229-12-王宝强-747条.xlsx'
    xls_file = '/home/great/projects/rbfans/classifier/data/艺人极性标注-20181229-2-李雨桐-794条.xlsx'
    xls_file = '/home/great/projects/rbfans/classifier/data/艺人极性标注-20181229-15-蒋劲夫-680条.xlsx'
    xls_file = '/home/great/projects/rbfans/classifier/data/艺人极性标注-20181229-13-胡歌-677条.xlsx'


    # tu.read_xls_to_txt(xls_file, label_col=1, text_col=2)
    # xls_file = '/home/great/projects/rbfans/classifier/data/20000_3_data.xlsx'
    tu.read_xls_to_txt(xls_file, label_col=0, text_col=2, label_format='4class', need_seg=True)

    
if __name__ == "__main__":
    # args = parse_args()
    process_xls_file()
