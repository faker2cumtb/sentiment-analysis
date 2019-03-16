# -*- coding: utf-8 -*-
"""
（1）语料格式
1）原始语料：xls格式，未分词
2）结果语料：
第一列为分类标签， \tab隔开，接着是已分词的文本，在一行中，
文件路径被传入分类器
3)缓存语料：pickle格式
4）标签说明，极性分类为0：负面，1中性，2正面

（2)字典生成 ，格式为文本格式，一行一个词

"""

import xlrd
import numpy as np
import random
import os
from pyltp import Segmentor
import six
import pickle
import multiprocessing
from gensim.models.word2vec import Word2Vec
import re

from . import file_utils as fu
from . import calculate_utils as cu
from . import mysql_utils as mysql

"""
PAD_CHAR 补齐的词
START_CHAR 句子开头
END_CHAR 句子结束
UNK_CHAR 忽略的词，或者说unknown的词
"""
PAD_CHAR = 0
START_CHAR = 1
END_CHAR = 2
UNK_CHAR = 3

STR_PAD_CHAR = "<PAD>"
STR_START_CHAR = "<START>"
STR_UNK_CHAR = "<UNK>"
STR_END_CHAR = "<END>"

MAX_SENTENCE_LENGTH = 40
VOCABULARY_SIZE = 6000

STOP_WORDS_FILE = 'data/dict/stopwords.txt'

DB_LABEL = "db_sentiment_corpus"

CACHE_DIR = 'data/cache'

N_GRAM = 3


class Corpus:
    """
    关于语料处理的类
    """

    def __init__(self, word_dict_path, data_from_db, file_path=None, sql_str=None, one_hot=True,
                 padding=True, sentence_len=MAX_SENTENCE_LENGTH, vocabulary_size=VOCABULARY_SIZE,
                 stop_words_file=STOP_WORDS_FILE, fasttext=False, for_test=False, is_shuffle=True):
        """
        构造方法
        :param word_dict_path: 词典路径
        :param data_from_db: 语料是否来自数据库
        :param file_path: 语料（格式为上面说明中结果语料）的文件路径
        :param sql_str: sql语句用于提取语料
        :param one_hot: 是否为np数组的格式，即标签为one-hot
        :param padding: 是否统一句子长度，不够的补齐，过长的截取
        :param sentence_len: 统一的句子长度
        :param stop_words_file: 停用词的文件路径
        :param fasttext: 是否使用fasttext模型
        :param for_test: True，用于测试（即eval和infer模式），False：用于训练
        :param is_shuffle: 对语料是否进行shuffle操作
        """
        self.sentence_len = sentence_len
        self.corpus_path = file_path

        if data_from_db and sql_str:
            corpus_list = get_corpus_list_from_db(sql_str=sql_str)
        elif data_from_db:
            corpus_list = get_corpus_list_from_db(data_path=file_path)
        else:
            corpus_list = get_corpus_list_from_file(file_path)

        self.data, self.labels, self.word_dict = prepare_data(corpus_list, word_dict_path, shuffle=is_shuffle,
                                                              vocabulary_size=vocabulary_size,
                                                              stop_words_file=stop_words_file)

        self.voc_size = len(self.word_dict)

        # 从0开始分类
        self.num_class = max(self.labels) + 1
        if one_hot:
            self.labels = cu.to_categorical(self.labels)
            self.num_class = self.labels.shape[1]

        if fasttext:
            self.voc_size, self.data = add_n_gram_features(self.data, N_GRAM, self.voc_size,
                                                           use_cached_ngram=for_test)

        if padding:
            self.data = pad_sequences(self.data, maxlen=sentence_len, padding='post', value=PAD_CHAR)


def prepare_data(corpus_list, word_dict_path, shuffle, vocabulary_size, stop_words_file=None):
    """
    从数据库中准备语料
    :param corpus_list:  语料list
    :param word_dict_path:  字典保存地址
    :param stop_words_file:停用词的文件路径
    :param shuffle:将语料转化成向量时是否需要shuffle
    :return:
    """
    if word_dict_path and os.path.exists(word_dict_path):
        # assert os.path.exists(word_dict_path), "The given word dictionary dose not exist."
        print("load_dict")
        word_dict = load_dict(word_dict_path)
    else:
        print ("gen_dict")
        word_dict = gen_dict(corpus_list, count_word_frequece = True,stop_words_file=stop_words_file, vocabulary_size=vocabulary_size)
        # print (word_dict)
        # 保存词典
        save_word_dict(word_dict_path, word_dict)
    # 将语料和标签转换成向量的形式, 同时向量
    x, y = corpus_to_vector(corpus_list, word_dict, is_shuffle=shuffle)

    return x, y, word_dict


def get_corpus_list_from_db(data_path=None, sql_str=None):
    """
    从数据库的中查询语料 放入corpus_list中
    :param data_path: sql语句放在文件里, 这里是文件的路径
    :param sql_str: sql语句
    :return: 包含语料的list
    """

    if sql_str:
        sql = sql_str
    else:
        assert os.path.exists(data_path), "The given sql_file_path does not exist."
        sql = get_sql_str(data_path)

    result = mysql.execute_mysql_sql(DB_LABEL, sql)
    corpus_list = []
    for record in result:
        label = str(record["label"])
        sen = process_text(record["text"])
        corpus_list.append(label + "\t" + sen)
    return corpus_list


def get_corpus_list_from_file(corpus_path):
    """
    从原料文件中生成包含语料的list
    :param corpus_path:语料文件路径
    :return:
    """
    assert os.path.exists(corpus_path), "The given data file does not exist."
    with open(corpus_path, "r", encoding="utf-8") as lines:
        corpus_list = [line.split("\t")[0] + "\t" + process_text(line.split("\t")[1]) for line in lines.readlines() if "\t" in line]
    return corpus_list


def process_text(text):
    """
    预处理语料
    :param text:
    :return:
    """
    sen = ' '.join(text.strip().split())
    sen = pre_process_weibo(sen)
    return sen


def get_sql_str(data_path):
    """
    获得文件中的sql语句
    :param data_path:
    :return:
    """
    with open(data_path, "r", encoding="utf-8") as file:
        sql_str = file.readline()
    return sql_str


def save_word_dict(word_dict_path, word_dict):
    """
    将词典保存成文件
    :param word_dict_path: 要将词典保存的路径
    :param word_dict:
    :return:
    """
    with open(word_dict_path, 'w', encoding="utf-8") as f:
        for k,v in word_dict.items():
            f.write(str(k) + "\t"+str(v)+'\n')


def gen_word_frequece_dict(word_dict,list, stop_words):
    """
    生成词频字典
    :param list: 词列表
    :param stop_words: 停用词列表
    :return: 词频字典
    """
    for word in list:
        if word not in stop_words:
            if word not in word_dict:
                word_dict[word] = 1
            else:
                word_dict[word] += 1
    return word_dict


def gen_dict(lines, vocabulary_size, stop_words_file=None, use_common_words=True, count_word_frequece=False):
    """
    从结果语料中生成词典
    :param lines: 语料list
    :param stop_words_file: 停用词文件路径
    :param use_common_words: 是否加入通用词
    :return:
    """

    wid = 0
    word_dict = {}
    word_frequece_dict = {}
    stop_words = []

    if stop_words_file:
        if os.path.exists(stop_words_file):
            with open(stop_words_file, 'r', encoding="utf-8") as f:
                stop_words = f.readlines()
            stop_words = [word.strip() for word in stop_words]

    for line in lines:

        cols = line.strip().split("\t")
        # assert len(cols) > 1, '语料格式有问题，句子为空'
        if len(cols) == 1 and len(cols[0]) == 1:
            # 第一列为label， 第二列为空
            cols.append('')
        else:
            assert len(cols) > 1, '语料格式有问题，句子为空: ' + cols[0]

        if count_word_frequece:
            word_frequece_dict = gen_word_frequece_dict(word_frequece_dict,cols[1].split(), stop_words)

        else:
            for word in cols[1].split():
                if word not in word_dict and word not in stop_words:
                    word_dict[word] = wid
                    wid += 1
    if count_word_frequece:
        sorted_word_frequece_list = sorted(word_frequece_dict.items(), key=lambda x: x[1], reverse=True)[
                                    :vocabulary_size]
        # print(sorted_word_frequece_list)
        for tuple in sorted_word_frequece_list:
            word_dict[tuple[0]] = wid
            wid += 1
    if use_common_words:
        word_dict = add_common_words_to_dict(word_dict)

    return word_dict


def add_common_words_to_dict(word_dict):
    """
    在原来词典中增加通用词
    :param word_dict: 原始词典
    :return:
    """
    # 增加4个通用词
    vocab_from = 4
    new_word_dict = {k: (v + vocab_from) for k, v in word_dict.items()}

    new_word_dict[STR_PAD_CHAR] = PAD_CHAR
    new_word_dict[STR_END_CHAR] = END_CHAR
    new_word_dict[STR_START_CHAR] = START_CHAR
    new_word_dict[STR_UNK_CHAR] = UNK_CHAR

    return new_word_dict


def reverse_word_index(vocab):
    """
    反转词典的k，v，为了能够打印出原始语料
    :param vocab:
    :return:
    """
    word_index = dict([(value, key) for (key, value) in vocab.items()])

    return word_index


def decode_review(text_vec):
    """
    根据词表的index值将词向量输出为原始语料
    :param text_vec:
    :return:
    """
    return ' '.join([reverse_word_index.get(i, '?') for i in text_vec])


def corpus_to_vector(lines, word_dict, is_shuffle=True):
    """
    参考百度Senta项目
    Convert word sequence into slot

    :param lines: 原始句子的list，一行一个句子
    :param word_dict: 字典
    :param is_shuffle: 是否打乱原来的顺序
    :return:
    """

    unk_id = UNK_CHAR
    all_data = []

    for line in lines:
        cols = line.strip().split("\t")
        # if len(cols) < 2: continue
        if len(cols) == 1 and len(cols[0]) == 1:
            # 第一列为label， 第二列为空
            cols.append('')
        else:
            assert len(cols) > 1, '语料格式有问题，句子为空: ' + cols[0]
        label = int(cols[0])
        wids = [word_dict[x] if x in word_dict else unk_id
                for x in cols[1].split(" ")]
        all_data.append((wids, label))
    if is_shuffle:
        random.shuffle(all_data)

    x = []
    y = []
    for doc, label in all_data:
        x.append(doc)
        y.append(label)
    return x, y


def load_dict(word_dict_path, use_common_words=True):
    """
    :param word_dict_path: 已有词典路径，词典的每一行为一个词
    :param use_common_words: 是否利用通用词
    load the given vocabulary
    """
    word_dict = {}
    with open(word_dict_path, encoding="utf-8") as f:
        for line in f:
            word_dict[line.strip().split("\t")[0]] =  int(line.strip().split("\t")[1])

    return word_dict

def pre_process_weibo(weibo_text):
    """
    处理的内容包括： # #话题，//@转发， 目标entity， @微博号， 表情
    停用词(常见无意义词，高频词，非汉语，数字，url）
    标点符号？
    :param weibo_text:
    :return:
    """
    # 替换entity, 避免对于实体对象产生记忆
    weibo_text = re.sub(re.compile("entity_7_(\d)+_target"), '[obj_star] ', weibo_text)
    weibo_text = re.sub(re.compile("entity_7_(\d)+([^_\d]|$)"), '[other_star] ', weibo_text).strip()
    weibo_text = re.sub(re.compile("entity_4_(\d)+_target"), '[obj_movie] ', weibo_text)
    weibo_text = re.sub(re.compile("entity_4_(\d)+([^_\d]|$)"), '[other_movie] ', weibo_text).strip()
    weibo_text = re.sub(re.compile("entity_5_(\d)+_target"), '[obj_telp] ', weibo_text)
    weibo_text = re.sub(re.compile("entity_5_(\d)+([^_\d]|$)"), '[other_telp] ', weibo_text).strip()
    weibo_text = re.sub(re.compile("entity_680_(\d)+_target"), '[obj_ent] ', weibo_text)
    weibo_text = re.sub(re.compile("entity_680_(\d)+([^_\d]|$)"), '[other_ent] ', weibo_text).strip()
    weibo_text = re.sub(re.compile("entity_790_(\d)+_target"), '[obj_general] ', weibo_text)
    weibo_text = re.sub(re.compile("entity_790_(\d)+([^_\d]|$)"), '[other_general] ', weibo_text).strip()

    # 去掉停用词
    # TODO 主体停用词目前是在生成词典的时候进行处理的，那么会用UNK_CHAR代替，还是在预处理的时候直接过滤掉
    # TODO 微博号的处理应该在分词之前

    # 去掉url
    weibo_text = re.sub(re.compile("http : / / t.cn / [a-zA-Z0-9]+"), '', weibo_text)

    # 去掉话题
    weibo_text = re.sub(re.compile("#.*#"), '', weibo_text)

    # 去掉转发的微博号
    weibo_text = re.sub(re.compile("/ / @ .* :"), '', weibo_text)

    # 整个去掉原微博 目前使用后性能则降低
    # weibo_text = re.sub(re.compile("/ / @ .*"), '', weibo_text)

    # 还原分开的表情
    re.sub(re.compile("\[ (.+?) \]"), r"[\1]", weibo_text)

    return weibo_text


def read_mysql_to_txt(corpus_type):
    db_label = 'db_sentiment_corpus'

    sql = 'SELECT s1.* from sentiment_label_no_aspect s1, sentiment_label_no_aspect s2 ' \
          'where s1.corpus_type = 5 and s1.origin_sentence_id = s2.id and s2.corpus_type = %d' % corpus_type
    result = mysql.execute_mysql_sql(db_label, sql)
    pass

    text_file = 'data/%s_%d.txt' % (db_label, corpus_type)
    with open(text_file, 'w', encoding="utf-8") as f:
        for sentence in result:
            label = 1 if sentence['label'] > 2 else sentence['label']
            sen = ' '.join(sentence['sentence_now'].strip().split())
            sen = pre_process_weibo(sen)
            f.write('%d\t%s\n' % (label, sen))


def read_xls_to_txt(path, sheets=[0], drop_head=True, label_col=0,
                    text_col=1, label_format='common', need_seg=False):
    """
    :param path:
    :param sheets:
    :param drop_head:
    :param label_col:
    :param text_col:
    :param label_format:
    :param need_seg:
    :return:
    """

    data = xlrd.open_workbook(path)  # 打开xls文件
    folder, label_text, _ = fu.get_file_info(path)
    label_text = folder + '/' + label_text + ".txt"
    with open(label_text, 'w') as fout:
        if need_seg:
            # import ainlpResource

            # resource = ainlpResource.Resource()
            # resourcepath = resource.getResource(True, "global")

            LTP_DATA_DIR = '/home/great/data/models/ltp/ltp_data_v3.4.0'
            cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  ##模型的地址
            ltp = Segmentor()
            ltp.load(cws_model_path)  # , resourcepath) load_with_lexicon

        for s in sheets:
            table = data.sheets()[s]  # 打开sheet1
            nrows = table.nrows  # 行数
            for i in range(nrows):
                if drop_head and i == 0:  # 跳过第一行
                    continue

                row_value = table.row_values(i)
                if label_format == 'common':
                    label = '0' if row_value[label_col] == -1 else '1' if row_value[label_col] == 0 else '2'
                else:
                    label_map = {-1: '0', 0: '1', 1: '2', 5: '1'}

                    label = label_map[row_value[label_col]] if row_value[label_col] in label_map else '1'

                text = ' '.join(row_value[text_col].strip().split())  # .encode('utf-8')
                if need_seg:
                    words = ltp.segment(text)
                    text = ' '.join(words)

                if not text: continue
                one_line = label + '\t' + text + '\n'
                fout.write(one_line)

        if need_seg:
            ltp.release()


def _remove_long_seq(maxlen, seq, label):
    """参考 tf.keras.load IMDB dataset的方法.
    Removes sequences that exceed the maximum length.

    # Arguments
        maxlen: Int, maximum length of the output sequences.
        seq: List of lists, where each sublist is a sequence.
        label: List where each element is an integer.

    # Returns
        new_seq, new_label: shortened lists for `seq` and `label`.
    """
    new_seq, new_label = [], []
    for x, y in zip(seq, label):
        if len(x) < maxlen:
            new_seq.append(x)
            new_label.append(y)
    return new_seq, new_label


def pad_sentence(sentence, sentence_length):
    """Pad the given sentense at the end.
 
    If the input is longer than sentence_length,
    the remaining portion is dropped.
    END_CHAR is used for the padding.

    Args:
      sentence: A numpy array of integers.
      sentence_length: The length of the input after the padding.
    Returns:
      A numpy array of integers of the given length.
    """
    sentence = sentence[:sentence_length]
    if len(sentence) < sentence_length:
        sentence = np.pad(sentence, (0, sentence_length - len(sentence)),
                          "constant", constant_values=(START_CHAR, END_CHAR))

    return sentence


def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    """Pads sequences to the same length.

    This function transforms a list of
    `num_samples` sequences (lists of integers)
    into a 2D Numpy array of shape `(num_samples, num_timesteps)`.
    `num_timesteps` is either the `maxlen` argument if provided,
    or the length of the longest sequence otherwise.

    Sequences that are shorter than `num_timesteps`
    are padded with `value` at the end.

    Sequences longer than `num_timesteps` are truncated
    so that they fit the desired length.
    The position where padding or truncation happens is determined by
    the arguments `padding` and `truncating`, respectively.

    Pre-padding is the default.

    # Arguments
        sequences: List of lists, where each element is a sequence.
        maxlen: Int, maximum length of all sequences.
        dtype: Type of the output sequences.
            To pad sequences with variable length strings, you can use `object`.
        padding: String, 'pre' or 'post':
            pad either before or after each sequence.
        truncating: String, 'pre' or 'post':
            remove values from sequences larger than
            `maxlen`, either at the beginning or at the end of the sequences.
        value: Float or String, padding value.

    # Returns
        x: Numpy array with shape `(len(sequences), maxlen)`

    # Raises
        ValueError: In case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if isinstance(value, six.string_types) and dtype != object and not is_dtype_str:
        raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
                         "You should set `dtype=object` for variable length strings."
                         .format(dtype, type(value)))

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def load_word_embedding_vectors(word_dict, vector_file):
    """

    :param word_dict: 词典
    :param vector_file: 词向量文件
    :return:
    """
    path, cache_name, _ = fu.get_file_info(vector_file)
    cache_file = os.path.join(path, cache_name + '.pkl')
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            word_embedding_vectors = pickle.load(f)
            return word_embedding_vectors

    with open(vector_file, 'r') as f:
        # first line
        s = f.readline()
        num_words, dimension = s.strip().split()
        dimension = int(dimension)

        print('######### 词向量文件 %s #################' % vector_file)
        print('词数： %s   维度： %d' % (num_words, dimension))

        # 初始化 索引为0的词语，词向量全为0
        word_embedding_vectors = np.zeros((len(word_dict), dimension))
        num_got_vec = 0
        num_lines = 0

        for line in f:
            segs = line.strip().split()
            word = segs[0]
            vector = segs[1:]
            num_lines += 1
            if word in word_dict:
                word_embedding_vectors[word_dict[word]] = vector
                num_got_vec += 1
                if num_got_vec % 1000 == 0:
                    print('### 已读取 %d 个词向量， 已生成 %d 个词向量' % (num_lines, num_got_vec))

    path, cache_name, _ = fu.get_file_info(vector_file)
    with open(os.path.join(path, cache_name + '.pkl'), 'wb') as f:
        pickle.dump(word_embedding_vectors, f)

    print('########### 原词表的词数为%d  加载了的向量的词数为：%d #########' % (len(word_dict), num_got_vec))

    return word_embedding_vectors


def average_sequence_length(sequences):
    """
    序列的每个元素为一个数组，此方法用来得到序列中的数组的平均长度
    :param sequences: 数组的序列
    :return: 平均长度
    """
    # print('Average train sequence length: {}'.format(  dtype=int)))
    return np.mean(list(map(len, sequences)))


def train_word_embedding_vectors(corpus, emb_dim):
    """
    从训练语料中得到词向量
    :param corpus: 语料对象
    :param emb_dim: 词向量的维数
    :return: 词向量数组
    """
    cpu_count = multiprocessing.cpu_count()  # 4
    n_exposures = 5  # 所有频数超过10的词语
    window_size = 15
    # n_iterations = 1  # ideally more..
    # n_epoch = 4
    # input_length = 100
    # maxlen = 100

    sentences = corpus.data
    text_array = [list(map(lambda x: str(x), item)) for item in sentences]

    model = Word2Vec(text_array, size=emb_dim, window=window_size, min_count=n_exposures, workers=cpu_count)
    # 初始化 索引为0的词语，词向量全为0
    word_embedding_vectors = np.zeros((len(corpus.word_dict), emb_dim))
    for index in model.vocab.keys():
        word_embedding_vectors[int(index)] = model[index]

    return word_embedding_vectors






