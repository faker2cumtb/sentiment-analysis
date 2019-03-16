# -*- coding: utf-8 -*-
"""
一些字符串处理的方法
"""

import re

def string_is_float(str_float):
    """
    判断一个字符串是否是浮点数的字符串，整型也不是浮点数
    :param str_float:
    :return:
    """

    is_float = True
    try:
        int(str_float)
        is_float = False
    except ValueError:
        try:
            float(str_float)
        except ValueError:
            is_float = False

    return is_float


def format_weibo(weibo):  # , after_segment=False):
    """
    对于话题进行格式化，统一成一样含义的话题
    :param weibo:
    # :param after_segment: 是否分词之后的微博
    :return:
    """
    # 微博话题还原
    # if after_segment:
    #     pattern = re.compile('#[^#]+#')
    #     new_weibo = re.sub(pattern, lambda x: x.group(0).replace(' ', ''), weibo)
    # else:

    # topic 去掉表情符号
    pattern = re.compile('(\[[^\[\]]{1,10}\])+')
    new_weibo = re.sub(pattern, '', weibo)
    new_weibo = " ".join(new_weibo.strip().upper().split())

    return new_weibo


if __name__ == '__main__':
    digits = '12'

    print(string_is_float(digits))