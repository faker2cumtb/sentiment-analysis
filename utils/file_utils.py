# -*- coding: utf-8 -*-

import os


def get_file_info(file_full_name):
    """
	返回文件的所在路径，文件名称，文件后缀
	:param filename:
	:return:
	"""
    (file_path, temp_filename) = os.path.split(file_full_name)
    (file_name, extension) = os.path.splitext(temp_filename)
    return file_path, file_name, extension
