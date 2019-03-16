# -*- coding: utf-8 -*-
#
# 格式为 "%Y-%m-%d %H:%M:%S"， 如'2018-10-01 00:00:01'
#
#  常用函数
#  (1) 时间戳转为struct_time对象
#    time.localtime(ts))
#    time.struct_time(tm_year=1970, tm_mon=1, tm_mday=1, tm_hour=8, tm_min=0, tm_sec=12,
#                     tm_wday=3, tm_yday=1, tm_isdst=0)
#  (2)
# date_obj = datetime.date(int(start_date[0:4]), int(start_date[5:7]), int(start_date[8:10]))
# end_date = (date_obj + datetime.timedelta(days=2)).strftime("%Y-%m-%d")
#   (3) 某一天开始的时间戳
#  tu.string_to_timestamp('%d-%02d-%02d 00:00:00' % (year, month, day))

import time
import datetime

def datetime_to_string(dt):
    """
    # 把datetime转成字符串
    :param dt:
    :return:
    """
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def string_to_datetime(str_time):
    """
    #把字符串转成datetime
    :param str_time:
    :return: time.struct_time(tm_year=2018, tm_mon=10, tm_mday=1, tm_hour=0,
    tm_min=0, tm_sec=1, tm_wday=0, tm_yday=274, tm_isdst=-1)

    """
    return time.strptime(str_time, "%Y-%m-%d %H:%M:%S")


def string_to_timestamp(str_time):
    """
    #把字符串转成时间戳形式
    :param str_time:
    :return:
    """
    return time.mktime(string_to_datetime(str_time))


def timestamp_to_string(stamp):
    """
    #把时间戳转成字符串形式
    :param stamp:
    :return:
    """
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stamp))


def datetime_to_timestamp(date_time):
    """
    #把datetime类型转外时间戳形式
    :param date_time:
    :return:
    """
    return time.mktime(date_time.timetuple())


def change_days(com_date, day):
    """
    推算多少天之后的日期，
    :param com_date: 日期字符串，格式为‘2018-10-01’  %Y-%m-%d
    :param day: 推算的天数,，往后是整数，之前是负数
    :return: 推算之后的日期字符串
    """
    date_obj = datetime.date(int(com_date[0:4]), int(com_date[5:7]), int(com_date[8:10]))
    days_com_date = (date_obj + datetime.timedelta(days=day)).strftime("%Y-%m-%d")

    return days_com_date


def get_the_last_day_of_month(year, month):
    """
    返回某个月的最后一天的日期
    :param year: int
    :param month: int
    :return: 日期，int
    """
    next_month = month + 1 if month < 12 else 1
    last_day = datetime.date(year, next_month, 1) - datetime.timedelta(days=1)

    return last_day.day


def gen_date_series(begin_date, end_date):
    """
    生成一个连续日期的序列
    :param begin_date:
    :param end_date:
    :return: list of date string
    """
    this_date = begin_date
    date_obj = datetime.date(int(begin_date[0:4]), int(begin_date[5:7]), int(begin_date[8:10]))
    end_date_obj = datetime.date(int(end_date[0:4]), int(end_date[5:7]), int(end_date[8:10]))
    date_series = []

    while date_obj <= end_date_obj:
        date_series.append(this_date)
        date_obj = date_obj + datetime.timedelta(days=1)
        this_date = date_obj.strftime("%Y-%m-%d")

    return date_series