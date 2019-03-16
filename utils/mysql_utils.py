import pymysql
import logging
import traceback
import configparser

logger = logging.getLogger('mysqldb')


def get_mysql_conn(db_label):
    db = None

    try:
        cf = configparser.ConfigParser()
        cf.read('data/config.properties')

        # 打开数据库连接
        DATABASE_HOST = cf.get(db_label, 'DATABASE_HOST')
        DATABASE_USERNAME = cf.get(db_label, 'DATABASE_USERNAME')
        DATABASE_PASSWORD = cf.get(db_label, 'DATABASE_PASSWORD')
        DATABASE_NAME = cf.get(db_label, 'DATABASE_NAME')

        db = pymysql.connect(DATABASE_HOST, DATABASE_USERNAME, DATABASE_PASSWORD, DATABASE_NAME)
    except pymysql.err.OperationalError:
        logger.error('traceback.format_exc():\n%s' % traceback.format_exc())

    return db


# 执行mysql的sql语句，默认是查询，默认
# query : 是否是查询
# use_fields ： 则某一条记录是已经附加字段信息的字典格式的结果，否则是记录list加上字段描述的二元组
# 注意字段名称一定要不同！！！

# use_PK ： 是否把第一高字段当作是主键，如是返回字典格式，key是主键，如该不是，则返回记录的list，必须use_fields为True才生效


def execute_mysql_sql(db_label, sql, query=True, use_fields=True, use_PK=False):
    db = get_mysql_conn(db_label)
    if db:
        cursor = db.cursor()
        result = 1
    else:
        return 0

    try:
        # 执行SQL语句
        cursor.execute(sql)

        if query:
            result = cursor.fetchall()
            if use_fields:
                fields = cursor.description

                if use_PK:
                    res_data = {}
                    for row in result:
                        one_data = {}

                        for i in range(len(row)):
                            one_data[fields[i][0]] = row[i]

                        res_data[row[0]] = one_data
                else:
                    res_data = []
                    for row in result:
                        one_data = {}

                        for i in range(len(row)):
                            one_data[fields[i][0]] = '' if row[i] == 'nan' else row[i]

                        res_data.append(one_data)

                result = res_data
            else:
                result = (result, cursor.description)

            if not result: result = []

        else:
            # 提交到数据库执行
            db.commit()

    except Exception:
        logger.error(sql)
        logger.error('traceback.format_exc():\n%s' % traceback.format_exc())
        # 发生错误时回滚
        db.rollback()

        if query:
            result = []
        else:
            result = 0

    # 关闭数据库连接
    db.close()
    return result
