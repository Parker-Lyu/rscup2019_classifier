# -*- coding: utf-8 -*-
"""
@File    : get_logger.py
@Time    : 2019/6/22 12:32
@Author  : Parker
@Email   : now_cherish@163.com
@Software: PyCharm
@Des     : 
"""

def get_logger(log_file):
    import logging

    # 1、创建一个logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 2、创建一个handler，用于写入日志文件
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # 再创建一个handler，用于输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # 3、定义handler的输出格式（formatter）
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 4、给handler添加formatter
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # 5、给logger添加handler
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

if __name__ == '__main__':
    logger = get_logger('./logs/0622log.log')
    logger.info('gggooo')