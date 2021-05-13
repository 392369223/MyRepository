# -*- coding: utf-8 -*-
# @Time : 2021/4/14 18:43
# @Author : 张涛
# @Software: PyCharm
import pandas as pd

# 读取csv文件，返回pandas对象
class DataInput:
    def load_train_data(self):
        return pd.read_csv('./datas/origin/df_train_data.csv', dtype=object)

    def load_test_data(self):
        return pd.read_csv('./datas/origin/df_test_data.csv', dtype=object)

    def load_label_data(self):
        return pd.read_csv('./datas/origin/df_label_data.csv', dtype=object)