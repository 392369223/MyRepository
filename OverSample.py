# -*- coding: utf-8 -*-
# @Time : 2021/4/14 20:05
# @Author : 张涛
# @Software: PyCharm
from imblearn.over_sampling import BorderlineSMOTE

# 对数据集进行过采样处理
class OverSample:
    def over_sample(self,x_train,y_train):
        sm = BorderlineSMOTE(random_state=151, kind='borderline-1')
        # 获取过采样后的数据集
        x_train, y_train = sm.fit_resample(x_train, y_train)
        return x_train, y_train