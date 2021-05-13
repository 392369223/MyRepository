# -*- coding: utf-8 -*-
# @Time : 2021/4/14 20:10
# @Author : 张涛
# @Software: PyCharm
from sklearn.feature_selection import VarianceThreshold

# 方差过滤
class VarianceFilter:
    def variance_filter(self,x_train,x_test):
        # 经过调参，threshold为6e-5时效果最好
        selector = VarianceThreshold(6e-5)
        # 返回方差过滤后的数据集
        return selector.fit_transform(x_train),selector.transform(x_test)