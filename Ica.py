# -*- coding: utf-8 -*-
# @Time : 2021/4/14 20:24
# @Author : 张涛
# @Software: PyCharm
from sklearn.decomposition import FastICA

# ICA降维
class Ica:
    def ica(self,x_train,x_test):
        ica = FastICA(n_components=35, random_state=0)
        # 返回ICA降维后的数据
        return ica.fit_transform(x_train),ica.fit_transform(x_test)