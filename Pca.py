# -*- coding: utf-8 -*-
# @Time : 2021/4/14 20:21
# @Author : 张涛
# @Software: PyCharm
from sklearn.decomposition import PCA

# PCA降维
class Pca:
    def pca(self,x_train,x_test):
        pca = PCA(n_components=250, svd_solver="randomized", random_state=0)
        # 返回PCA降维后的数据集
        return pca.fit_transform(x_train),pca.fit_transform(x_test)