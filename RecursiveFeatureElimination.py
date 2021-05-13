# -*- coding: utf-8 -*-
# @Time : 2021/4/14 20:16
# @Author : 张涛
# @Software: PyCharm
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier as RFC

# 递归特征消除法
class RecursiveFeatureElimination:
    def recursive_feature_elimination(self,x_train,x_test,y_train):
        RFC_ = RFC(n_estimators=10, random_state=0)
        # 将训练集数据传入到RFE模型中训练，每次减少30个特征，一直到1000维停止
        selector = RFE(RFC_, n_features_to_select=1000, step=30).fit(x_train, y_train)
        return selector.transform(x_train),selector.transform(x_test)