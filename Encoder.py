# -*- coding: utf-8 -*-
# @Time : 2021/4/14 19:36
# @Author : 张涛
# @Software: PyCharm
import pandas as pd

class Encoder:
    # 定义一个列表存储需要进行one-hot编码的列名
    need_one_hot = []

    def encoder(self,data):
        # 对need_one_hot进行one-hot编码，并到data中然后返回
        return data.join(pd.get_dummies(data[self.need_one_hot]))

    def prepareEncoder(self,data):
        for index, row in data.iteritems():
            # 尝试转化为数字
            tmp = pd.to_numeric(data[index], errors='coerce').isnull().value_counts()
            # 如果此列能够转化为浮点数的数据能够超过九成五
            true_proportion = tmp[False] / data.shape[0] if (False in tmp) else 0
            if true_proportion > 0.95:
                # 除非是这几列
                if index in ('train_or_test', 'emd_lable2'):
                    continue
                # 那么就将此列强制转化为数字，不能转化为数字的各别值变为NaN
                tmp2 = pd.to_numeric(data[index], errors='coerce')
                # 对存在无效值的数据项
                if true_proportion != 1:
                    # 填入有效值的平均值（保留两位）
                    avg = tmp2[tmp2 > 0].mean()
                    tmp2.fillna(round(avg, 2), inplace=True)
                data.loc[:, index] = tmp2
            # 不能转化为数字，则需重新独热编码
            else:
                if index != 'pax_name' and index != 'pax_passport':
                    self.need_one_hot.append(index)
        return data

    def get_need_one_hot(self):
        return self.need_one_hot