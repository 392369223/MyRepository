# -*- coding: utf-8 -*-
# @Time : 2021/4/14 19:46
# @Author : 张涛
# @Software: PyCharm

# 重新拆分训练集与测试集
class SplitTrOrTe:
    def split(self,data,need_one_hot):
        # 目标列
        tmpemd = data.pop('emd_lable2')
        data.insert(data.shape[1], 'emd', tmpemd)
        # 删除被重新编码的原列
        data.drop(need_one_hot, axis=1, inplace=True)
        # 记录测试集的用户id
        test_user_id = data.loc[data['train_or_test'] == '2', ["pax_name", "pax_passport"]]
        test_user_id.reset_index(inplace=True, drop=True)
        # 删除id列
        data.drop(["pax_name", "pax_passport"], axis=1, inplace=True)
        df_train_data = data.loc[data['train_or_test'] == '1', :]
        df_test_data = data.loc[data['train_or_test'] == '2', :]
        # 重置索引
        df_test_data.reset_index(inplace=True, drop=True)
        x_train = df_train_data.iloc[:, :-1]
        y_train = df_train_data.iloc[:, -1]
        x_test = df_test_data.iloc[:, :-1]
        y_test = df_test_data.iloc[:, -1]
        return [x_train,y_train,x_test,y_test,test_user_id]