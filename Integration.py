# -*- coding: utf-8 -*-
# @Time : 2021/4/14 20:27
# @Author : 张涛
# @Software: PyCharm
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RFC

# 对两种降维方法得出的结果进行集成
class Integration:
    # 将模型，训练的数据，测试的数据传入函数，模型经过训练后传入测试集预测，获得预测的结果，然后按照概率排序,然后排序
    def get_result(self, model, train_x, train_y, test_x, user):
        """
        model:训练的模型
        train_x:训练的数据x
        train_y:训练的标签
        test_X:测试的数据
        test_y:测试的标签
        user:用户含pax_name,pax_passport的表、
        df_:排序的结果
        """
        model.fit(train_x, train_y)
        df = pd.DataFrame(model.predict_proba(test_x), columns=['0', '1'])
        df = pd.concat([user, df['1']], axis=1)
        df_ = df.sort_values(by='1', ascending=False)
        return df_

    # 将获得的多个预测结果合并
    def merge(self, data):
        """
        data:表示存放结果的列表
        l:表示最终合并的结果
        """
        # data在这里是一个盛放着三个结果的列表
        l = data[0]
        for i in range(1, len(data)):
            l = pd.concat([l, data[i]], axis=0, ignore_index=True)
        return l

    # 计算合并后的数据的权重合概率平均值，只能找先按照平均概率在按照权重进行倒排序，最后获得前i条结果
    def calculate(self, l, i):
        """
        l:表示合并的结果
        i:表示保留多少条数据
        """
        # 采用分组求平均值的方式计算新概率
        result = l.groupby(['pax_name', 'pax_passport'])
        dic = {'pax_name': [], 'pax_passport': [], '1': [], 'weight': []}
        for key, value in result:
            dic['pax_name'].append(key[0])
            dic['pax_passport'].append(key[1])
            dic['1'].append(value['1'].mean())
            dic['weight'].append(len(value['1'].index))
        result = pd.DataFrame(dic)
        # 先按准确率排序，在按照权重排序
        result.sort_values(by=['1', 'weight'], ascending=False, inplace=True)
        result = result.iloc[:i, :]
        result.index = list(range(0, len(result.index)))
        return result

    # 利用前面定义的三个方法获得结果后，将结果保存到csv文件
    def get_end_result(self,x_tr_1,x_te_1,x_tr_2,x_te_2,y_train,test_user_id):
        # 使用x_test3作为测试集，随机森林选择40最好
        RFC1 = RFC(n_estimators=40, random_state=0)
        result1 = self.get_result(RFC1, x_tr_1, y_train, x_te_1, test_user_id)
        RFC3 = RFC(n_estimators=140, random_state=0)
        result3 = self.get_result(RFC3, x_tr_1, y_train, x_te_1, test_user_id)
        # 使用x_test4作为测试集，随机森林选择140最好
        RFC2 = RFC(n_estimators=140, random_state=0)
        result2 = self.get_result(RFC2, x_tr_2, y_train, x_te_2, test_user_id)
        RFC4 = RFC(n_estimators=130, random_state=0)
        result4 = self.get_result(RFC4, x_tr_2, y_train, x_te_2, test_user_id)
        result1 = result1[:150]
        result2 = result2[:150]
        result3 = result3[:150]
        result4 = result4[:150]
        results = self.calculate(self.merge([result1,result2,result3,result4]), 150)
        results = results[['pax_name','pax_passport','1']]
        results.to_csv("./datas/results.csv", index=False)
        return results