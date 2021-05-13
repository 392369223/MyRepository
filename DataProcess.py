# -*- coding: utf-8 -*-
# @Time : 2021/4/14 19:20
# @Author : 张涛
# @Software: PyCharm
import pandas as pd

class DataProcess:
    # 定义有效信息列表（存储有效列的列名）
    useful_list = []

    # 删除无效列
    def removingInvalid(self,data):
        # 若数据集合并后仍然所有的数据项都相同，说明这一列完全没有用处，需要拿出来并删掉
        not_same = data.loc[:, (data != data.iloc[0]).any()]
        not_same = not_same.columns

        for index, row in data.iteritems():
            row.unique()
            counts = row.value_counts()
            if index in ('emd_lable2', 'train_or_test'):
                self.useful_list.append(index)
            elif index in ('emd_lable', 'birth_date', 'recent_flt_day'):
                continue
            elif index not in not_same:
                continue
            elif '0' in counts or '0.0' in counts:
                # 算出无效值的占比
                try:
                    percent = counts['0'] / data.shape[0]
                except:
                    percent = counts['0.0'] / data.shape[0]
                # 如果无效比值超过9成，视作完全无效，舍弃
                if percent > 0.90:
                    continue
                else:
                    self.useful_list.append(index)
            # 剩余的以字符串的形式读入
            else:
                self.useful_list.append(index)
        return data

    # 将一些无法处理的、处理难度较高的数据转化为可处理的、易处理的数据
    def prd(self,df):
        df = df.loc[:, self.useful_list]

        # 修改数据统计错误
        df['gender'] = df['gender'].replace('U', '0')

        # 起飞时间处理
        t = pd.to_datetime(df['seg_dep_time']).dt
        y, m, d, h = t.year, t.month, t.day, t.hour
        df['seg_dep_time_year'] = y

        # 春夏秋冬
        df.loc[(m >= 3) & (m <= 5), 'seg_dep_time_month'] = '1'
        df.loc[(m >= 6) & (m <= 8), 'seg_dep_time_month'] = '2'
        df.loc[(m >= 9) & (m <= 11), 'seg_dep_time_month'] = '3'
        df.loc[(m == 12) | (m <= 2), 'seg_dep_time_month'] = '4'

        # 前后半月
        df.loc[d <= 15, 'seg_dep_time_day'] = '1'
        df.loc[d > 15, 'seg_dep_time_day'] = '2'

        # 前后半天
        df.loc[d <= 12, 'seg_dep_time_hour'] = '1'
        df.loc[d > 12, 'seg_dep_time_hour'] = '2'

        # 会员号->有无会员
        df.loc[df['ffp_nbr'] != '0', 'ffp_nbr'] = '1'

        # 删除账号密码、航空
        df.drop(['seg_dep_time'], axis=1, inplace=True)
        return df

    # 返回有效列名列表
    def get_useful_list(self):
        return self.useful_list