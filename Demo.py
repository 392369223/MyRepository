# -*- coding: utf-8 -*-
# @Time : 2021/4/14 18:55
# @Author : 张涛
# @Software: PyCharm
from DataInput import DataInput
from DataProcess import DataProcess
from Encoder import Encoder
from SplitTrOrTe import SplitTrOrTe
from OverSample import OverSample
from VarianceFilter import VarianceFilter
from RecursiveFeatureElimination import RecursiveFeatureElimination
from Pca import Pca
from Ica import Ica
from Integration import Integration

if __name__ == '__main__':
    # 导入数据
    info = DataInput()
    df_train_data = info.load_train_data()
    df_test_data = info.load_test_data()
    del info
    # 添加标签，以便分隔
    df_train_data['train_or_test'] = '1'
    df_test_data['train_or_test'] = '2'
    #合并生成总集，方便编码
    df_all = df_train_data.append(df_test_data, ignore_index=True)

# 数据预处理:
    # 数据初处理
    pro = DataProcess()
    df_all = pro.prd(pro.removingInvalid(df_all))
    del pro
    print("数据初处理完成！")

    # 数据编码
    enc = Encoder()
    df_all = enc.encoder(enc.prepareEncoder(df_all))
    need_one_hot = enc.get_need_one_hot()
    del enc
    print("数据编码完成！")

    # 重新拆分训练集与测试集
    all_list = SplitTrOrTe().split(df_all,need_one_hot)
    x_train,y_train,x_test,y_test,test_user_id = all_list[0],all_list[1],all_list[2],all_list[3],all_list[4]
    del all_list
    print("重新拆分训练集与测试集完成！")

# 核心算法:
    # 过采样
    x_train, y_train = OverSample().over_sample(x_train,y_train)
    print("过采样完成！")

    # 方差过滤
    x_tr_temp,x_te_temp = VarianceFilter().variance_filter(x_train,x_test)
    print("方差过滤完成！")

    # 递归特征消除法
    x_tr_temp,x_te_temp = RecursiveFeatureElimination().recursive_feature_elimination(x_tr_temp,x_te_temp,y_train)
    print("递归特征消除法完成！")

    # PCA降维
    x_tr_1,x_te_1 = Pca().pca(x_tr_temp,x_te_temp)
    print("PCA降维完成！")

    # ICA降维
    x_tr_2,x_te_2 = Ica().ica(x_tr_temp,x_te_temp)
    print("ICA降维完成！")

    # 集成结果
    results = Integration().get_end_result(x_tr_1,x_te_1,x_tr_2,x_te_2,y_train,test_user_id)
    print("集成完成！")
    print("结果如下：")
    print(results)