#-- coding: utf-8 --
#@Time : 16/10/2021 下午 10:13
#@Author : wkq

import pandas as pd
from sklearn.decomposition import PCA
'''
2.6.2
案例：探究用户对物品类别的喜好细分
用户
物品类别
user_id
aisle
1）需要将user_id和aisle放在同一个表中 - 合并
2）找到user_id和aisle - 交叉表和透视表
3）特征冗余过多 -> PCA降维
'''
if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    # 1.获取数据
    order_products = pd.read_csv("../../../resource/instacart/order_products__prior.csv")
    products = pd.read_csv("../../../resource/instacart/products.csv")
    orders = pd.read_csv("../../../resource/instacart/orders.csv")
    aisles = pd.read_csv("../../../resource/instacart/aisles.csv")
    #2.合并表
    tab1 = pd.merge(aisles, products, on=["aisle_id", "aisle_id"])[:10000]
    tab2 = pd.merge(tab1, order_products, on=["product_id", "product_id"])[:10000]
    tab3 = pd.merge(tab2, orders, on=["order_id", "order_id"])[:10000]
    #3.找到user和aisle 之间的关系 列user_id 行aisle
    table = pd.crosstab(tab3["user_id"], tab3["aisle"])[:10000]
    data = table[:10000]
    #4.pac降维处理
    # 4.1实例化一个转换器 n_components=0.95降维保留95%的数据，n_components=2 降为2维
    transform = PCA(n_components=0.95)
    # 4.2调用fit_transform
    data_new = transform.fit_transform(data)
    print("data_new:\n",data_new)





