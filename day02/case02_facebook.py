#-- coding: utf-8 --
#@Time : 19/10/2021 下午 5:41
#@Author : wkq

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

if __name__ == '__main__':
    # 1.获取数据
    data = pd.read_csv("../resource/FBlocation/train.csv")
    # 2) 处理时间特征
    time_value = pd.to_datetime(data["time"], unit="s")
    date = pd.DatetimeIndex(time_value.values)
    data["day"] = date.day
    data["weekday"] = date.weekday
    data["hour"] = date.hour
    # 3) 过滤签到次数少的地点
    place_count = data.groupby("place_id").count()["row_id"]
    place_count[place_count > 3].head()
    data_final = data[data["place_id"].isin(place_count[place_count > 3].index.values)]
    data_final.head()
    # 筛选特征值和目标值
    x = data_final[["x", "y", "accuracy", "day", "weekday", "hour"]]
    y = data_final["place_id"]
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    # 3.特征工程标准化  此数据不用做降维处理
    transfer = StandardScaler()
        # fit() 计算 每一列的平均值、标准差
        # transform() (x-mean)/std进行最终的转换
    x_train = transfer.fit_transform(x_train)
        # 对测试集进行标准化，训练集调用fit_transfor(),测试集调用transform用训练集的平均值、标准差带入公示进行最终的转换
    x_test = transfer.transform(x_test)
    # 4.KNN算法预估器 p=1曼哈顿距离 p=2欧氏距离 默认欧氏距离
    estimator = KNeighborsClassifier()

    # 加入网格搜索和交叉验证
    # 参数准备
    params_dict = {"n_neighbors":[5,7,9]}
    # estimator:估计器对象 param_grid：估计器参数(dict){"n_neighbors":[1,9,11]} cv:指定几折交叉验证 fit():输入训练集数据 score()准确率
    estimator = GridSearchCV(estimator, param_grid=params_dict, cv=3)
    estimator.fit(x_train, y_train)
    # 5.模型评估
        #方法1：直接对比真实值和与测试
    y_predict = estimator.predict(x_test)
    print("y_predict:\n",y_predict)
    print("y_test:\n", y_test)
    print("直接对比真实值和预测值:\n",y_predict==y_test)
    #方法2：计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为:\n",score)

    # 最佳参数:best_params_
    print("最佳参数\n",estimator.best_params_)
    # 最佳结果：best_score_
    print("最佳结果\n",estimator.best_score_)
    # 最佳估计器：best_estimator_
    print("最佳估计器\n",estimator.best_estimator_)
    # 交叉验证结果：cv_results_
    print("交叉验证结果\n",estimator.cv_results_)