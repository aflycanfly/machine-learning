# -- coding: utf-8 --
# @Time : 21/10/2021 下午 5:21
# @Author : wkq
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error
import joblib


'''
    关于优化的方法：GD,SGD，SAG
'''


def linear_LinearRegression():
    '''
    正规方程的优化方法对波士顿房价进行预测
    :return:
    '''
    # 1.数据获取
    boston = load_boston()
    # 2.划分数据集
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=22)
    # 3.标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # 4.预估器
    estimator = LinearRegression()
    estimator.fit(x_train, y_train)
    # 5.得出模型
    print("正规方程-权重系数\n", estimator.coef_)
    print("正规方程-偏置为\n", estimator.intercept_)
    # 6.模型评估
    y_predict = estimator.predict(x_test)
    print("正规方程-预测房价:\n", y_predict)
    error = mean_squared_error(y_test, y_predict)
    print("正规方程-均方误差为:\n", error)

    return None


def linear_SGDRegressor():
    '''
    梯度下降方程的优化方法对波士顿房价进行预测
    :return:
    '''
    # 1.获取数据
    boston = load_boston()
    # 2.划分数据集
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=22)
    # 3.标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # 4.预估器
    # loss=“squared_loss”默认最小二乘法损失类型，fit_intercept=True是否计算偏置，learning_rate学习率 具体他将eta0带入公式中，eta0=0.01 penalty="l2" l2岭回归
    estimator = SGDRegressor(learning_rate="constant", eta0=0.01, max_iter=10000)
    estimator.fit(x_train, y_train)
    # 5.得出模型
    print("梯度下降-权重系数\n", estimator.coef_)
    print("梯度下降-偏置为\n", estimator.intercept_)
    # 6.模型评估
    y_predict = estimator.predict(x_test)
    print("梯度下降-预测房价:\n", y_predict)
    error = mean_squared_error(y_test, y_predict)
    print("梯度下降-均方误差为:\n", error)
    return None


def linear_ridge():
    '''
    岭回归方程的优化方法对波士顿房价进行预测
    :return:
    '''
    # 1.获取数据
    boston = load_boston()
    # 2.划分数据集
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=22)
    # 3.标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # 4.预估器
    # loss=“squared_loss”默认最小二乘法损失类型，fit_intercept=True是否计算偏置，learning_rate学习率 具体他将eta0带入公式中，eta0=0.01 penalty="l2" l2岭回归
    # estimator = Ridge(alpha=0.05, max_iter=10000)
    # estimator.fit(x_train, y_train)

    # 保存模型
    # joblib.dump(estimator,"./estimator/my_ridge.pkl")
    estimator = joblib.load("estimator/my_ridge.pkl")
    # 5.得出模型
    print("岭回归-权重系数\n", estimator.coef_)
    print("岭回归-偏置为\n", estimator.intercept_)
    # 6.模型评估
    y_predict = estimator.predict(x_test)
    print("岭回归-预测房价:\n", y_predict)
    #均方误差
    error = mean_squared_error(y_test, y_predict)#每个样本的 （预测值-平均值）平方 再求平均值
    print("岭回归-均方误差为:\n", error)
    return None


column_name = ["Sample code number", "Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape",
               "Marginal Adhesion", "Single Epithelial Cell Size",
               "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses", "Class"]

if __name__ == '__main__':
    # 1.代码1：正规方程-优化方法对波士顿房价进行预测
    # linear_LinearRegression()
    # 2.代码2：梯度下降-的优化方法对波士顿房价进行预测
    # linear_SGDRegressor()
    # 3.代码3：岭回归-的优化方法对波士顿房价进行预测
    linear_ridge()
