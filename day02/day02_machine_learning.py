# -- coding: utf-8 --
# @Time : 19/10/2021 上午 11:26
# @Author : wkq
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier,export_graphviz


def knn_iris():
    '''
    用KNN算法对鸢尾花进行分类
    :return:
    '''
    # 1.获取数据
    iris = load_iris()
    # 2.划分数据集
    # 返回 训练集特征值 测试集特征值  训练集目标值 测试集目标值
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=22, test_size=0.2)
    # 3.特征工程标准化  此数据不用做降维处理
    transfer = StandardScaler()
    # fit() 计算 每一列的平均值、标准差
    # transform() (x-mean)/std进行最终的转换
    x_train = transfer.fit_transform(x_train)
    # 对测试集进行标准化，训练集调用fit_transfor(),测试集调用transform用训练集的平均值、标准差带入公示进行最终的转换
    x_test = transfer.transform(x_test)
    # 4.KNN算法预估器
    estimator = KNeighborsClassifier(n_neighbors=2)
    estimator.fit(x_train, y_train)
    # 5.模型评估
    # 方法1：直接对比真实值和与测试
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("y_test:\n", y_test)
    print("直接对比真实值和预测值:\n", y_predict == y_test)
    # 方法2：计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为:\n", score)
    return None


def knn_iris_gscn():
    '''
    用KNN算法对鸢尾花进行分类，添加网格搜索和交叉验证
    :return:
    '''
    # 1.获取数据
    iris = load_iris()
    # 2.划分数据集
    # 返回 训练集特征值 测试集特征值  训练集目标值 测试集目标值
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=22, test_size=0.2)
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
    params_dict = {"n_neighbors": [1, 3, 5, 7, 9, 11]}
    # estimator:估计器对象 param_grid：估计器参数(dict){"n_neighbors":[1,9,11]} cv:指定几折交叉验证 fit():输入训练集数据 score()准确率
    estimator = GridSearchCV(estimator, param_grid=params_dict, cv=10)
    estimator.fit(x_train, y_train)
    # 5.模型评估
    # 方法1：直接对比真实值和与测试
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("y_test:\n", y_test)
    print("直接对比真实值和预测值:\n", y_predict == y_test)
    # 方法2：计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为:\n", score)

    # 最佳参数:best_params_
    print("最佳参数\n", estimator.best_params_)
    # 最佳结果：best_score_
    print("最佳结果\n", estimator.best_score_)
    # 最佳估计器：best_estimator_
    print("最佳估计器\n", estimator.best_estimator_)
    # 交叉验证结果：cv_results_
    print("交叉验证结果\n", estimator.cv_results_)
    return None


def nb_news():
    '''
    用朴素贝叶斯算法对新闻进行分类
    :return:
    '''
    # 1.获取数据
    news = fetch_20newsgroups(subset="all")
    # 2.划分数据集
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, train_size=0.25)
    # 3.特征工程 特征抽取-TfidfVectorizer
    transfer = TfidfVectorizer()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # 4.朴素贝叶斯算法预估器
    estimator = MultinomialNB()
    estimator.fit(x_train, y_train)
    # 5.模型评估
        #方法一: 直接对比真实值和预测值
    y_predict = estimator.predict(x_test)
    print("y_predict:\n",y_predict)
    print("直接对比真实值和预测值:\n",y_test  == y_predict)
        #方法二: 计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为：\n",score)

    return None

def decision_iris():
    '''
    用决策树对鸢尾花进行分类

    :return:
    '''
    # 1.获取数据集
    iris = load_iris()
    # 2.划分数据集
    x_train,x_test,y_train,y_test = train_test_split(iris.data, iris.target)
    # 3.决策树预估器
    estimator = DecisionTreeClassifier(criterion="entropy")
    estimator.fit(x_train,y_train)
    # 4.模型评估
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("直接对比真实值和预测值:\n", y_predict == y_test)
    # 方法2：计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为:\n", score)

    export_graphviz(estimator, out_file="decision_tree_ picture/iris_tree.dot", feature_names=iris.feature_names)

    return  None


if __name__ == '__main__':
    # 代码1：用KNN算法对鸢尾花进行分类
    # knn_iris()
    # 代码2：用KNN算法对鸢尾花进行分类
    # knn_iris_gscn()
    # 代码3:用朴素贝叶斯算法对新闻进行分类
    # nb_news()
    # 代码4：用决策树对鸢尾花进行分类
    decision_iris()
