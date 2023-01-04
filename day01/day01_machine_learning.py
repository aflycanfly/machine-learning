# -- coding: utf-8 --
# @Time : 16/10/2021 下午 9:39
# @Author : wkq

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import jieba
import pandas as pd

'''
1）获取数据
2）数据处理
3）特征工程
4）机器学习算法训练 - 模型
5）模型评估
6）应用
'''

'''sklearn特征集'''


def dataset_demo():
    """
    sklearn数据集使用

    1.sklearn.datasets
        load* 获取小规模数据集 已经保存到了本地
        fetch_* 获取大规模数据集 还需要从网上下载这个数据集
        数据集是以字典的方式返回的，所以调用数据的时候可以有两种调用方式。 1)dict["key"] = values

    2.dict.key = values
        学习任何一个深度学习视频都知道，有了数据，需要将数据分成训练集和测试集。
        sklearn中使用train_split将数据集分开： sklearn.model_selection.train_split(array, *options)
        x:数据集的特征值
        y：数据集的标签值
        test_size 测试集的大小，一般为float
        random_state 随机数种子
        return 训练集特征值（x_train），测试纸的特征值（x_test），训练集的目标值（y_train），测试集的目标值（y_test）

    """
    # 获取数据集
    iris = load_iris()
    # print(iris.data)
    # print(iris.target.shape)
    # print(iris.feature_names)
    # print(iris.target_names)

    # print("鸢尾花数据集:\n", iris)
    print("查看特征值:\n", iris.data, iris.data.shape)
    print("查看数据集描述:\n", iris["DESCR"])
    print("查看特征值名字:\n", iris.feature_names)
    print("查看目标值名字:\n", iris.target_names)
    print("查看目标值:\n", iris.target)

    # 数据集划分 传入训练集 目标值 test_size默认是0.25 测试集总数的数目是总数的0.2  random_state随机数种子,不同的种子会造成不同的随机采样结果
    # 返回值顺序 训练集特征值 测试集特征值  训练集目标值 测试集目标值
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)
    print("训练集的特征值：\n", x_train, x_train.shape)
    print("训练集的目标值：\n", y_train, y_train.shape)
    return None


'''特征抽取 sklearn.feature_extraction'''


# DictVectorizer（）
def dict_demo():
    '''
    字典特征抽取
        1.实例化sklearn功能函数
        2.调用fit_transform（数据的根据实例化函数的功能，对数据进行响应的处理。）
        3.print
    :return:
    '''
    data = [{'city': '北京', 'temperature': 100}, {'city': '上海', 'temperature': 60}, {'city': '深圳', 'temperature': 30}]
    print(data)
    # 1.实例化一个转换器类
    # sparse默认False 则返回稀疏矩阵，将非零值 按位置表示出来。可以节省内存
    transfer = DictVectorizer(sparse=False)
    # 2.调用fit_transform()
    data_new = transfer.fit_transform(data)
    print("data_new\n", data_new)
    print("特征名字：\n", transfer.get_feature_names_out())
    return None


def dect_demo2():
    data = [{'pclass': '1st', 'age': 29.0, 'sex': 'female'},
            {'pclass': '1st', 'age': 2.0, 'sex': 'female'},
            {'pclass': '1st', 'age': 30.0, 'sex': 'male'},
            {'pclass': '1st', 'age': 25.0, 'sex': 'female'},
            {'pclass': '1st', 'age': 0.9167, 'sex': 'male'},
            {'pclass': '1st', 'age': 47.0, 'sex': 'male'},
            {'pclass': '1st', 'age': 63.0, 'sex': 'female'},
            {'pclass': '1st', 'age': 39.0, 'sex': 'male'},
            {'pclass': '1st', 'age': 58.0, 'sex': 'female'},
            {'pclass': '1st', 'age': 71.0, 'sex': 'male'}]
    # 1.实例化一个转换器类
    transfer = DictVectorizer()
    # 2.调用fit_transform()
    data_new = transfer.fit_transform(data)
    print(data_new[:2], transfer.get_feature_names_out())

# CountVectorizer（） 统计出现次数
def count_english_demo():
    '''
    文本特征抽取：CountVecotrizer()  统计每个样本文字出现的个数
    :return:
    '''
    data = ["life is short,i like like python", "life is too long,i dislike python"]
    # 1.实例化一个转换器类 stop_words停用词
    transfer = CountVectorizer(stop_words=["is", "too"])
    # 2.调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new.toarray())
    print("特征名字：\n", transfer.get_feature_names_out())
    return None


def count_chinese_demo1():
    '''
      文本特征抽取：CountVecotrizer()  统计每个样本文字出现的个数
      :return:
      '''
    data = ["我 爱 北京 天安门", "天安门 上 太阳 升"]
    # 1.实例化一个转换器类
    transfer = CountVectorizer()
    # 2.调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new.toarray())
    print("特征名字：\n", transfer.get_feature_names_out())
    return None


def cut_word(text):
    '''
    进行中文分词：“我爱北京天安门”-->“我 爱 北京 天安门”
    :param text:
    :return:
    '''
    return " ".join(list(jieba.cut(text)))


def count_chinese_demo2():
    '''
      文本特征抽取：CountVecotrizer()  统计每个样本文字出现的个数
      :return:
      '''
    # 1.将中文文本进行分词
    data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
            "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]
    data_new = []
    for sentence in data:
        data_new.append(cut_word(sentence))
    print("分词后的新数据\n",data_new)
    # 2.实例化一个转换器类
    transfer = CountVectorizer(stop_words=["一种"])
    # 3.调用fit_transform
    data_final = transfer.fit_transform(data_new)
    print("data_final:\n", data_final.toarray())
    print("特征名字：\n", transfer.get_feature_names_out())
    return None


# TfidfVectorizer 根据重要性程度进行抽取
def tdidf_demo():
    '''
    用TF-IDF的方法进行文本特征抽取
    :return:
    '''
    # 1.将中文文本进行分词
    data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
            "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]
    data_new = []
    for sentence in data:
        data_new.append(cut_word(sentence))
    # 2.实例化一个转换器类
    transfer = TfidfVectorizer(stop_words=["一种"])
    # 3.调用fit_transform
    data_final = transfer.fit_transform(data_new)
    print("data_final:\n", data_final.toarray())
    print("特征名字：\n", transfer.get_feature_names_out())
    return None


'''特征预处理 进行归一化/标准化 sklearn.preprocessing'''


# MinMaxScaler（） 归一化
def minmax_demo():
    '''
    归一化
    :return:
    '''
    # 1.获取数据
    data = pd.read_csv("../../../resource/dating.txt")
    data = data.iloc[:, :3]
    print("data:\n", data)
    # 2.实例化一个转换器类 feature_range=[2,3]生成的值在2-3之间 默认不设置在0-1之间
    transfer = MinMaxScaler(feature_range=[2, 3])
    # 3.代用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new)
    return None


# StandardScaler 标准化
def stand_demo():
    '''
    标准化
    :return:
    '''
    # 1.获取数据
    data = pd.read_csv("../../../resource/dating.txt")
    data = data.iloc[:, :3]
    print("data:\n", data)
    # 2.实例化一个转换器类
    transfer = StandardScaler()
    # 3.代用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new)

    return None


'''特征降维 特征选择/主成分分析'''


# VarianceThreshold() 过滤式  删除低方差特征：删除相性大的特征 相关系数（-1,1）
def variance_demo():
    '''
    过滤低方差特征
    :return:
    '''
    # 1.获取数据
    data = pd.read_csv("../../../resource/factor_returns.csv")
    data = data.iloc[:, 1:-2]
    print("data:\n", data, "\n", data.shape)
    # 2.实例化一个转换器类   训练集差异低于threshold的特征将被删除。默认值是保留所有非零方差特征，即删除所有样本中具有相同值的特征。
    transfer = VarianceThreshold(threshold=10)
    # 3.调用fit_transform()
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new, "\n", data_new.shape)
    # 计算某两个变量之间的相关系数
    factor = []
    for title in data:
        factor.append(title)
    for i in range(len(factor)):
        for j in range(i, len(factor) - 1):
            if i != j:
                r = pearsonr(data[factor[i]], data[factor[j]])
                print("%s与%s的相关性是%f" % (factor[i], factor[j], r[0]))

    r1 = pearsonr(data["pe_ratio"], data["pb_ratio"])
    print("pe_ratio与pb_ratio的相关性是\n", r1)
    # r2 = pearsonr(data["revenue"], data["total_expense"])
    # print("revenue与total_expense的相关性是\n",r2)

    return None


# 主成分分析 PCA降维过程中尽可能保留比较多的信息
def pca_demo():
    '''
    pca降维
    :return:
    '''
    # 1、获取数据
    data = [[2, 8, 4, 5], [6, 3, 0, 8], [5, 4, 9, 1]]
    # 2.实例化一个转换器 n_components=0.95降维保留95%的数据，n_components=2 降为2维
    transform = PCA(n_components=0.95)
    # 3.调用fit_transform
    data_new = transform.fit_transform(data)

    print("data_new:\n", data_new)
    return None


if __name__ == '__main__':
    # 代码1：sklearn数据集使用 load_iris() train_test_split
    dataset_demo()
    # 代码2：字典特征抽取 DictVectorizer
    # dict_demo()
    # dect_demo2()
    # 代码3：英文文本特征抽取：CountVecotrizer()
    # count_english_demo()
    # 代码4：中文文本特征抽取：CountVecotrizer() 无自动分词
    # count_chinese_demo1()
    # 代码5：中文文本特征抽取：CountVecotrizer() 自动分词
    # count_chinese_demo2()
    # 代码6：中文分词
    # print(cut_word("我爱北京天安门"))
    # 代码7：用TF-IDF的方法进行文本特征抽取
    tdidf_demo()
    # 代码8：归一化 MinMaxScaler（）
    # minmax_demo()
    # 代码9：标准化 StandardScaler（）
    # stand_demo()
    # 代码10：过滤低方差特征
    # variance_demo()
    # pca降维
    # pca_demo()







