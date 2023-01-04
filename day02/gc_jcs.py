# coding: utf-8
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn import svm
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from six import StringIO
import pydotplus

drug = 'grass-category'
fn = 40
n_splits = 3    # 将数据随机切分成三份，每份数据训练集70%，测试集30%，模型准确率为三份数据对应的平均
data = pd.read_csv('../../../resource/data.csv', header=1)
X = data.iloc[:, 2:-1]
y = data.iloc[:, 0]
# random_state=1不变的话，每次得到的数据都是一样的，random_state=None，每次的数据不一样
train_size = 0.6
MAX_ITER = 8

fpr = []
tpr = []

best_auc = 0;
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=2)
y_pred_best = []

import joblib

feature_name = ['xiancaochangliang(gongjin/mu)','gancaochangliang(gongjin/mu)','gancaozongliang(wangongjin)','keliyonggancaochongliang(gongjin/mu)','keliyonggancaozongliang(wangongjin)','zaixuliang(yangdanwei)']
target_name = ['Didicaodianlei','Gaohancaodianlei','Gaohancaoyuanlei', 'Wenxingcaoyuanlei', 'Wenxinghuanmocaoyuanlei', 'Wenxinghuanmolei']

for iter in range(MAX_ITER):
    #model = svm.SVC(C=373.2254, gamma=0.00032, decision_function_shape='ovo', max_iter=iter)  # gamma缺省值为 1.0/x.shape[1]
    #model = svm.SVC(C=760.3932, gamma=0.00019, decision_function_shape='ovo', max_iter=iter)
    #model = svm.SVC(C=0.1, decision_function_shape='ovo', max_iter=iter)
    model = DecisionTreeClassifier(max_depth=iter+1, random_state=0)
    model.fit(x_train, y_train)
    joblib.dump(model, 'gini/rfc.pkl')
    #y_score = model.decision_function(x_test)
    y_pre = model.predict(x_test)
    roc_auc = accuracy_score(y_test, y_pre)
    fpr.append(iter)
    tpr.append(roc_auc)
    if best_auc < roc_auc:  # 更新全局最优
        best_auc = roc_auc
        y_pred_best = y_pre

    print("%d-----auc %0.2f" % (iter, roc_auc))
    dot_data = StringIO()
    tree.export_graphviz(model,out_file = dot_data,feature_names=feature_name,
                     class_names=target_name,filled=True,rounded=True,
                     special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    # graph.write_pdf("./gini/jcs.pdf")
#可视化auc
# plt.figure()
lw = 3
plt.figure(figsize=(10, 8))
plt.rcParams['font.sans-serif']=['SimHei']
# 假正率为横坐标，真正率为纵坐标做曲线
plt.plot(fpr, tpr, color='red', lw=lw)
#plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('number of generations', fontsize=15)
plt.ylabel('准确率', fontsize=15)
plt.title('决策树 (AUC = %0.2f)' % best_auc, fontsize=20)
plt.legend(loc="lower right", fontsize=20)
plt.savefig('./gini/svm.png')
plt.show()

plt.figure(figsize=(10, 8))
t = range(0, len(y_test))
plt.rcParams['font.sans-serif']=['SimHei']
plt.plot(t, y_test, 'bs', t, y_pred_best, 'g^')
plt.xlabel('result')
plt.ylabel('准确率')
plt.title('test（蓝） vs pred（绿）')
plt.savefig('./gini/svm_com.png')
plt.show()



