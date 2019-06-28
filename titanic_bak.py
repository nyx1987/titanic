#encoding=utf-8
import numpy as np
import pandas
import matplotlib.pyplot as plt
titanic=pandas.read_csv("/Users/wwwroot/python/titanic/data/train.csv")
test=pandas.read_csv("/Users/wwwroot/python/titanic/data/test.csv")
PassengerId=test['PassengerId']

#sex是字符串，无法进行计算，将它转成数字，用0代表man，1代表female
titanic.loc[titanic["Sex"]=="male","Sex"] = 0
titanic.loc[titanic["Sex"]=="female","Sex"] = 1

#登船的地点也是字符串，需要变换成数字,并填充缺失值
titanic["Embarked"] = titanic["Embarked"].fillna('S')
titanic.loc[titanic["Embarked"]=="S","Embarked"] = 0
titanic.loc[titanic["Embarked"]=="C","Embarked"] = 1
titanic.loc[titanic["Embarked"]=="Q","Embarked"] = 2
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].mean())
#使用回归算法(二分类)进行预测
#线性回归
from sklearn.linear_model import LinearRegression
#交叉验证:将训练数据集分成3份，对这三份进行交叉验证，比如使用1，2样本测试，3号样本验证
#对最后得到得数据取平均值
#from sklearn.cross_validation import KFold
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
#选中一些特征 
predictors = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]
alg = LinearRegression()
#n_folds代表将数据切分成3份，存在3层的交叉验证，titanic.shape[0]代表样本个数
#kf = train_test_split(titanic.shape[0],n_folds=3,random_state=1)
#kf = KFold(titanic.shape[0],n_folds=3,random_state=1)
#kf = KFold(n_splits=3,random_state=1)
kf = KFold(n_splits=3)
predictions = []
for train,test in kf.split(titanic):
	#iloc通过行号获取数据
	train_predictors = titanic[predictors].iloc[train,:]
	#获取对应的label值
	train_target = titanic["Survived"].iloc[train]
	#进行训练
	alg.fit(train_predictors,train_target)
	#进行预测
	test_predictors = alg.predict(titanic[predictors].iloc[test,:])
	#将结果加入到list中
	predictions.append(test_predictors)
#	print("\n")

import numpy as np
predictions = np.concatenate(predictions,axis=0)
#将0到1之间的区间值，变成具体的是否被获救，1代表被获救
predictions[predictions>.5] = 1
predictions[predictions<=.5]= 0
predictions.dtype = "float64"
titanic["Survived"] = titanic["Survived"].astype(float)
print("测试数据总数量:",len(predictions))
print("正确的数量:",sum(predictions == titanic["Survived"]))
#print("正确的数:",sum(predictions[predictions == titanic["Survived"]]))
#accuracy = sum(predictions[predictions == titanic["Survived"]])/len(predictions)
num1=sum(predictions == titanic["Survived"])
num2=len(predictions)
accuracy = sum(predictions == titanic["Survived"])/float(len(predictions))
print("准确率:",accuracy)
exit()
accuracy = sum(predictions[predictions == titanic["Survived"]])/len(predictions)
print predictions
