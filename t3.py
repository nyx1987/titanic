import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
train=pd.read_csv("/Users/wwwroot/python/titanic/data/train.csv")
test=pd.read_csv("/Users/wwwroot/python/titanic/data/test.csv")
PassengerId=test['PassengerId']
#print(test.info())
#print(train.describe())
print(train.head(8))



