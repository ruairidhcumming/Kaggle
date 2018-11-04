# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 13:15:53 2018

@author: ruair
"""

#import statements 
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from math import sqrt
import statistics as st
import matplotlib.pyplot as plt
#load data
train = pd.read_csv('C:\\Users\\ruair\\Documents\\GitHub\\Kaggle\\Moscow Property/train.csv')
tst =pd.read_csv('C:\\Users\\ruair\\Documents\\GitHub\\Kaggle\\Moscow Property/test.csv')
#process training set 
X=train.drop(['SalePrice'], axis = 1)
X[X.columns.values[X.dtypes == 'object' ]]=X[X.columns.values[X.dtypes == 'object' ]].fillna('none')
y=train['SalePrice']

#process test set (same as X)
tst[tst.columns.values[tst.dtypes == 'object' ]]=tst[tst.columns.values[tst.dtypes == 'object' ]].fillna('none')
#assign skl elements 
reg1 = RandomForestRegressor(max_depth = 2, random_state = 0, n_estimators =1000, verbose = True)
le= preprocessing.LabelEncoder()
#label encode dataframes 
Xenc = X.apply(le.fit_transform)
tstenc = tst.apply(le.fit_transform)
#create subsample for accuract testing
Xtrain,Xtst,Ytrain,Ytst = train_test_split(Xenc,y,test_size = 0.2, random_state = 42)
reg1.fit(Xtrain,Ytrain)
#create results dataframe
Ans =reg1.predict(Xtst)
Xtst['Ans']=Ans
Xtst['SalePrice']=Ytst
Xtst['err']=(Xtst.SalePrice-Xtst.Ans)/Xtst.SalePrice
Xtst['AbsErr']=Xtst.SalePrice-Xtst.Ans

plt.hist(Xtst.err)
plt.scatter(Xtst.AbsErr,Xtst.SalePrice, c = Xtst.MSZoning)
plt.title('Zoneing')
plt.scatter(Xtst.AbsErr,Xtst.LotArea, c = Xtst.Fence)
plt.title('Fence')

