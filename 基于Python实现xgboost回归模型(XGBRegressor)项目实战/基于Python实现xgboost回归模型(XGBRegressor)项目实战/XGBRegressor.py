#!/usr/bin/env python
# coding: utf-8

# 导入需要的各种包
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.stats import norm
from sklearn import preprocessing, metrics, svm
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
from xgboost import plot_importance

# 读取数据
data_train = pd.read_excel("train.xlsx")
# 数据统计
print(data_train['SalePrice'].describe())
# 画图
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
sns.distplot(data_train['SalePrice'], color="r")
plt.title("SalePrice直方图")
plt.show()

print("Skewness: %f" % data_train['SalePrice'].skew())
print("Kurtosis: %f" % data_train['SalePrice'].kurt())

var = 'OverallQual'
data = pd.concat([data_train['SalePrice'], data_train[var]], axis=1)
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)
plt.title("OverallQual 总体评价箱型图")
plt.show()

var = 'YearBuilt'
data = pd.concat([data_train['SalePrice'], data_train[var]], axis=1)
data.plot.scatter(x=var, y="SalePrice", ylim=(0, 800000))
plt.title("YearBuilt 建造年份散点图")
plt.show()
# 相关性分析
df_tmp1 = data_train[
    ['GrLivArea', 'TotRmsAbvGrd', 'FullBath', 'TotalBsmtSF', 'GarageCars', 'YearBuilt', 'OverallQual', 'SalePrice']]
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
sns.heatmap(df_tmp1.corr(), cmap="YlGnBu", annot=True)
plt.title("相关性分析图")
plt.show()
# 特征工程
cols = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']
x = data_train[cols].values
y = data_train['SalePrice'].values
X_train, X_validation, y_train, y_validation = train_test_split(x, y, test_size=0.2, random_state=42)
ss_X = preprocessing.StandardScaler()# 标准化处理
ss_Y = preprocessing.StandardScaler()
X_train_scaled = ss_X.fit_transform(X_train)
y_train_scaled = ss_Y.fit_transform(y_train.reshape(-1, 1))
print(X_train_scaled)
X_validation_scaled = ss_X.transform(X_validation)
y_validation_scaled = ss_Y.transform(y_validation.reshape(-1, 1))
# 建模
xgb_model = xgb.XGBRegressor(max_depth=3,
                             learning_rate=0.1,
                             n_estimators=100,
                             objective='reg:squarederror',
                             booster='gbtree',
                             random_state=0)
# 拟合
xgb_model.fit(X_train_scaled, y_train_scaled)
y_validation_pred = xgb_model.predict(X_validation_scaled) # 预测

'''
# 建立SVR模型
svr_model=svm.SVR()
svr_model.fit(X_train_scaled, y_train_scaled)
y_validation_pred1 = svr_model.predict(X_validation_scaled)
print('SVR可解释方差值：{}'.format(round(metrics.explained_variance_score(y_validation_scaled, y_validation_pred1), 2)))
print('SVR平均绝对误差：{}'.format(round(metrics.mean_absolute_error(y_validation_scaled, y_validation_pred1), 2)))
print('SVR均方误差：{}'.format(round(metrics.mean_squared_error(y_validation_scaled, y_validation_pred1), 2)))
print('SVR R方值：{}'.format(round(metrics.r2_score(y_validation_scaled, y_validation_pred1), 2)))
'''
# 画图
plt.plot(range(y_validation_scaled.shape[0]), y_validation_scaled, color="blue", linewidth=1.5, linestyle="-")
plt.plot(range(y_validation_pred.shape[0]), y_validation_pred, color="red", linewidth=1.5, linestyle="-.")
plt.legend(['真实值', '预测值'])
plt.title("真实值与预测值比对图")
plt.show()  #显示图片
# 模型评估
print('可解释方差值：{}'.format(round(metrics.explained_variance_score(y_validation_scaled, y_validation_pred), 2)))
print('平均绝对误差：{}'.format(round(metrics.mean_absolute_error(y_validation_scaled, y_validation_pred), 2)))
print('均方误差：{}'.format(round(metrics.mean_squared_error(y_validation_scaled, y_validation_pred), 2)))
print('R方值：{}'.format(round(metrics.r2_score(y_validation_scaled, y_validation_pred), 2)))

# 显示重要特征
importances = list(xgb_model.feature_importances_)
data_tmp=data_train.drop(columns='SalePrice')
feature_list = list(data_tmp.columns)

feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

import matplotlib.pyplot as plt

x_values = list(range(len(importances)))
print(x_values)
plt.bar(x_values, importances, orientation='vertical')
plt.xticks(x_values, feature_list, rotation=6)
plt.ylabel('Importance')
plt.xlabel('Variable')
plt.title('Variable Importances')
plt.show()
# 对测试数据进行预测
cols = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']
data_test = pd.read_excel("test.xlsx")
print(data_test[cols].isnull().sum())
mean_GarageCars = data_test['GarageCars'].mean()
mean_TotalBsmtSF = data_test['TotalBsmtSF'].mean()
data_test['GarageCars'].fillna(mean_GarageCars, inplace=True)
data_test['TotalBsmtSF'].fillna(mean_TotalBsmtSF, inplace=True)

x_test = data_test.values
x_test_scaled = ss_X.transform(x_test)
y_test_pred = xgb_model.predict(x_test_scaled)
data_test['SalePrice_Pred']=ss_Y.inverse_transform(y_test_pred)
# data_test.to_excel('test_pred.xlsx')

