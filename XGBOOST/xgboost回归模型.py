#!/usr/bin/env python
# coding: utf-8

# 导入需要的各种包
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing, metrics, svm
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb

# 读取数据
data_train = pd.read_csv(r"C:\Users\阿韩想养二哈\Desktop\数模经典数据集\train.csv")

# 数据统计
print(data_train['Survived'].describe())
# 画图
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

sns.distplot(data_train['Survived'], color="r")
plt.title("Survived直方图")
plt.show()

print("Skewness: %f" % data_train['Survived'].skew())
print("Kurtosis: %f" % data_train['Survived'].kurt())

var = 'Pclass'
data = pd.concat([data_train['Survived'], data_train[var]], axis=1)
fig = sns.boxplot(x=var, y="Survived", data=data)
fig.axis(ymin=0, ymax=800000)
plt.title("Pclass 总体评价箱型图")
plt.show()

var = 'Age'
data = pd.concat([data_train['Survived'], data_train[var]], axis=1)
data.plot.scatter(x=var, y="Survived", ylim=(0, 800000))
plt.title("Age 散点图")
plt.show()
# 相关性分析
df_tmp1 = data_train[
    ['Pclass','Age','SibSp','Parch','Fare','Survived']]
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
sns.heatmap(df_tmp1.corr(), cmap="YlGnBu", annot=True)
plt.title("相关性分析图")
plt.show()
# 特征工程
cols = ['Pclass','Age','SibSp','Parch','Fare']
x = data_train[cols].values
y = data_train['Survived'].values
X_train, X_validation, y_train, y_validation = train_test_split(x, y, test_size=0.2, random_state=42)
ss_X = preprocessing.StandardScaler()# 标准化处理
ss_Y = preprocessing.StandardScaler()
X_train_scaled = ss_X.fit_transform(X_train)
y_train_scaled = ss_Y.fit_transform(y_train.reshape(-1, 1))
print(X_train_scaled)
X_validation_scaled = ss_X.transform(X_validation)
y_validation_scaled = ss_Y.transform(y_validation.reshape(-1, 1))

import optuna
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


# 定义贝叶斯优化目标函数
def objective(trial):
    param = {
        'learning_rate': trial.suggest_uniform('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 800),
        'max_depth': trial.suggest_int('max_depth', 2, 5),
        'reg_alpha': trial.suggest_uniform('reg_alpha', 0.00001, 5),
        'reg_lambda': trial.suggest_uniform('reg_lambda', 0.00001, 5),
        'min_child_weight': trial.suggest_uniform('min_child_weight', 1, 100),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
        'colsample_bynode': trial.suggest_uniform('colsample_bynode', 0.6, 1.0),
        'colsample_bylevel': trial.suggest_uniform('colsample_bylevel', 0.6, 1.0),
        'gamma': trial.suggest_uniform('gamma', 0.0, 0.3),
        'scale_pos_weight': trial.suggest_uniform('scale_pos_weight', 1.0, 1.5)
    }

    # 使用贝叶斯优化参数创建XGBoost模型
    xgb_model = XGBRegressor(**param, nthread=10, n_jobs=10, objective='reg:squarederror', booster='gbtree',
                             random_state=7)

    # 在训练集上进行交叉验证
    xgb_train = xgb.DMatrix(X_train, y_train)
    result_train = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
    result_test = cross_val_score(xgb_model, X_validation, y_validation, cv=5, scoring='neg_root_mean_squared_error')
    return -result_train.mean()-result_test.mean()  # 可以根据需要调整优化目标，例如只考虑训练误差或测试误差等


# 设置贝叶斯优化参数和样本数
study = optuna.create_study()
study.optimize(objective, n_trials=100)  # 根据需要调整样本数

# 分析结果
print("Best parameters:", study.best_params)
print("Best value:", study.best_value)

# 训练最优模型
best_param = study.best_params
best_model = XGBRegressor(**best_param, nthread=10, n_jobs=10, objective='reg:squarederror', booster='gbtree', random_state=7)
best_model.fit(X_train, y_train)

# 拟合
best_model.fit(X_train_scaled, y_train_scaled)
y_validation_pred = best_model.predict(X_validation_scaled) # 预测

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
importances = list(best_model.feature_importances_)
data_tmp=data_train.drop(columns='Survived')
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
cols = ['Pclass','Age','SibSp','Parch','Fare']

data_test = pd.read_csv(r"C:\Users\阿韩想养二哈\Desktop\数模经典数据集\test.csv")
print(data_test[cols].isnull().sum())
'''
mean_GarageCars = data_test['B'].mean()
mean_TotalBsmtSF = data_test['C'].mean()
data_test['B'].fillna(mean_GarageCars, inplace=True)
data_test['C'].fillna(mean_TotalBsmtSF, inplace=True)
'''

x_test = data_test.values
x_test_scaled = ss_X.transform(x_test)
y_test_pred = best_model.predict(x_test_scaled)
y_test_pred = y_test_pred.reshape(-1, 1)
data_test['Survived_Pred'] = ss_Y.inverse_transform(y_test_pred)
# data_test.to_excel('test_pred.xlsx')
print('Predict:')
print(data_test)


