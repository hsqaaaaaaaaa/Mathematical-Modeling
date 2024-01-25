# -*-coding:utf-8 -*-
import os
import json
import pickle
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_absolute_error, roc_auc_score, accuracy_score, confusion_matrix


class GBDTModel(object):
    def __init__(self, output_dir, n_thread=10, n_jobs=10, random_state=7):
        self.best_model = None
        self.best_param = None
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.n_thread = n_thread
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.clf_dict = {
            'lgb': lgb.LGBMClassifier,
            'xgb': xgb.XGBClassifier
        }
        self.params_dict = {
            'lgb': {
                'learning_rate': (0.03, 0.2),
                'n_estimators': (100, 1000),
                'num_leaves': (20, 100),
                'max_depth': (2, 5),
                'max_bin': (30, 255),
                'min_data_in_bin': (30, 256),
                'min_data_in_leaf': (30, 500),
                'min_split_gain': (0.1, 10),
                'min_child_weight': (0.1, 100),
                'min_child_samples': (30, 256),
                'colsample_bytree': (0.6, 1.0),
                'reg_alpha': (0.3, 10),
                'reg_lambda': (0.3, 10),
                'subsample': (0.6, 1.0),
                'subsample_for_bin': (200000, 400000),
                'scale_pos_weight': (1.0, 1.5),
            },
            'xgb': {
                'learning_rate': (0.01, 0.3),
                'n_estimators': (100, 800),
                'max_depth': (2, 5),
                'reg_alpha': (0.00001, 5),
                'reg_lambda': (0.00001, 5),
                'min_child_weight': (1, 100),
                'subsample': (0.6, 1.0),
                'colsample_bytree': (0.6, 1.0),
                'colsample_bynode': (0.6, 1.0),
                'colsample_bylevel': (0.6, 1.0),
                'gamma': (0.0, 0.3),
                'scale_pos_weight': (1.0, 1.5)
            }
        }
        self.int_params = [
            'max_depth', 'n_estimators', 'num_leaves', 'max_bin', 'scale_pos_weight',
            'min_child_samples', 'min_data_in_bin', 'min_data_in_leaf', 'subsample_for_bin','objective']

    def save_model(self, model_name="model.pkl"):
        file = os.path.join(self.output_dir, model_name)
        with open(file, 'wb') as f:
            pickle.dump(self.best_model, f)

    def read_model(self, model_name="model.pkl"):
        file = os.path.join(self.output_dir, model_name)
        with open(file, 'rb') as f:
            self.best_model = pickle.load(f)

    def save_best_param(self, file_name="params.json"):
        file = os.path.join(self.output_dir, file_name)
        with open(file, mode="w", encoding="utf-8") as fw:
            json.dump(self.best_param, fw, indent=4, ensure_ascii=False)

    def read_best_param(self, file_name="params.json"):
        file = os.path.join(self.output_dir, file_name)
        with open(file=file, mode="r", encoding="utf-8") as fr:
            self.best_param = json.load(fr)

    def bayes_optimization(self, train_x, train_y, test_x, test_y, oot_x=[], oot_y=[],
                           clf_name='lgb', target='acc', scale_pos_use=True, **bayes_args):
        res_list = []
        n_thread = self.n_thread
        n_jobs = self.n_jobs
        random_state = self.random_state
        int_params = self.int_params
        xgb_train = xgb.DMatrix(X_train, y_train)
        xgb_test = xgb.DMatrix(X_test, y_test)

        if scale_pos_use is True:
            scale_pos_weight = round(np.sum(train_y == 0) / np.sum(train_y == 1), 1)
            if 'scale_pos_weight' in self.params_dict[clf_name].keys():
                self.params_dict[clf_name]['scale_pos_weight'] = \
                    tuple([x * scale_pos_weight for x in list(self.params_dict[clf_name]['scale_pos_weight'])])
            else:
                self.params_dict[clf_name]['scale_pos_weight'] = scale_pos_weight
        else:
            self.params_dict[clf_name]['scale_pos_weight'] = 1

        print("[bayes_optimization] scale_pos_weight: {}".format(
            self.params_dict[clf_name]['scale_pos_weight']), flush=True)

        def _eval(**clf_args):
            for param in int_params:
                if clf_args.get(param):
                    clf_args[param] = int(round(clf_args[param]))

            print("[eval] args: {}".format(clf_args))

            if clf_name == 'lgb':
                model = lgb.LGBMClassifier(**clf_args,
                                           n_jobs=n_jobs,
                                           random_state=random_state)
            else:
                model = xgb.XGBClassifier(**clf_args,
                                          nthread=n_thread,
                                          n_jobs=n_jobs,
                                          objective='binary:logistic',
                                          random_state=random_state)

            model.fit(train_x, train_y)
            # 绘制变量重要性
            #xgb.plot_importance(model, height=0.5, importance_type='gain', max_num_features=10)
            #plt.show()

            # 交叉验证
            result = xgb.cv(params=clf_args, dtrain=xgb_train, nfold=10, metrics='rmse',  # 'auc'
                            num_boost_round=300, as_pandas=True, seed=123)
            print(result.head())

            # Plot CV Errors
            plt.plot(range(1, 301), result['train-rmse-mean'], 'k', label='Training Error')
            plt.plot(range(1, 301), result['test-rmse-mean'], 'b', label='Test Error')
            plt.xlabel('Number of Trees')
            plt.ylabel('RMSE')
            plt.axhline(0, linestyle='--', color='k', linewidth=1)
            plt.legend()
            plt.title('CV Errors for XGBoost')
            plt.show()

            train_prob = model.predict_proba(train_x)[:, 1]
            test_prob = model.predict_proba(test_x)[:, 1]
            #oot_prob = model.predict_proba(oot_x)[:, 1]

            train_acc = round(accuracy_score(train_y, train_prob > 0.5), 5)
            train_auc = round(roc_auc_score(train_y, train_prob), 5)

            test_acc = round(accuracy_score(test_y, test_prob > 0.5), 5)
            test_auc = round(roc_auc_score(test_y, test_prob), 5)

            #oot_acc = round(accuracy_score(oot_y, oot_prob > 0.5), 5)
            #oot_auc = round(roc_auc_score(oot_y, oot_prob), 5)

            res_list.append({
                'train_acc': train_acc, 'train_auc': train_auc,
                'test_acc': test_acc, 'test_auc': test_auc,
                #'oot_acc': oot_acc, 'oot_auc': oot_auc
            })

            if target == 'auc':
                target_value = test_auc
            else:
                target_value = test_acc

            print('[train_acc] {}, [train_auc] {}, [test_acc] {}, [test_auc] {}'
                  #',[oot_acc] {}, [oot_auc] {}'
            .format(
                train_acc, train_auc, test_acc, test_auc
            #,oot_acc, oot_auc
            ), flush=True)

            return target_value

        print("[bayes_optimization] {}".format(self.params_dict[clf_name]), flush=True)
        clf_bo = BayesianOptimization(f=_eval, pbounds=self.params_dict[clf_name])
        clf_bo.maximize(**bayes_args)

        self.best_param = clf_bo.max['params']

        for param in int_params:
            if self.params_dict[clf_name].get(param):
                self.best_param[param] = int(round(self.best_param[param]))

        self.best_param['nthread'] = n_thread
        self.best_param['n_jobs'] = n_jobs
        self.best_param['random_state'] = random_state
        self.best_param['objective'] = 'binary:logistic'

        res_bo = []
        for id, bo in enumerate(clf_bo.res):
            for param in int_params:
                if bo['params'].get(param):
                    bo['params'][param] = int(round(bo['params'][param]))
            res_bo.append([bo.update(res_list[id])])

        return clf_bo, self.best_param, res_bo

    def eval(self, x, y):
        # 使用模型对测试集数据进行预测
        predictions = self.best_model.predict_proba(x)[:, 1]
        print("[MAE] {}".format(mean_absolute_error(y_true=y, y_pred=predictions)), flush=True)
        print("[ACC] {}".format(accuracy_score(y_true=y, y_pred=predictions > 0.5), flush=True))
        print("[AUC] {}".format(roc_auc_score(y_true=y, y_score=predictions)), flush=True)
        print("[confusion_matrix] \n{} ".format(confusion_matrix(y_true=y, y_pred=predictions > 0.5)), flush=True)

        return predictions

    def train(self, clf_name, x, y):
        print("[train] best_param: {}".format(self.best_param), flush=True)
        if clf_name == 'lgb':
            print("[train] use model: LGBMClassifier", flush=True)
            self.best_model = lgb.LGBMClassifier(**self.best_param)
        else:
            print("[train] use model: XGBClassifier", flush=True)
            self.best_model = xgb.XGBClassifier(**self.best_param)

        self.best_model.fit(x, y)
        print("[train] best_model: {}".format(self.best_model), flush=True)

        self.save_model()



import pandas as pd
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv(r"C:\Users\阿韩想养二哈\Desktop\数模经典数据集\PimaIndiansdiabetes.csv") # 输入特征

X = ['A','B','C','D','E','F','G','H']
#print(X)
y = ['I']
#print(y)
#此处为了后续输出混淆矩阵时，用原始数据输出

from sklearn.model_selection import train_test_split
# 将数据分为训练数据和测试数据
X_train, X_test, y_train, y_test = train_test_split(data[X], data[y], test_size=0.3, train_size=0.7, random_state=42)

print(X_train, X_test, y_train, y_test)

use_columns = pd.DataFrame(X)
label_columns = pd.DataFrame(y)

train_manager = GBDTModel(output_dir="./")

#进行贝叶斯自动调参，并保存最优的参数。(数据均用pandas的DataFrame保存)
train_manager.bayes_optimization(
    train_x=X_train, train_y=np.array(y_train),
    test_x=X_test, test_y=y_test,
    clf_name="xgb"
)
train_manager.save_best_param(file_name="params.json")

#读取最优的参数，训练模型与评估
print('result:')

train_manager.read_best_param(file_name="params.json")
train_manager.train(clf_name="xgb",
                    x=X_train, y=y_train)

train_manager.eval(x=X_test, y=y_test)
#train_manager.eval(x=val_data[use_columns], y=val_data[label_columns])


