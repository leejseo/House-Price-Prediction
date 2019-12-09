from sklearn.linear_model import ElasticNet, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.externals import joblib
import xgboost as xgb
import lightgbm as lgb
import time
import numpy as np

from data_processing import *

"""
    reference: www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
"""
class AverageModel(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        i = 0
        for model in self.models_:
            print(i, 'fit...')
            i += 1
            model.fit(X, y)
        return self

    def predict(self, X):
        print('predict...')
        predictions = np.column_stack([
            model.predict(X) for model in self.models_])
        return np.mean(predictions, axis=1)

class StackedModel(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            print(i,'fit...')
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                pred_y = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = pred_y

        new_X = np.concatenate((X[:,SECOND_LAYER_INDEX],out_of_fold_predictions),axis=1)
        self.meta_model_.fit(new_X, y)
        return self

    def predict(self, X):
        print('predict...')
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        new_X = np.concatenate((X[:,SECOND_LAYER_INDEX],meta_features),axis=1)
        return self.meta_model_.predict(new_X)


class Model(object):
    def __init__(self, unique = True, load_dir = None):
        if load_dir:
            self.model = joblib.load(load_dir)
            return

        # base models
        ENet = ElasticNet(alpha=1.0, copy_X=True, fit_intercept=True, l1_ratio=0.5,
                          max_iter=1000, normalize=False, positive=False, precompute=False,
                          random_state=42, selection='cyclic', tol=0.0001, warm_start=False)
        GBoost = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05,
                                           max_depth=4, max_features='sqrt',
                                           min_samples_leaf=15, min_samples_split=10,
                                           loss='huber', random_state=5)
        XGB = xgb.XGBRegressor(colsample_bytree=0.4400, gamma=0.05,
                               learning_rate=0.3, max_depth=4,
                               min_child_weight=2, n_estimators=2200,
                               reg_alpha=0.48, reg_lambda=0.85,
                               subsample=0.5213, silent=True,
                               random_state=7, nthread=-1)
        LGB = lgb.LGBMRegressor(objective='regression', num_leaves=5,
                                learning_rate=0.05, n_estimators=720,
                                max_bin=55, bagging_fraction=0.8,
                                bagging_freq=5, feature_fraction=0.22,
                                feature_fraction_seed=9, bagging_seed=9,
                                min_data_in_leaf=6, min_sum_hessian_in_leaf=11)
        if unique:
            # meta model
            lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=42))
            self.model = StackedModel(base_models=(ENet, GBoost, XGB, LGB), meta_model=lasso)
        else:
            self.model = AverageModel(models = (ENet,GBoost,XGB,LGB))

    def fit(self, X, y):
        return self.model.fit(X,np.log1p(y))

    def predict(self, X):
        return np.exp(self.model.predict(X)) - 1.

    def evaluate(self, test_x, test_y):
        pred_y = self.predict(test_x)
        performance = 1. - np.mean(np.absolute(test_y - pred_y) / test_y)
        print("performance:", performance)

    def save(self, save_dir):
        joblib.dump(self.model, save_dir)


def test_by_train_data(unique=True, using_ratio = 1.0, training_ratio = 0.9):
    data = shuffle(get_train_data())
    tot_sz = int(data.shape[0]*using_ratio)
    train_sz = int(tot_sz*training_ratio)
    data = data[:tot_sz]
    data = data[data[:, 0].argsort()]

    train = data[:train_sz]
    test = data[train_sz:]
    train_x, train_y = train[:,:-1], train[:,-1]
    test_x, test_y = test[:,:-1], test[:,-1]

    model = Model(unique)
    model.fit(train_x, train_y)
    model.evaluate(test_x, test_y)

def train(unique=True):
    if unique:
        filename = 'unique'
    else:
        filename = 'base'

    model = Model(unique)
    train = get_train_data()
    train_x, train_y = train[:,:-1], train[:,-1]

    # fit model
    model.fit(train_x, train_y)
    print('model fitted')

    # save model
    model.save(filename+'.pkl')
    print('model saved')


def test(unique=True):
    if unique:
        filename = 'unique'
    else:
        filename = 'base'

    model = Model(unique,filename+'.pkl')
    test = get_test_data()

    # predict
    res = model.predict(test)

    # save result
    with open(filename+'.csv', 'w') as fout:
        for it in res:
            fout.write(str(it) + '\n')


np.random.seed(seed=42)
if __name__ == '__main__':
    start_time = time.time()
    # ----------------------------
    test_by_train_data(False)
    #train(True)
    #test(True)
    # ----------------------------
    print("--- %s seconds ---" % (time.time() - start_time))
