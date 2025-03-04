import cuml
from cuml.preprocessing import StandardScaler
from cuml.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import KFold, cross_val_score
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import cupy as cp
import cudf
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.linear_model import ElasticNetCV
from cuml.linear_model import ElasticNet

from .Prediction import SklearnLinearRegressionModel, SklearnRegressionModel, RegressionModel

# class CuMLRegressionModel(RegressionModel):
#     hyper_params = None
#
#     def __init__(self, estimator, moldata, target_property, features, scaler=True):
#         super().__init__(
#             estimator,
#             moldata,
#             target_property,
#             features,
#             StandardScaler() if scaler else None,
#         )
#
#     def get_id(self):
#         id = self.get_id_base()
#
#         params = self.get_params()
#         if len(params) > 0:
#             id += ':'
#             for key in sorted(params.keys()):
#                 id += f"{key}={params[key]} "
#
#         return f"{id.rstrip()}:{self.Status.to_string(self.status, self.selection_threshold)}"
#
#     def get_params(self):
#         if self.hyper_params is None:
#             return self.estimator.get_params()
#
#         params = self.estimator.get_params()
#         for p in self.hyper_params:
#             self.params[p] = params[p]
#
#         return self.params
#
#     def set_params(self, **kwargs):
#         self.estimator.set_params(**kwargs)
#
#     def predict_single_val(self, vector):
#         if self.status < self.Status.FIT:
#             raise ValueError('Model fitting is not yet done')
#
#         data = cp.array(vector).reshape(1, -1)
#
#         if self.scaler is not None:
#             scale_data = self.scaler.transform(data.astype(float))
#         else:
#             scale_data = data
#
#         estimate = self.estimator.predict(scale_data)
#         return estimate[0]
#
#     def predict_val(self, data):
#         if self.status < self.Status.FIT:
#             raise ValueError('Model fitting is not yet done')
#
#         if self.scaler is not None:
#             scale_data = self.scaler.transform(data.astype(float))
#         else:
#             scale_data = data
#
#         estimate = self.estimator.predict(scale_data)
#         return estimate
#
#     def fit(self, data=None, verbose=2):
#         self.cross_validation(data=data, verbose=verbose)
#
#     def cross_validation(self, data=None, n_splits=3, shuffle=True, verbose=2):
#         if data is None:
#             data = self.get_data()
#
#         target = self.get_target()
#         if self.scaler is not None:
#             self.scaler.fit(data)
#             scale_data = self.scaler.transform(data)
#         else:
#             scale_data = data
#
#         if verbose:
#             print(f"Cross-validation: model={self.get_id_base()} n_splits={n_splits}")
#
#         kf = KFold(n_splits=n_splits, shuffle=shuffle)
#
#         self.cv_score = cross_val_score(self.estimator, scale_data, target, cv=kf, scoring='r2')
#         self.estimator.fit(scale_data, target)
#         self.score = self.estimator.score(scale_data, target)
#
#         estimate = self.estimator.predict(scale_data)
#
#         target_np = target.get()
#
#         self.set_prediction_std(target_np, estimate)
#         self.set_mse(target_np, estimate)
#         self.status = self.Status.FIT
#         self.selection_mask = None
#
#     def param_optimization(self, data=None, param_grid=None, n_splits=3, shuffle=True, verbose=2):
#         if data is None:
#             data = self.get_data()
#
#         target = self.get_target()
#         if self.scaler is not None:
#             self.scaler.fit(data)
#             scale_data = self.scaler.transform(data)
#         else:
#             scale_data = data
#
#         if verbose:
#             print(f"Hyperparameter optimisation: model={self.get_id_base()} n_splits={n_splits}")
#
#         if param_grid is None:
#             param_grid = self.param_grid
#
#         search = GridSearchCV(self.estimator, param_grid, cv=KFold(n_splits=n_splits, shuffle=shuffle))
#         search.fit(scale_data, target)
#
#         self.estimator.set_params(**search.best_params_)
#         self.cross_validation(data=data, n_splits=n_splits, shuffle=shuffle, verbose=verbose)
#
#     def get_data(self):
#         df = cudf.DataFrame.from_pandas(self.moldata.get_feature_vector(self.features.id)).astype(float)
#         return df.values[self.target_mask]
#
#
#     def get_target(self):
#         target_df = cudf.DataFrame.from_pandas(self.moldata.get_property_vector())
#         target_df[self.target_property] = target_df[self.target_property].astype(float)
#         target = target_df[self.target_property].values
#         return target[self.target_mask]

class CuMLElasticNetRegressionModel(SklearnLinearRegressionModel):
    hyper_params = ['alpha', 'l1_ratio']

    param_grid = {'alpha': np.logspace(-6, 2, 9), 'l1_ratio': np.linspace(0.0, 1.0, 6)}

    def __init__(self, moldata, target_property, features, alpha=1.0, l1_ratio=0.5,
                 scaler=True, **kwargs):
        super().__init__(ElasticNet(alpha=alpha, l1_ratio=l1_ratio, **kwargs),
                         moldata, target_property, features, scaler=scaler)

    def search_optimized_parameters(self, data, target, param_grid, n_splits, shuffle, verbose):
        kf = RepeatedKFold(n_splits=n_splits)
        params = self.estimator.get_params()
        search = ElasticNetCV(l1_ratio=param_grid['l1_ratio'], alphas=param_grid['alpha'], cv=kf,
                              fit_intercept=params['fit_intercept'],
                              verbose=verbose)
        search.fit(data, target)
        self.estimator.set_params(alpha=search.alpha_, l1_ratio=search.l1_ratio_)

