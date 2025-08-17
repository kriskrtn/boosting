from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor

from typing import Optional

import matplotlib.pyplot as plt

np.random.seed(42)


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
        self,
        base_model_class = DecisionTreeRegressor,
        base_model_params: Optional[dict] = None,
        n_estimators: int = 10,
        learning_rate: float = 0.1,
        early_stopping_rounds: int | None = 0,
        subsample: float | int = 1.0,
        bagging_temperature: float | int = 1.0,
        bootstrap_type: str | None = 'Bernoulli',
        rsm: float | int = 1.0,
        quantization_type: str | None = None,
        nbins: int = 255
    ):
        self.base_model_class = base_model_class
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate

        self.history = defaultdict(list) # {"train_roc_auc": [], "train_loss": [], ...}

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z:  -1 * (1 - self.sigmoid(y * z)) * y  # Исправьте формулу на правильную. 
        self.from_fit = False
        self.early_stopping_rounds = early_stopping_rounds
        self.subsample = subsample
        self.bagging_temperature = bagging_temperature
        self.bootstrap_type = bootstrap_type
        self.rsm = rsm
        self.quantization_type = quantization_type
        self.nbins = nbins
        self.model_features: list = []
        self.feature_importances_ = None

    def features(self, X):
        if isinstance(self.rsm, int):
            proportion = self.rsm
        else:
            proportion = int(self.rsm * X.shape[1])
        features = np.random.choice(X.shape[1], size=proportion, replace=False)
        return features

    def quantization(self, X):
        X_data = X.toarray() 
        if self.quantization_type == 'Uniform':
            X_min = X_data.min(axis=0)
            X_max = X_data.max(axis=0)
            X_new = np.zeros_like(X_data)
            for i in range(X_data.shape[1]):
                X_new[:, i] = np.digitize(X_data[:, i], np.linspace(X_min[i], X_max[i], self.nbins + 1)) - 1
        elif self.quantization_type == 'Quantile':
            X_new = np.zeros_like(X_data)
            for i in range(X.shape[1]):
                percentiles = np.percentile(X_data[:, i], np.linspace(0, 100, self.nbins + 1))
                X_new[:, i] = np.digitize(X_data[:, i], percentiles) - 1
        return X_new
    
    def bootstrap(self, n):
        if self.bootstrap_type == 'Bernoulli':
            if isinstance(self.subsample, float):
                indexes = np.array([i for i in range(n) if np.random.rand() < self.subsample])
            else:
                indexes = np.array([i for i in range(n) if np.random.rand() < (self.subsample / n)])
            if len(indexes) == n:
                return indexes
            elif len(indexes) < n:
                indexes = np.random.choice(indexes, size=n)
                return indexes
        elif self.bootstrap_type == 'Bayesian':
            w = -np.log(np.clip(np.random.rand(n), 1e-10, 1)) ** self.bagging_temperature
            return np.random.choice(n, size = n, p = w / w.sum())

    def partial_fit(self, X, y):
        objects = self.bootstrap(X.shape[0])
        X_new = X[objects]
        y_new = y[objects]
        features = self.features(X_new)
        X_new = X_new[:, features]
        self.model_features.append(features)
        if self.quantization_type:
            X_new = self.quantization(X_new)
        model = self.base_model_class(**self.base_model_params)
        model.fit(X_new, y_new)  
        predictions = model.predict(X_new) 
        self.models.append(model)
        return predictions

    def fit(self, X_train, y_train, X_val=None, y_val=None, plot=False):
        """
        :param X_train: features array (train set)
        :param y_train: targets array (train set)
        :param X_val: features array (eval set)
        :param y_val: targets array (eval set)
        :param plot: bool 
        """
        train_predictions = np.zeros(y_train.shape[0])
        predictions_val = np.zeros(X_val.shape[0]) if X_val != None else None

        best_loss = 1e10
        best_i = 0 

        for _ in range(self.n_estimators):
            if _ == 0:
                predictions = self.partial_fit(X_train, y_train)
                self.gammas.append(1)
                train_predictions = predictions
                if X_val is not None:
                    predictions_val = self.models[-1].predict(X_val[:, self.model_features[-1]])
            else:
                s = -self.loss_derivative(y_train, train_predictions)
                predictions = self.partial_fit(X_train, s)
                best_gamma = self.find_optimal_gamma(y_train, train_predictions, predictions)
                self.gammas.append(best_gamma)
                train_predictions += best_gamma * predictions
                if X_val is not None:
                    predictions_val += best_gamma * self.models[-1].predict(X_val[:, self.model_features[-1]])
            if X_val is not None:
                valid_loss = self.loss_fn(y_val, predictions_val)
                self.history["valid_loss"].append(valid_loss)
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    best_i = 0
                else:
                    best_i += 1
                    if self.early_stopping_rounds and best_i >= self.early_stopping_rounds:
                        break

            train_loss = self.loss_fn(y_train, train_predictions)
            self.history["train_loss"].append(train_loss)
        
        self.feature_importances_ = np.zeros(X_train.shape[1])
        for model in self.models:
            self.feature_importances_ += model.feature_importances_
        self.feature_importances_ /= self.n_estimators


        if plot:
            self.from_fit = True
            self.plot_history(X_train, y_train)

    def predict_proba(self, X):
        predictions = np.zeros(X.shape[0])
        for model, gamma, features in zip(self.models, self.gammas, self.model_features):
            predictions += gamma * model.predict(X[:, features])
        class_1 = self.sigmoid(predictions)
        class_0 = 1 - class_1
        return np.array([class_0, class_1]).T

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]
        return gammas[np.argmin(losses)]

    def score(self, X, y):
        return score(self, X, y)
        
    def plot_history(self, X, y):
        """
        :param X: features array (any set)
        :param y: targets array (any set)
        """
        if self.from_fit == True:
            plt.plot(self.history["train_loss"], label="Ошибка на трейне")
            if "valid_loss" in self.history:
                plt.plot(self.history["valid_loss"], label="Ошибка на валидации")
                plt.xlabel("Итерация")
                plt.ylabel("Ошибка")
                plt.legend()
                plt.title('График ошибки на трейне и валидации')
            self.from_fit = False
        else:
            predictions = np.zeros(y.shape[0])
            losses = []
            for model, gamma in zip(self.models, self.gammas):
                predictions += gamma * model.predict(X)
                loss = self.loss_fn(y, predictions)
                losses.append(loss)
            plt.plot(losses)
            plt.xlabel("Итерация")
            plt.ylabel("Ошибка")
            plt.title('График ошибки')
        plt.show()
