import math

from sklearn.model_selection import train_test_split
from quantile_forest import RandomForestQuantileRegressor
from config.knobs_list import *
from util import *
import numpy as np
import pandas as pd
from optuna.samplers import TPESampler
from optuna.integration import BoTorchSampler
import optuna
import random


class PerformanceModel:
    def __init__(self,
                 core_thresholds=(CORE_MIN, CORE_MAX),
                 memory_thresholds=(MEMORY_MIN, MEMORY_MAX),
                 logger=None,
                 weights=None):
        self.qrf = None
        self.cal_score = None
        self.q_hat = None

        self.core_thresholds = core_thresholds
        self.memory_thresholds = memory_thresholds
        self.logger = logger
        self.task_embedding = None
        self.is_minimum = True
        self.is_zero_history = False

        if weights is not None:
            self.probabilities = [(1 - rate_tradeoff) * weights[0] / sum(weights) + rate_tradeoff / 2,
                                  (1 - rate_tradeoff) * weights[1] / sum(weights) + rate_tradeoff / 2]
        else:
            self.probabilities = [1 - rate_tradeoff, rate_tradeoff]

        self.selected_index = -1

    def train(self, hist_data: pd.DataFrame):
        if len(hist_data) < 200:
            self.is_zero_history = True
            sorted_hist_data = hist_data.sort_values(by="duration", ascending=False)
            cal_size = math.ceil(0.15 * len(hist_data))
            X = sorted_hist_data[KNOBS].values
            y = sorted_hist_data['duration'].values
            X_cal, X_train = X[:cal_size], X[cal_size:]
            y_cal, y_train = y[:cal_size], y[cal_size:]
            self.qrf = RandomForestQuantileRegressor(n_estimators=10, max_depth=3, max_samples_leaf=None, n_jobs=-1)
            self.qrf.fit(X_train, y_train)
        else:
            X = hist_data[KNOBS + embedding_columns].values
            y = hist_data['duration'].values
            X_train, X_cal, y_train, y_cal = train_test_split(X, y, test_size=0.15)
            self.qrf = RandomForestQuantileRegressor(n_jobs=-1)
            self.qrf.fit(X_train, y_train)

        cal_lower, cal_mean, cal_upper = self.predict_ci(X_cal)
        n = len(y_cal)
        self.cal_score = np.maximum(y_cal - cal_lower, cal_upper - y_cal)
        # 防止n ≤ 10时出现分位数 > 1的情况
        self.q_hat = np.quantile(self.cal_score, min(1, np.ceil((n + 1) * (1 - alpha)) / n), method='higher')

        self.logger.info(f"Model train finished using {len(X_train)} records with {len(X_cal)} calibrations.")

    # 预测置信区间和值
    def predict_ci(self, data):
        if data.ndim == 1:
            data = data.reshape(1, -1)
        pred_lower, pred_mean, pred_upper = self.qrf.predict(data, quantiles=[alpha / 2, 0.5, 1 - alpha / 2]).T
        pred_upper = np.maximum(pred_upper, pred_lower + 1e-6)
        return pred_lower, pred_mean, pred_upper

    # 预测值
    def predict(self, data):
        if data.ndim == 1:
            data = data.reshape(1, -1)

        if self.is_zero_history:
            data = data[:, :len(KNOBS)]
        pred_mean = self.qrf.predict(data, quantiles=None, aggregate_leaves_first=False).T
        return pred_mean

    def _try_sample(self, trial, updated_knob_details):
        tmp = list()
        for key in KNOBS:
            details = updated_knob_details[key]
            knob_type = details['type']
            if knob_type == KnobType.CATEGORICAL:
                v = trial.suggest_categorical(key, [_ for _ in range(0, len(details['candidates']))])
            elif knob_type == KnobType.INTEGER:
                min_value, max_value, step_length = details['range'][0: 3]
                v = trial.suggest_int(key, min_value, max_value, step_length)
            elif knob_type == KnobType.NUMERIC:
                min_value, max_value, step_length = details['range'][0: 3]
                v = round(trial.suggest_float(key, min_value, max_value, step=step_length), 3)
            else:
                continue
            tmp.append(v)

        if self.is_zero_history:
            sample = np.array(tmp)
        else:
            sample = np.append(np.array(tmp), self.task_embedding)

        if self.is_minimum:
            out = -self.predict(sample)[0]
        else:
            pred_lower, pred_mean, pred_upper = self.predict_ci(sample)
            sample_score = np.maximum(pred_mean - pred_lower, pred_upper - pred_mean)
            p_value = np.mean(self.cal_score >= sample_score)
            out = (pred_upper + self.q_hat) / (pred_lower - self.q_hat)
            # out = (pred_upper + self.q_hat * (1 - p_value)) / (pred_lower - self.q_hat * (1 - p_value))[0]
        return out

    def resource_constraint(self, trial):
        total_cores, total_memory = get_resource_usage_of_config(trial.params)
        # 可行域是≤0
        c0 = self.core_thresholds[0] - total_cores
        c1 = total_cores - self.core_thresholds[1]
        m1 = self.memory_thresholds[0] - total_memory
        m2 = total_memory - self.memory_thresholds[1]
        return c0, c1, m1, m2

    def search_new_config(self, X_embedding, updated_knob_details):
        self.task_embedding = X_embedding
        ran = random.random()
        if ran < self.probabilities[0]:
            self.selected_index = 0
            self.is_minimum = True
            self.logger.info(f"Sample with probability {self.probabilities}, and generate config using minimum value.")
        else:
            self.selected_index = 1
            self.is_minimum = False
            self.logger.info(f"Sample with probability {self.probabilities}, and generate config using uncertainty.")

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = TPESampler(constraints_func=lambda trial: self.resource_constraint(trial))
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(lambda trial: self._try_sample(trial, updated_knob_details if self.is_minimum else KNOB_DETAILS), n_trials=100)
        return study.best_params
