import math
from typing import Callable, Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline


class PurgedKFold(BaseCrossValidator):
    def __init__(self, t1: pd.Series, n_folds: int, pct_embargo: float = 0.0):
        """
        :param t1: A series whose index is t0, the start date of each observation, and whose values are the
        observation end dates (exclusive).
        :param n_folds: The number of training/test set partitions.
        :param pct_embargo: The percent of observations following the test set to be removed from the training set.
        """
        super().__init__()
        if not isinstance(t1, pd.Series):
            raise TypeError("dates must be a pd.Series of Timestamp")
        if n_folds < 2:
            raise ValueError("n_folds must be >= 2. Found " + str(n_folds))

        self.t1 = t1
        self.n_folds = n_folds
        self.pct_embargo = pct_embargo if pct_embargo is not None else 0.0

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_folds

    def split(self, X, y=None, groups=None):
        for (
            train_indices_pre_split,
            train_indices_post_split,
            test_indices,
        ) in self.split_noconcat(X, y, groups):
            train_indices = np.concatenate(
                (train_indices_pre_split, train_indices_post_split)
            )
            yield train_indices, test_indices

    def split_noconcat(self, X, y=None, groups=None):
        if (X.index == self.t1.index).sum() != len(self.t1):
            raise ValueError("X and t1 must have the same index")

        num_dates = len(self.t1)
        indices = np.arange(num_dates)
        num_embargoed = int(num_dates * self.pct_embargo)
        test_starts = [(i[0], i[-1] + 1) for i in np.array_split(indices, self.n_folds)]

        for i, j in test_starts:
            t0 = self.t1.index[i]  # start of the test set
            test_indices = indices[i:j]
            max_t1_idx = self.t1.index.searchsorted(self.t1[test_indices].max())
            train_indices_pre_split = self.t1.index.searchsorted(
                self.t1[self.t1 <= t0].index
            )
            train_indices_post_split = indices[max_t1_idx + num_embargoed :]
            yield train_indices_pre_split, train_indices_post_split, test_indices

    def _iter_test_indices(self, X=None, y=None, groups=None):
        pass


def train_test_split(
    df: pd.DataFrame, t1: pd.Series, test_size: float
) -> (pd.DataFrame, pd.DataFrame, pd.Series, pd.Series):
    n_samples = len(df)
    n_test = math.ceil(test_size * n_samples)
    test_df = df.iloc[-n_test:, :]
    test_t0 = df.index[-n_test]
    train_df = df.loc[t1 <= test_t0, :]
    return train_df, test_df


class PooledPurgedKFold(PurgedKFold):
    """
    Adapts PurgedKFold to work with pooled time series (with a cross-section across assets or another dimensions).
    Runs the PurgedKFold CV split in date index space, and then selects rows indexes from the higher-dimension data
    frame for those dates.
    """

    def __init__(self, t1: pd.Series, n_folds, pct_embargo=0.0):
        if isinstance(t1.index, pd.MultiIndex):
            dates = t1.index.get_level_values(0)
            name = t1.index.levels[0].name
        else:
            dates = t1.index.values
            name = t1.index.name

        self.dates = pd.Series(dates)
        unique_t1 = t1.reset_index().groupby(name).nth(0)
        self.t0 = pd.Series(unique_t1.t1.index)
        self.t1 = unique_t1.t1
        # print(self.dates.head())
        # print(t1.head())
        # print(self.t0.head())
        # print(self.t1.head())

        super().__init__(self.t1, n_folds, pct_embargo)

    def split_noconcat(self, X, y=None, groups=None):
        def translate_idx(date_idx):
            if len(date_idx) == 0:
                return date_idx
            else:
                date_set = self.t0.iloc[date_idx].values.reshape(-1)
                return self.dates.index.values[self.dates.isin(date_set)]

        super_iter = super().split_noconcat(self.t1, y, groups)
        for train_idx_pre, train_idx_post, test_idx in super_iter:
            yield translate_idx(train_idx_pre), translate_idx(
                train_idx_post
            ), translate_idx(test_idx)


def cv_score(
    # factory: Callable[[], Any],
    clf,
    X: pd.DataFrame,
    Y: pd.Series,
    scoring="neg_mean_squared_error",
    n_folds=None,
    cv_gen=None,
    pct_embargo=0.0,
    has_weights=False,
):
    if isinstance(scoring, str):
        scoring = [scoring]

    for metric in scoring:
        if metric not in [
            "neg_log_loss",
            "accuracy",
            "neg_mean_squared_error",
            "r2_score",
        ]:
            raise Exception("Unexpected scoring method: " + metric)

    if cv_gen is None:
        cv_gen = PooledPurgedKFold(Y.t1, n_folds, pct_embargo)

    score = []
    for (train, test) in cv_gen.split(X=X):
        # clf = factory()
        params = _fit_params_for(clf, X, Y, train, has_weights=has_weights)
        clf.fit(**params)
        X_test = X.iloc[test, :]
        y_test = Y.iloc[test, 0]
        w_test = (
            Y.w.iloc[test].values.reshape(-1)
            if has_weights and "w" in Y.columns
            else None
        )
        score_ = {}
        for metric in scoring:
            if metric == "neg_log_loss":
                prob = clf.predict_proba(X_test.values)
                value = -log_loss(y_test, prob, labels=clf.classes_)
            else:
                pred = clf.predict(X_test.values)
                if metric == "neg_mean_squared_error":
                    value = -math.sqrt(mean_squared_error(y_test, pred))
                elif metric == "r2_score":
                    value = r2_score(y_test, pred)
                else:
                    value = accuracy_score(y_test, pred)
            score_[metric] = value

        score.append(score_)

    return pd.DataFrame(score)


def cv_predict(
    clf,
    X: pd.DataFrame,
    Y: pd.Series,
    n_folds=None,
    methods=None,
    cv_gen=None,
    pct_embargo=0.0,
    use_weights=False,
):
    if methods is None:
        methods = ["predict"]

    for m in methods:
        if m not in ["predict", "predict_proba", "predict_level", "decision_function"]:
            raise Exception("wrong method", m)

    if cv_gen is None:
        cv_gen = PooledPurgedKFold(Y.t1, n_folds, pct_embargo)

    y = Y.bin
    y_pred_all = []
    for (i, (train, test)) in enumerate(cv_gen.split(X=X), 1):
        # logger.info("Fitting classifier for cv", i)
        params = _fit_params_for(clf, X, Y, train, has_weights=use_weights)
        fit = clf.fit(**params)
        X_test = X.iloc[test, :].values
        y_pred = []
        for m in methods:
            if m == "predict_proba":
                p = fit.predict_proba(X_test)[:, 1]
            elif m == "decision_function":
                p = fit.decision_function(X_test)
            elif m == "predict_level":
                p = fit.predict_level(X_test)
            else:
                p = fit.predict(X_test)

            y_pred.append(p)

        y_pred_all.append(y_pred)

    return map(lambda x: pd.Series(np.concatenate(x), index=y.index), zip(*y_pred_all))


def _fit_params_for(clf, X, Y, idx=None, has_weights=True):
    if idx is None:
        idx = range(0, X.shape[0])

    params = {"X": X.iloc[idx, :].values, "y": Y.iloc[idx, 0].values.reshape(-1)}
    if has_weights and "w" in Y.columns:
        params.update(
            _params_for(clf, "sample_weight", Y.w.iloc[idx].values.reshape(-1))
        )
    params.update(_params_for(clf, "t1", Y.t1.iloc[idx]))
    return params


def _params_for(clf, param, value, step=None):
    import inspect

    params = {}
    if isinstance(clf, Pipeline):
        for step_name, step_clf in clf.steps:
            if step is None:
                new_step = step_name
            else:
                new_step = f"{step}__{step_name}"
            params.update(_params_for(step_clf, param, value, step=new_step))
    else:
        fit_method = getattr(clf, "fit")
        if fit_method is not None and param in inspect.signature(fit_method).parameters:
            key = param if step is None else f"{step}__{param}"
            params[key] = value

    return params


def class_method(clf, method):
    return clf.__class__.__dict__.get(method, None)
