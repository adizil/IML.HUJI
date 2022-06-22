from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    x_sets = np.array_split(X,cv)
    y_sets = np.array_split(y,cv)
    validation_score = []
    train_score = []
    for i in range(cv):
        train_x = np.delete(x_sets,i,0)
        train_y = np.delete(y_sets,i,0)
        train_x = [train_x[x][i] for x in range(len(train_x)) for i in range(len(train_x[x]))]
        train_y = [train_y[y][i] for y in range(len(train_y)) for i in range(len(train_y[y]))]
        test_x = x_sets[i]
        test_y = y_sets[i]
        estimator.fit(train_x,train_y)
        validation_score.append(scoring(test_y,estimator.predict(test_x)))
        train_score.append(scoring(train_y, estimator.predict(train_x)))
    return float(np.mean(train_score)), float(np.mean(validation_score))




