from __future__ import annotations
from typing import Callable
from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from IMLearn.metrics import loss_functions


def default_callback(fit: Perceptron, x: np.ndarray, y: int):
    fit.training_loss.append(fit._loss(x,y))


class Perceptron(BaseEstimator):
    """
    Perceptron half-space classifier

    Finds a separating hyperplane for given linearly separable data.

    Attributes
    ----------
    include_intercept: bool, default = True
        Should fitted model include an intercept or not

    max_iter_: int, default = 1000
        Maximum number of passes over training data

    coefs_: ndarray of shape (n_features,) or (n_features+1,)
        Coefficients vector fitted by Perceptron algorithm. To be set in
        `Perceptron.fit` function.

    callback_: Callable[[Perceptron, np.ndarray, int], None]
            A callable to be called after each update of the model while fitting to given data
            Callable function should receive as input a Perceptron instance, current sample and current response
    """
    def __init__(self,
                 include_intercept: bool = True,
                 max_iter: int = 1000,
                 callback: Callable[[Perceptron, np.ndarray, int], None] = default_callback):
        """
        Instantiate a Perceptron classifier

        Parameters
        ----------
        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        max_iter: int, default = 1000
            Maximum number of passes over training data

        callback: Callable[[Perceptron, np.ndarray, int], None]
            A callable to be called after each update of the model while fitting to given data
            Callable function should receive as input a Perceptron instance, current sample and current response
        """
        super().__init__()
        self.include_intercept_ = include_intercept
        self.max_iter_ = max_iter
        self.callback_ = callback
        self.coefs_ = None
        self.training_loss = []

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit a halfspace to to given samples. Iterate over given data as long as there exists a sample misclassified
        or that did not reach `self.max_iter_`

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.fit_intercept_`
        """

        if self.include_intercept_:
          vec = np.array(np.ones(len(X), ), dtype=int).reshape(1, -1)
          X = np.concatenate((np.transpose(vec), X), axis=1)

        self.coefs_ = np.zeros(len(X[0]))
        misclassified = np.sign(X @ self.coefs_) - y
        iter_num=0
        while (iter_num<self.max_iter_ and np.any(misclassified) ):
            for i in range(len(X)) :
                if (X[i] @ self.coefs_)*y[i] <= 0 :
                    self.fitted_ = True
                    self.coefs_ += X[i] * y[i]
                    self.callback_(self, X,y)
                    break
            misclassified = np.sign(X @ self.coefs_) - y
            iter_num += 1



    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        if np.shape(X)[1] < len(self.coefs_):
            X = np.insert(X, 0, np.ones(np.shape(X)[0]), axis=1)
        return np.sign(X @ self.coefs_)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
<<<<<<< HEAD
        return loss_functions.misclassification_error(y, self._predict(X))
=======
        from ...metrics import misclassification_error
        raise NotImplementedError()
>>>>>>> c87be5d7872d40b4409d315bf2d2360bc8a3d675
