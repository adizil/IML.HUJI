from __future__ import annotations
from typing import NoReturn
from ...base import BaseEstimator, BaseModule
from ...desent_methods.gradient_descent import GradientDescent
import numpy as np
from sklearn import linear_model
from IMLearn.metrics.loss_functions import mean_square_error


class LassoObjective(BaseModule):
    """
    Module class of the Lasso objective
    """

    def __init__(self, lam: float, nfeatures: int, include_intercept: bool = False) -> LassoObjective:
        """
        Initialize a Lasso objective module

        Parameters
        ----------
        lam: float
            Value of regularization parameter lambda

        nfeatures: int
            Dimensionality of data

        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        Attributes
        ----------
        lam_: float
            Value of regularization parameter lambda

        include_intercept_: bool
            Should fitted model include an intercept or not
        """
        super().__init__()
        raise NotImplementedError()

    def compute_output(self, input: np.ndarray, compare=None) -> np.ndarray:
        raise NotImplementedError()

    def compute_jacobian(self, input: np.ndarray, compare=None) -> np.ndarray:
        raise NotImplementedError()


class LassoRegression(BaseEstimator):
    """
    Lassi Regression Estimator

    Solving Lasso regression optimization problem
    """

    def __init__(self, lam: float, optimizer: GradientDescent, include_intercept: bool = False):
        """
        Initialize a ridge regression model
        :param lam: scalar value of regularization parameter
        """
        super().__init__()
        self.lam_ = lam
        self.include_intercept_ = include_intercept
        self.optimizer_ = optimizer
        self._objective = None
        self.coefs_ = None
        self.lasso_ = None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Lasso regression model to given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model using specified `self.optimizer_` passed when instantiating class and includes an intercept
        if specified by `self.include_intercept_
        """
        """
        self.lasso_ = linear_model.Lasso(self.lam_,fit_intercept=self.include_intercept_)
        self.lasso_.fit(X,y)
        self.coefs_ = self.lasso_.coef_
        """
        raise NotImplementedError()

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
       # return self.lasso_.prefict(X)
        raise NotImplementedError()

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """

        """
        y_pred = self._predict(X)
        mse = mean_square_error(y, y_pred)
        return mse + (self.lam_* (np.linalg.norm (self.coefs_, ord=1)))
        """
        raise NotImplementedError()