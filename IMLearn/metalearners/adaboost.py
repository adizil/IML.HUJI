import numpy as np
from typing import Callable, NoReturn
from ..base import BaseEstimator
from IMLearn.metrics import loss_functions


class AdaBoost(BaseEstimator):
    """
    AdaBoost class for boosting a specified weak learner

    Attributes
    ----------
    self.wl_: Callable[[], BaseEstimator]
        Callable for obtaining an instance of type BaseEstimator

    self.iterations_: int
        Number of boosting iterations to perform

    self.models_: List[BaseEstimator]
        List of fitted estimators, fitted along the boosting iterations
    """

    def __init__(self, wl: Callable[[], BaseEstimator], iterations: int):
        """
        Instantiate an AdaBoost class over the specified base estimator

        Parameters
        ----------
        wl: Callable[[], BaseEstimator]
            Callable for obtaining an instance of type BaseEstimator

        iterations: int
            Number of boosting iterations to perform
        """
        super().__init__()
        self.wl_ = wl
        self.iterations_ = iterations
        self.models_, self.weights_, self.D_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an AdaBoost classifier over given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        D = np.array([1/len(y)]*len(y))
        self.models_ = np.array([])
        self.weights_ = np.array([])
        for iter in range(self.iterations_):
            print(iter)
            h = self.wl_()
            h.fit(X,y*D)
            self.models_ = np.append(self.models_,h)
            pred_h = h.predict(X)
           # epsilon = np.sum(D*(h._loss(X,y)))
            epsilon = np.sum(D * [np.sign(pred_h)!=np.sign(y)])
            w = (1/2) *np.log((1/epsilon)-1)
            self.weights_ = np.append(self.weights_,w)
            #for i in range (len(D)):
             #   D[i] = D[i]*(np.exp(-y[i]*w*pred_h[i]))
            D = list(np.exp(
                -y * np.array(self.weights_[-1]) * pred_h) * np.array(D))
            D = D/np.sum(D)
        self.D_=D


    def _predict(self, X):
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
        pred = np.zeros(np.shape(X)[0])
        for index, model in enumerate(self.models_):
            pred += self.weights_[index] * model.predict(X)
        return np.sign(pred)

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
        return loss_functions.misclassification_error(y, self._predict(X))

    def partial_predict(self, X: np.ndarray, T: int) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimators

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        pred = np.zeros(np.shape(X)[0])
        for index, model in enumerate(self.models_[0:T]):
            pred += self.weights_[index] * model.predict(X)
        return np.sign(pred)

    def partial_loss(self, X: np.ndarray, y: np.ndarray, T: int) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return loss_functions.misclassification_error(y, self.partial_predict(X,T))

