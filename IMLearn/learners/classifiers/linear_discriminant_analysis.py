from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv
import scipy.stats
from IMLearn.metrics import loss_functions


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.mu_ = np.array([])
        self.pi_ = np.array([])
        self.classes_ = np.unique(y)
        self.cov_ = np.zeros([X.shape[1],X.shape[1]])
        for clas in self.classes_:
            x_class = np.array([])
            for i in range(len(X)):
                if (y[i] == clas):
                    if (len(x_class) == 0):
                        x_class = np.append(x_class, X[i])
                    else:
                        x_class = np.vstack([x_class, X[i]])
            self.pi_ = np.append(self.pi_, len(x_class)/ len(y))
            x_mu = x_class.mean(axis=0)
            if (len(self.mu_)==0) :
                self.mu_ = np.append(self.mu_ , x_mu)
            else:
                self.mu_ = np.vstack([self.mu_,x_mu ])
            for x in x_class:
                self.cov_ += np.outer(x-x_mu,x-x_mu)
        self.cov_ = self.cov_ / (len(y)-len(self.classes_))
        self._cov_inv = inv(self.cov_)


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
        likelihood = self.likelihood(X)
        responses = np.array([])
        for arg in likelihood :
            responses = np.append ( responses, np.argmax(arg))
        return responses

    def calc_like_pdf(self,k,x,X):
        first = 1 / np.sqrt(
            ((2 * np.pi) ** X.shape[1]) * np.linalg.det(self.cov_))
        second = ((x - self.mu_[k]).T @ np.linalg.inv(self.cov_) @ (x - self.mu_[k]))
        return first * np.exp(-0.5 * second) * self.pi_[k]
    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        likelihoods=np.array([])
        for x in X:
            x_likelihood = np.array([])
            for ind_k in range(len(self.classes_)):
                k = int(self.classes_[ind_k])
                x_likelihood = np.append(x_likelihood,self.calc_like_pdf(k, x, X))
            if len(likelihoods) == 0:
                likelihoods = x_likelihood
            else:
                likelihoods = np.vstack([likelihoods, x_likelihood])
        return likelihoods


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
        return loss_functions.misclassification_error(y,self.predict(X))
=======
        from ...metrics import misclassification_error
        raise NotImplementedError()
>>>>>>> c87be5d7872d40b4409d315bf2d2360bc8a3d675
