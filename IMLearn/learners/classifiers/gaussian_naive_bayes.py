from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
import scipy.stats
from IMLearn.metrics import loss_functions


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.mu_ = np.array([])
        self.pi_ = np.array([])
        self.vars_ = np.array([])
        self.classes_ = np.unique(y)
        for clas in self.classes_:
            x_class = np.array([])
            for i in range(len(X)):
                if (y[i]==clas):
                    if (len(x_class)==0):
                        x_class = np.append(x_class, X[i])
                    else:
                        x_class=np.vstack([x_class,X[i]])
            self.pi_ = np.append(self.pi_, len(x_class) / len(y))
            if (len(self.mu_) == 0):
                self.mu_ = np.append(self.mu_, x_class.mean(axis=0))
                self.vars_ = np.append(self.vars_, x_class.var(axis=0, ddof=1))
            else:
                self.mu_ = np.vstack([self.mu_, x_class.mean(axis=0)])
                self.vars_ = np.vstack([self.vars_,x_class.var(axis=0, ddof=1)])

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
        responses = np.array([])
        for arg in self.likelihood(X):
            responses = np.append(responses, np.argmax(arg))
        return responses

    def calc_like_pdf(self,k,x,X):
        first = 1 / np.sqrt(
            ((2 * np.pi) ** X.shape[1]) * np.linalg.det(np.diag(self.vars_[k])))
        second = ((x - self.mu_[k]).T @ np.linalg.inv(np.diag(self.vars_[k])) @ (x - self.mu_[k]))
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

        likelihoods = np.array([])
        for x in X:
            x_likelihood = np.array([])
            for ind_k in range(len(self.classes_)):
                k = int(self.classes_[ind_k])
                x_likelihood = np.append(x_likelihood,self.calc_like_pdf(k,x,X))
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
