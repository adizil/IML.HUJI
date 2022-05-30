from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from IMLearn.metrics import loss_functions
from itertools import product


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        curr_threshold = ()
        for feature in range(len(X[0])):
            for sign in [-1,1]:
                if not curr_threshold:
                    curr_threshold = self._find_threshold(X[:,feature],y,sign)
                    self.sign_=sign
                    self.j_ = feature
                else:
                    threshold_return = self._find_threshold(X[:,feature],y,sign)
                    if threshold_return[1]<curr_threshold[1]:
                        self.sign_ = sign
                        self.j_ = feature
                        curr_threshold = threshold_return
        self.threshold_=curr_threshold[0]



    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """

        '''
        pred = np.array([])
        for i in range (len(X)):
            if X[i][self.j_] < self.threshold_:
                pred= np.append(pred, -self.sign_)
            else:
                pred = np.append(pred, self.sign_)'''
        y = np.array([self.sign_ if X[k][self.j_] >= self.threshold_ else
                      -self.sign_ for k in range(np.shape(X)[0])])
        return y


    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        pred_labels = np.array([sign]*len(labels))
        sortes_index = np.argsort(values)
        values = values[sortes_index]
        labels = labels[sortes_index]
        curr_threshold = (values[0]-1,np.sum(abs(labels[np.sign(labels)!= pred_labels])))
        min_threshold_index = values[0]-1
        min_threshold_errors = curr_threshold[1]
        for sample in range(len(values)):
            if sample == (len(values) - 1):
                curr_threshold_index = values[sample] + 1
            else:
                curr_threshold_index = (values[sample] + values[
                    sample + 1]) / 2
            pred_labels[sample]=-sign
            if np.sign(labels[sample]) != sign:
                curr_threshold = (curr_threshold_index,curr_threshold[1]-abs(labels[sample]))
            else:
                curr_threshold = (curr_threshold_index, curr_threshold[1] + abs(labels[sample]))
            if min_threshold_errors > curr_threshold[1]:
                min_threshold_errors = curr_threshold[1]
                min_threshold_index = curr_threshold_index
        threshold = (min_threshold_index, min_threshold_errors)
        return threshold

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
        return loss_functions.misclassification_error(y, self.predict(X))
