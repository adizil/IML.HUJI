from __future__ import annotations
import numpy as np
import pandas as pd

from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression, LassoRegression
from sklearn.linear_model import Lasso
from sklearn.datasets import load_diabetes
from IMLearn.model_selection import cross_validate


from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

MIN_RANGE = -1.2
MAX_RANGE = 2
K_DEG = 10

def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    f = lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    x = np.linspace(MIN_RANGE, MAX_RANGE, n_samples)
    y = f(x) + np.random.normal(0,noise,n_samples)
    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(x),
                                                        pd.Series(y), 2 / 3)
    train_X = np.array(train_X).reshape(len(train_X))
    train_y = np.array(train_y)
    test_X = np.array(test_X).reshape(len(test_X))
    test_y = np.array(test_y)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=x, y=f(x), mode="markers+lines",name="noiseless model"))
    fig1.add_trace(go.Scatter(x=train_X, y=train_y,mode="markers", marker=dict(color='red'),
                              name="train samples"))
    fig1.add_trace(go.Scatter(x=test_X, y=test_y, mode="markers", marker=dict(color='black'),
                              name="test samples")).show()


    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    k_results = []
    for k in range(K_DEG+1):
        k_results.append(cross_validate(PolynomialFitting(k),train_X,train_y,mean_square_error))
    validation_error = [k_results[i][1] for i in range(len(k_results))]
    average_training = [k_results[i][0] for i in range(len(k_results))]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=list(range(11)), y=validation_error, mode="markers+lines", marker=dict(color='black'),
                              name="validation_error "))
    fig2.add_trace(go.Scatter(x=list(range(11)), y=average_training, mode="markers+lines", marker=dict(color='red'),
                              name=" average_training"))
    fig2.update_layout(title="Average training and Validation error as a Function of Polynomial Degrees",
                       xaxis_title="Polynomial Degrees",
                       yaxis_title="Average training and Validation error")
    fig2.show()
    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    lowest_validation_k = np.argmin(validation_error)
    print(lowest_validation_k)
    polynomial_model = PolynomialFitting(int(lowest_validation_k))
    polynomial_model.fit(train_X, train_y)
    test_err = polynomial_model.loss(test_X, test_y)
    print(test_err)


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    data = load_diabetes()
    X, y = data.data, data.target
    train_X, train_y, test_X, test_y = X[:50], y[:50], X[50:], y[50:]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lam_ridge = np.linspace(0, 1, n_evaluations)
    lam_lasso = np.linspace(0, 3.5, n_evaluations)
    ridge_res = []
    lasso_res = []
    for i in range(n_evaluations):
        ridge_res.append(cross_validate(RidgeRegression(lam_ridge[i]),train_X,train_y,mean_square_error))
        lasso_res.append(cross_validate(Lasso(alpha=lam_lasso[i]), train_X, train_y, mean_square_error))
    ridge_train = [x[0] for x in ridge_res]
    ridge_error = [x[1] for x in ridge_res]
    lasso_train = [x[0] for x in lasso_res]
    lasso_error = [x[1] for x in lasso_res]
    fig7 = go.Figure(layout=go.Layout(title="5-Fold Cross-Validation Ridge",
                                       xaxis_title="lambda",
                                       yaxis_title="score"))
    fig7.add_trace(go.Scatter(x=lam_ridge, y=ridge_train, mode="lines", marker=dict(color='black'),name="ridge_train "))
    fig7.add_trace(go.Scatter(x=lam_ridge, y=ridge_error, mode="lines",marker=dict(color='red'), name="ridge_error "))
    fig7.show()
    fig8 = go.Figure(layout=go.Layout(title="5-Fold Cross-Validation Lasso",
                                       xaxis_title="lambda",
                                       yaxis_title="score"))
    fig8.add_trace(go.Scatter(x=lam_lasso, y=lasso_train, mode="lines", marker=dict(color='black'),name="lasso_train "))
    fig8.add_trace(go.Scatter(x=lam_lasso, y=lasso_error, mode="lines",marker=dict(color='red'), name="lasso_error "))
    fig8.show()


    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_ridge = lam_ridge[np.argmin([err[1] for err in ridge_res])]
    best_lasso = lam_lasso[np.argmin([err[1] for err in lasso_res])]
    print("Regularization Parameter:\nLasso: {0}\nRidge: {1}".format(
        best_ridge, best_lasso))

    est_lst = [(RidgeRegression(best_ridge),
                "RidgeRegression".format(best_lasso)),
               (Lasso(alpha=best_lasso), "Lasso"),
               (LinearRegression(), "LinearRegression")]
    for est, est_name in est_lst:
        est = est.fit(train_X, train_y)
        y_pred = est.predict(test_X)
        err = mean_square_error(test_y, y_pred)
        print("{} : {}".format(est_name, err))

if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree(100, 5)
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()
