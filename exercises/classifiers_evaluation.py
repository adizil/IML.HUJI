import sys
import os
from math import atan2, pi
from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from IMLearn.learners.classifiers import perceptron
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
pio.templates.default = "simple_white"
from IMLearn.metrics import loss_functions

import plotly.express as px
sys.path.append(r"C:\Users\AdiZ\IML.HUJI\datasets")


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(r'C:/Users/AdiZ/IML.HUJI/datasets/'+filename)
    return (data[:, :2], data[:, 2])

def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X , y = load_dataset(f)

        # Fit Perceptron and record loss in each fit iteration
        perceptr = Perceptron()
        perceptr.callback_ = perceptron.default_callback
        perceptr.fit(X,y)

        # Plot figure
        #px.line(y=perceptr.training_loss, title=n).show()
        go.Figure(
            [go.Scatter(x=list(range(len(perceptr.training_loss))), y=perceptr.training_loss, mode='lines',
                        line=dict(width=2, color="blue"))],
            layout=go.Layout(
                title=f"Classification LOSS as func of iteration",
                xaxis_title="iteration",
                yaxis_title="losse",
                height=750)).show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix
    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse
    cov: ndarray of shape (2,2)
        Covariance of Gaussian
    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (
        np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines",
                      marker_color="black")

def x_marker (mu) :
    return go.Scatter(x=[mu[0]], y=[mu[1]], mode='markers',
                      marker=dict(
                          color='black', symbol='cross', size=8))

def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X , y = load_dataset(f)
        # Fit models and predict over training set
        gaussian = GaussianNaiveBayes ()
        gaussian.fit(X,y)
        gaussian_pred = gaussian.predict(X)
        gaussian_accuracy = loss_functions.accuracy(y,gaussian_pred)

        lda = LDA()
        lda.fit(X,y)
        lda_pred = lda.predict(X)
        lda_accuracy = loss_functions.accuracy(y,lda_pred)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        from IMLearn.metrics import accuracy
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=(f"Gaussian Naive Bayes accuracy: {gaussian_accuracy}",
                                            f"LDA accuracy: {lda_accuracy}"))
        fig.update_layout(width=1500, height=700,
                          title='Gaussian Naive Bayes and LDA predictions',
                          showlegend=False)
        fig.add_trace(go.Scatter(x=X[:,0],y=X[:,1], mode='markers', marker=dict(
            color= gaussian_pred, symbol=y, size=7)), row=1, col=1)
        fig.add_trace(
            go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(
                color=lda_pred, symbol=y, size=7)), row=1, col=2)
        #add x marker
        for i in range(len(gaussian.classes_)) :
            fig.add_trace(x_marker(gaussian.mu_[i]),row =1, col =1)
            fig.add_trace(x_marker(lda.mu_[i]),row =1, col =2)

        #add ellipsis marker
        for i in range(len(gaussian.classes_)):
            fig.add_trace(get_ellipse(gaussian.mu_[i],np.diag(gaussian.vars_[i])),row =1, col =1)
            fig.add_trace(get_ellipse(lda.mu_[i], lda.cov_),row =1, col =2)
        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
    '''
    x=np.array([[0],[1],[2],[3],[4],[5],[6],[7]])
    y= np.array([[0],[0],[1],[1],[1],[1],[2],[2]])
    g= GaussianNaiveBayes()
    g.fit(x,y)
    print("mu", g.mu_)
    print("class",g.classes_)
    print("pi",g.pi_)
    
    
    x=np.array([[1,1],[1,2],[2,3],[2,4],[3,3],[3,4]])
    y= np.array([[0],[0],[1],[1],[1],[1]])
    g= GaussianNaiveBayes()
    g.fit(x,y)
    print("mu", g.mu_)
    print("class",g.classes_)
    print("sigma",g.vars_)
    '''




