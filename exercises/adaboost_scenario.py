import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IMLearn.metrics import loss_functions


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)
    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost = AdaBoost(lambda : DecisionStump(),n_learners)
    adaboost.fit(train_X,train_y)
    test_errors = []
    train_errors = []
    for learner_num in range(1,n_learners+1):
        test_errors.append(adaboost.partial_loss(test_X,test_y,learner_num))
        train_errors.append(adaboost.partial_loss(train_X,train_y,learner_num))
    first_graph = go.Figure()
    first_graph.update_layout(title=rf"$Train and Test errors as a function of fitted learners$ ")
    first_graph.add_trace(go.Line(x=np.arange(1, n_learners, 1, dtype=int), y=train_errors[:n_learners],name="train errors"))
    first_graph.add_trace(go.Line(x=np.arange(1, n_learners, 1, dtype=int), y=test_errors[:n_learners],name="test errors")).show()


    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]

    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    predictions = []
    for i in range(1,251):
        predictions.append(adaboost.partial_predict(test_X, i))
    second_graph = make_subplots(rows=2, cols=3,subplot_titles=[rf"$\text{{{t} learners}}$" for t in T],
                         horizontal_spacing=0.01, vertical_spacing=.03)
    symbols = np.array(["circle", "x"])
    y = np.where(test_y == 1, 1, 0)
    for i, m in enumerate(T):
        second_graph.add_traces(
            [decision_surface(lambda x: adaboost.partial_predict(x,T=m),lims[0], lims[1], showscale=False),
             go.Scatter(x=test_X[:, 0], y=test_X[:, 1],mode="markers",showlegend=False,marker=dict(color=y, symbol=symbols[y],
                                    colorscale=[custom[0],custom[-1]],line=dict(color="black",width=1)))],
                                    rows=(i // 3) + 1, cols=(i % 3) + 1)
    second_graph.update_layout(title=rf"$\text{{Models with [5, 50, 100,250] learners}}$ "
        , margin=dict(t=100)).update_xaxes(visible=False).update_yaxes(visible=False).show()

    # Question 3: Decision surface of best performing ensemble
    error_array=[test_errors[t-1] for t in T]
    min_error_index = np.argmin(error_array)
    predict_by_test = adaboost.partial_predict(test_X, T[min_error_index])
    third_graph = go.Figure()
    print(test_y)
    print(predict_by_test)
    accuracy = loss_functions.accuracy(test_y,predict_by_test)
    third_graph.update_layout(title=f"Ensemble size {T[min_error_index]}"f"learners, and {accuracy} accuracy")
    third_graph.add_traces([decision_surface(lambda x: adaboost.partial_predict(x, T=T[min_error_index]), lims[0], lims[1], showscale=False),
        go.Scatter(x=test_X[:, 0], y=test_X[:, 1],mode="markers",showlegend=False,
                   marker=dict(color=y,symbol=symbols[y],colorscale=[custom[0],custom[-1]],line=dict(color="black",width=1)))]).show()

    # Question 4: Decision surface with weighted samples
    sized = 15 * adaboost.D_ / (np.max(adaboost.D_))
    symbols = np.array(["circle", "x"])
    colors = np.array(['red', 'blue'])
    fourth_graph = go.Figure()
    fourth_graph.update_layout(title="Train set Proportional to it's weight")
    fourth_graph.add_traces([decision_surface(lambda x: adaboost.partial_predict(x, T=250),lims[0], lims[1], showscale=False),
        go.Scatter(x=train_X[:, 0], y=train_X[:, 1],mode="markers",showlegend=False,
                   marker=dict(color=colors[[1 if i == 1 else 0 for i in train_y]],
                    symbol=symbols[[1 if i == 1 else 0 for i in train_y]], size=sized,
                   colorscale=[custom[0],custom[-1]],line=dict(color="black",width=1)))]).show()

if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)

