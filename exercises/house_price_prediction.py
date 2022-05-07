from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"
import os


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """

    dataSet = pd.read_csv(filename)
    dataSet.drop(["id"],axis=1, inplace=True) #delete the id column
    dataSet.drop(["sqft_basement"],axis=1, inplace=True)  #delete the sqft_basement col to keep iid
    dataSet.drop(["date"], axis=1, inplace=True) #delete the date column
    dataSet.drop(["lat"], axis=1, inplace=True)  # delete the lat column
    dataSet.drop(["long"], axis=1, inplace=True)  # delete the long column
    dataSet = pd.get_dummies(dataSet, columns=["zipcode"])  # one-hot to zipcode column

    min_year = dataSet['yr_built'].min()
    dataSet['yr_built'] = dataSet['yr_built'] - min_year

    for col in ["price", "bathrooms", "sqft_living", "sqft_lot", "floors",
            "sqft_above", "yr_renovated", "sqft_living15", "sqft_lot15"]:
        dataSet = dataSet[dataSet[col] >= 0]

    dataSet = dataSet[dataSet["grade"].isin(range(1, 14))]
    dataSet = dataSet[dataSet["waterfront"].isin([0, 1])]
    dataSet = dataSet[dataSet["view"].isin([0, 1, 2, 3, 4])]
    dataSet = dataSet[dataSet["condition"].isin([1, 2, 3, 4, 5])]

    prices_y= dataSet.loc[:, "price"]
    dataSet.drop(["price"], axis=1, inplace=True) #delete the price column

    return dataSet,prices_y




def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    if not os.path.exists("Images"):
        os.mkdir("Images")
    sigma_Y = np.std(y)
    for feature in X:
        sigma_X = np.std(X[feature])
        pearson_corr = np.cov(X[feature], y)[0, 1] / (sigma_X * sigma_Y)
        plot = go.Figure([go.Scatter(x=X[feature], y=y, mode='markers',
                              name=r'$\widehat\mu$')], layout=go.Layout(
                      title_text="Feature "+str(feature)+ " as a function"
            " of the response"+"<br>Pearson Correlation is "+str(pearson_corr),
                      xaxis_title=str(feature),
                      yaxis_title="response"))
        plot.write_image(output_path+"/Images\{x}_as_f_response.png".format(x=str(feature)))

if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    df, reasponse = load_data("..\datasets\house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(df,reasponse)

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(df,reasponse)


    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    p = np.arange(10,101,1)
    mean_loss=[]
    var_loss=[]
    test_X_np = test_X.to_numpy()
    test_y_np = test_y.to_numpy()
    for i in range (10, 101):
        loss=[]
        for j in range(10):
            lr = LinearRegression()
            x_train_samp=train_X.sample(frac= i/100)
            y_train_samp=train_y.loc[x_train_samp.index]
            lr.fit(x_train_samp.to_numpy(), y_train_samp.to_numpy())
            loss.append(lr.loss(test_X_np, test_y_np))
        mean_loss.append(np.mean(loss))
        var_loss.append(np.std(loss))
    mean_loss_np=np.array(mean_loss)
    var_loss_np=np.array(var_loss)
    go.Figure([go.Scatter(x=p,
                          y=(mean_loss_np + (var_loss_np * 2)),
                          mode='lines',
                          marker=dict(color='rgb(204, 221, 255)'),
                          showlegend=False),
               go.Scatter(x=p,
                          y=(mean_loss_np - (var_loss_np * 2)),
                          mode='lines',
                          marker=dict(color='rgb(204, 221, 255)'),
                          showlegend=False, fill='tonexty',
                          fillcolor='rgb(204, 221, 255)'),
               go.Scatter(x=p, y=mean_loss_np, mode="lines",
                          marker=dict(color='rgb(0, 48, 153)'))],
              layout=go.Layout(
                  title="Mean loss as a function of percentage n Error",
                  xaxis=dict(title="percentage of training data"),
                  yaxis=dict(title="MSE"))).show()





