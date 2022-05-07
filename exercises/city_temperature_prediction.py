import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    dataSet = pd.read_csv(filename, parse_dates=[2]).drop_duplicates()
    dataSet = dataSet[dataSet["Month"].isin(range(1, 13))]
    dataSet = dataSet[dataSet["Day"].isin(range(1, 32))]
    dataSet = dataSet[dataSet["Temp"] > -70]
    dataSet["DayOfYear"] = dataSet["Date"].dt.dayofyear
    return dataSet

if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("..\datasets\City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    df['Year'] = df['Year'].astype(str)
    israel_df = df.loc[df["Country"] == "Israel"]
    px.scatter(israel_df, x="DayOfYear", y="Temp", color="Year", title="Daily temperature Israel per year").show()
    month_israel = israel_df.groupby("Month").agg("std")
    px.bar(month_israel, y="Temp", title="Israel STD daily temperatures per month").show()


    # Question 3 - Exploring differences between countries

    country_month = df.groupby(["Country", "Month"]).agg({"Temp": ['mean', 'std']})
    country_month.columns = ["mean_temp", "std_temp"]
    country_month = country_month.reset_index()
    px.line(country_month, x="Month", y="mean_temp", error_y="std_temp",
            color="Country", title="Average monthly temperature per country").show()

    # Question 4 - Fitting model for different values of `k`
    israel_temp_y= israel_df.Temp
    israel_df_x= israel_df.DayOfYear
    train_X, train_y, test_X, test_y = split_train_test(israel_df_x,israel_temp_y)
    loss=[]
    for k in range(10):
        pf = PolynomialFitting(k)
        pf.fit(train_X.to_numpy(), train_y.to_numpy())
        loss.append(round(pf.loss(test_X.to_numpy(), test_y.to_numpy()), 2))
    px.bar(x=range(1, 11), y=loss,
                 title='Test error for each value of K',
                 labels={"x": 'k- Polynoms Degree', 'y': 'loss'}).show()


    # Question 5 - Evaluating fitted model on different countries

    loss_countries = []
    pf = PolynomialFitting(5)
    pf.fit(train_X.to_numpy(), train_y.to_numpy())
    countries =[]
    for country in df["Country"].unique():
        if country!="Israel":
            pf_country = df[df.Country == country]
            loss_countries.append(pf.loss(pf_country.DayOfYear.to_numpy(),pf_country.Temp.to_numpy()))
            countries.append(country)
    px.bar(x=np.array(countries), y=np.array(loss_countries),
                 title="Error as func of Country ",
                 labels={"x": 'Countries', 'y': "Error"}).show()