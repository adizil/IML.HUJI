from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    m = 1000
    actual_mu = 10
    actual_var = 1
    X = np.random.normal(actual_mu, actual_var, m)

    uni_class = UnivariateGaussian()
    uni_class.fit(X)
    print(uni_class.mu_, uni_class.var_)
    # Question 2 - Empirically showing sample mean is consistent
    uni_abs = UnivariateGaussian()
    abs_gap=[]
    for i in range(10,1001,10):
        uni_abs.fit(X[:i])
        abs_gap.append(abs(uni_abs.mu_ - actual_mu))
    x_axis = np.arange(10,1001,10)
    go.Figure([go.Scatter(x=x_axis, y=abs_gap, mode='markers+lines',
                          name=r'$\widehat\mu$')],
              layout=go.Layout(
                  title=r"$\text{ Distance between the estimated "
                        r" and true value of the expectation}$",
                  xaxis_title="$\\text{ number of samples}$",
                  yaxis_title="r$|\hat\mu - \mu|$",
                  height=600)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdf_X = uni_class.pdf(X)
    go.Figure([go.Scatter(x=X, y=pdf_X, mode='markers',
                          name=r'$\widehat\mu$')],
              layout=go.Layout(
                  title=r"$\text{ PDF function under the fitted model}$",
                  xaxis_title="$\\text{ sample values }$",
                  yaxis_title="r$\\text{pdf}$",
                  height=600)).show()



def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    m=1000
    actual_mu = np.array([0, 0, 4, 0])
    actual_sigma=np.array(
        [[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    X = np.random.multivariate_normal(actual_mu, actual_sigma, m)
    multi_class = MultivariateGaussian()
    multi_class.fit(X)
    print("estimated expectation",multi_class.mu_)
    print ("covariance matrix\n", multi_class.cov_)

    # Question 5 - Likelihood evaluation
    steps = np.linspace(-10, 10, 200)
    mat=[]
    for i in steps:
        in_mat=[]
        f1=i
        for j in steps:
            f3=j
            in_mat.append(multi_class.log_likelihood(np.array([f1, 0, f3, 0]), actual_sigma, X))
        mat.append(in_mat)
    go.Figure(go.Heatmap(x=steps, y=steps, z=mat)).update_layout(
        title="Heatmap of f1 and f3 as a log-likelihood optimization"
            , xaxis_title="f3", yaxis_title="f1",
            height=600, width=900).show()

    # Question 6 - Maximum likelihood
    max_val = np.amax(mat)
    result = np.where(mat == max_val)
    print ("f1:", steps[result[0][0]].round(3,out= None))
    print ("f3:", steps[result[1][0]].round(3,out= None))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
