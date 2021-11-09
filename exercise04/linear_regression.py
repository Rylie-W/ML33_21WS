import numpy as np

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

def fit_least_squares(X, y):
    """Fit ordinary least squares model to the data.

    Parameters
    ----------
    X : array, shape [N, D]
        (Augmented) feature matrix.
    y : array, shape [N]
        Regression targets.

    Returns
    -------
    w : array, shape [D]
        Optimal regression coefficients (w[0] is the bias term).

    """
    x_transposed=np.transpose(X)
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(x_transposed,X)),x_transposed),y)


def fit_ridge(X, y, reg_strength):
    """Fit ridge regression model to the data.

    Parameters
    ----------
    X : array, shape [N, D]
        (Augmented) feature matrix.
    y : array, shape [N]
        Regression targets.
    reg_strength : float
        L2 regularization strength (denoted by lambda in the lecture)

    Returns
    -------
    w : array, shape [D]
        Optimal regression coefficients (w[0] is the bias term).

    """
    x_transposed = np.transpose(X)
    lamb=np.zeros((np.shape(X)[1],np.shape(X)[1]))
    np.fill_diagonal(lamb,reg_strength)
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(x_transposed,X)+lamb),x_transposed),y)


def predict_linear_model(X, w):
    """Generate predictions for the given samples.

    Parameters
    ----------
    X : array, shape [N, D]
        (Augmented) feature matrix.
    w : array, shape [D]
        Regression coefficients.

    Returns
    -------
    y_pred : array, shape [N]
        Predicted regression targets for the input data.

    """
    return np.matmul(X,w)


def mean_squared_error(y_true, y_pred):
    """Compute mean squared error between true and predicted regression targets.

    Reference: `https://en.wikipedia.org/wiki/Mean_squared_error`

    Parameters
    ----------
    y_true : array
        True regression targets.
    y_pred : array
        Predicted regression targets.

    Returns
    -------
    mse : float
        Mean squared error.

    """
    n=np.shape(y_true)[0]
    return np.matmul(np.transpose(y_true-y_pred),y_true-y_pred)/n

