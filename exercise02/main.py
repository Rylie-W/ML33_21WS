import sklearn.datasets as datasets
import matplotlib.pyplot as plt
from sklearn import model_selection
import numpy as np


def load_dataset(split):
    """Load and split the dataset into training and test parts.

    Parameters
    ----------
    split : float in range (0, 1)
        Fraction of the data used for training.

    Returns
    -------
    X_train : array, shape (N_train, 4)
        Training features.
    y_train : array, shape (N_train)
        Training labels.
    X_test : array, shape (N_test, 4)
        Test features.
    y_test : array, shape (N_test)
        Test labels.
    """
    dataset = datasets.load_iris()
    X, y = dataset['data'], dataset['target']
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=123, test_size=(1 - split))
    return X_train, X_test, y_train, y_test
def prepare_data():
    # prepare data
    split = 0.75
    X_train, X_test, y_train, y_test = load_dataset(split)
    return X_train, y_train

def plot_data(X_train, y_train):
    f, axes = plt.subplots(4, 4, figsize=(15, 15))
    for i in range(4):
        for j in range(4):
            if j == 0 and i == 0:
                axes[i, j].text(0.5, 0.5, 'Sepal. length', ha='center', va='center', size=24, alpha=.5)
            elif j == 1 and i == 1:
                axes[i, j].text(0.5, 0.5, 'Sepal. width', ha='center', va='center', size=24, alpha=.5)
            elif j == 2 and i == 2:
                axes[i, j].text(0.5, 0.5, 'Petal. length', ha='center', va='center', size=24, alpha=.5)
            elif j == 3 and i == 3:
                axes[i, j].text(0.5, 0.5, 'Petal. width', ha='center', va='center', size=24, alpha=.5)
            else:
                axes[i, j].scatter(X_train[:, j], X_train[:, i], c=y_train, cmap=plt.cm.cool)
    plt.show()


def euclidean_distance(x1, x2):
    """Compute pairwise Euclidean distances between two data points.

    Parameters
    ----------
    x1 : array, shape (N, 4)
        First set of data points.
    x2 : array, shape (M, 4)
        Second set of data points.

    Returns
    -------
    distance : float array, shape (N, M)
        Pairwise Euclidean distances between x1 and x2.
    """
    # https://www.dabblingbadger.com/blog/2020/2/27/implementing-euclidean-distance-matrix-calculations-from-scratch-in-python]
    M=x1.shape[0]
    N=x2.shape[0]

    x1_dots=(x1*x1).sum(axis=1).reshape(M,1)*np.ones(shape=(1,N))
    # x2_dots=(x2*x2).sum(axis=1)*np.ones(shape=(M*1))
    x2_dots = (x2 * x2).sum(axis=1)
    return x1_dots+x2_dots-2*x1.dot(x2.T)


def get_neighbors_labels(X_train, y_train, X_new, k):
    """Get the labels of the k nearest neighbors of the datapoints x_new.

    Parameters
    ----------
    X_train : array, shape (N_train, 4)
        Training features.
    y_train : array, shape (N_train)
        Training labels.
    X_new : array, shape (M, 4)
        Data points for which the neighbors have to be found.
    k : int
        Number of neighbors to return.

    Returns
    -------
    neighbors_labels : array, shape (M, k)
        Array containing the labels of the k nearest neighbors.
    """
    neighbors_labels=[]

    distance_matrix = euclidean_distance(X_new, X_train)
    for d in distance_matrix:
        # https://stackoverflow.com/questions/34226400/find-the-index-of-the-k-smallest-values-of-a-numpy-array
        k_nearest_idex=np.argpartition(d,k)
        labels=[y_train[i] for i in k_nearest_idex[:k]]
        neighbors_labels.append(labels)

    return np.array(neighbors_labels)


def get_response(neighbors_labels, num_classes=3):
    """Predict label given the set of neighbors.

    Parameters
    ----------
    neighbors_labels : array, shape (M, k)
        Array containing the labels of the k nearest neighbors per data point.
    num_classes : int
        Number of classes in the dataset.

    Returns
    -------
    y : int array, shape (M,)
        Majority class among the neighbors.
    """
    # TODO
    y=[]
    for labels in neighbors_labels:
        unique, counts=np.unique(labels,return_counts=True)
        y.append(unique[np.argmax(counts)])
    return np.array(y)


def compute_accuracy(y_pred, y_test):
    """Compute accuracy of prediction.

    Parameters
    ----------
    y_pred : array, shape (N_test)
        Predicted labels.
    y_test : array, shape (N_test)
        True labels.
    """
    false_counter=0
    for i in range(len(y_pred)):
        if y_pred[i]!=y_test[i]:
            false_counter+=1
    return 1-false_counter/len(y_pred)


# This function is given, nothing to do here.
def predict(X_train, y_train, X_test, k):
    """Generate predictions for all points in the test set.

    Parameters
    ----------
    X_train : array, shape (N_train, 4)
        Training features.
    y_train : array, shape (N_train)
        Training labels.
    X_test : array, shape (N_test, 4)
        Test features.
    k : int
        Number of neighbors to consider.

    Returns
    -------
    y_pred : array, shape (N_test)
        Predictions for the test data.
    """
    neighbors = get_neighbors_labels(X_train, y_train, X_test, k)
    y_pred = get_response(neighbors)
    return y_pred
if __name__ == '__main__':
    # X_train, y_train=prepare_data()
    # plot_data(X_train, y_train)

    # prepare data
    split = 0.75
    X_train, X_test, y_train, y_test = load_dataset(split)
    print('Training set: {0} samples'.format(X_train.shape[0]))
    print('Test set: {0} samples'.format(X_test.shape[0]))

    # generate predictions
    k = 3
    y_pred = predict(X_train, y_train, X_test, k)
    accuracy = compute_accuracy(y_pred, y_test)
    print('Accuracy = {0}'.format(accuracy))