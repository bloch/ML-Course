#################################
# Your name: Nathan Bloch
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

"""
Q4.1 skeleton.

Please use the provided functions signature for the SVM implementation.
Feel free to add functions and other code, and submit this file with the name svm.py
"""

# generate points in 2D
# return training_data, training_labels, validation_data, validation_labels
def get_points():
    X, y = make_blobs(n_samples=120, centers=2, random_state=0, cluster_std=0.88)
    return X[:80], y[:80], X[80:], y[80:]


def create_plot(X, y, clf):
    plt.clf()

    # plot the data points
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.PiYG)

    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0] - 2, xlim[1] + 2, 30)
    yy = np.linspace(ylim[0] - 2, ylim[1] + 2, 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])


def train_three_kernels(X_train, y_train, X_val, y_val):
    """
    Returns: np.ndarray of shape (3,2):
        A two dimensional array of size 3 that contains the number of support vectors for each class(2) in the three kernels.
    """
    linear_clf = svm.SVC(kernel='linear', C=1000)
    linear_clf.fit(X_train, y_train)
    
    quadratic_clf = svm.SVC(kernel='poly', C=1000, degree=2)
    quadratic_clf.fit(X_train, y_train)

    rbf_clf = svm.SVC(C=1000)
    rbf_clf.fit(X_train, y_train)

    create_plot(X_train, y_train, rbf_clf)
    # plt.show()

    create_plot(X_train, y_train, quadratic_clf)
    # plt.show()

    create_plot(X_train, y_train, linear_clf)
    # plt.show()

    return np.array([linear_clf.n_support_, quadratic_clf.n_support_, rbf_clf.n_support_])


def linear_accuracy_per_C(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """
    valid_accuracy = np.zeros(11)
    train_accuracy = np.zeros(11)
    C = [pow(10, i) for i in range(-5, 6)]
    for i in range(11):
        clf = svm.SVC(kernel='linear', C = C[i])
        clf.fit(X_train, y_train)
        valid_accuracy[i] = calc_accuracy(X_val, y_val, clf)
        train_accuracy[i] = calc_accuracy(X_train, y_train, clf)

    valid_line, = plt.plot(C, valid_accuracy)
    train_line, = plt.plot(C, train_accuracy)
    plt.xscale('log')
    plt.legend((valid_line, train_line), ('Validation Set', 'Train Set'))
    # plt.show()

    return valid_accuracy


def rbf_accuracy_per_gamma(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """
    valid_accuracy = np.zeros(11)
    train_accuracy = np.zeros(11)
    gammas = [pow(10, i) for i in range(-5, 6)]
    for i in range(11):
        clf = svm.SVC(C = 10, gamma=gammas[i])
        clf.fit(X_train, y_train)
        valid_accuracy[i] = calc_accuracy(X_val, y_val, clf)
        train_accuracy[i] = calc_accuracy(X_train, y_train, clf)

    valid_line, = plt.plot(gammas, valid_accuracy)
    train_line, = plt.plot(gammas, train_accuracy)
    plt.xscale('log')
    plt.legend((valid_line, train_line), ('Validation Set', 'Train Set'))
    # plt.show()

    return valid_accuracy

#====================================  HELPER FUNCTION  =========================================

def calc_accuracy(X_data, Y_data, clf):
    true = 0
    predictions = clf.predict(X_data)
    for i in range(X_data.shape[0]):
        if(predictions[i] == Y_data[i]):
            true += 1

    return true / X_data.shape[0]


# train_data, train_labels, validation_data, validation_labels = get_points();

# 1(a)
# sv = train_three_kernels(train_data, train_labels, validation_data, validation_labels)
# print("# of SVs in linear model:", sum(sv[0]));
# print("# of SVs in quaratic model:", sum(sv[1]));
# print("# of SVs in RBF model:", sum(sv[2]));


# 1(b)
# linear_accuracy_per_C(train_data, train_labels, validation_data, validation_labels)

# 1(c)
# rbf_accuracy_per_gamma(train_data, train_labels, validation_data, validation_labels)
