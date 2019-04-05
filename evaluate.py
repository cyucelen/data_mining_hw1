from sklearn import metrics
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def print_formatted_score(modelName, scoreType, value):
    print('{:^28} | {:^12}: {:^8}'.format(modelName, scoreType, str(np.around(value, decimals=8))))


def cross_val_by_mse(model, X, y):
    return -cross_val_score(model, X, y, cv=10, scoring='neg_mean_squared_error').mean()


def cross_val_by_r2(model, X, y):
    return cross_val_score(model, X, y, cv=10, scoring="r2").mean()


def cross_val_by_accuracy(model, X, y):
    return cross_val_score(model, X, y, cv=10, scoring='accuracy').mean()


def print_confusion_matrix(model, X, y):
    print("\nConfusion matrix: ")
    y_pred = cross_val_predict(model, X, y, cv=10)
    print(metrics.confusion_matrix(y, y_pred), "\n")
    print("------------------------------")


def get_classification_models():
    models = [
        ("Logistic Regression", LogisticRegression()),
        ("Linear Discriminant Analysis", LinearDiscriminantAnalysis()),
        ("Gaussian Naive Bayes", GaussianNB())
    ]
    for i in range(1, 5):
        models.append(("K-NN K = %d" % i, KNeighborsClassifier(n_neighbors=i)))

    return models


def get_regression_models():
    models = [
        ("Linear Regression", LinearRegression()),
    ]

    for i in range(1, 6):
        alpha = pow(10, -i)
        models.append(("Ridge | alpha = %.5f" % alpha, Ridge(alpha=alpha)))

    for i in range(1, 6):
        alpha = pow(10, -i)
        models.append(("Lasso | alpha = %.5f" % alpha, Lasso(alpha=alpha, max_iter=10e5)))

    return models


def evaluate_regression_models(X, y):
    for name, model in get_regression_models():
        print_formatted_score(name, "MSE", cross_val_by_mse(model, X, y))
        print_formatted_score(name, "R2", cross_val_by_r2(model, X, y))


def evaluate_classification_models(X, y):
    for name, model in get_classification_models():
        score = cross_val_by_accuracy(model, X, y)
        print_formatted_score(name, "Accuracy Score", score)
        print_confusion_matrix(model, X, y)
