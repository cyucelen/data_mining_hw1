from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def print_formatted_score(modelName, scoreType, value):
    print('{:^28} | {:^12}: {:^8}'.format(modelName, scoreType, str(value)))
    print("-----------------------------")


def cross_val_by_mse(model, X, y):
    return -cross_val_score(model, X, y, cv=10, scoring='neg_mean_squared_error').mean()


def cross_val_by_accuracy(model, X, y):
    return cross_val_score(model, X, y, cv=10, scoring='accuracy').mean()


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

    for i in range(1, 11):
        alpha = i * 0.01
        models.append(("Ridge | alpha = %.2f" % alpha, Ridge(alpha=alpha)))

    for i in range(1, 11):
        alpha = i * 0.01
        models.append(("Lasso | alpha = %.2f" % alpha, Lasso(alpha=alpha, max_iter=10e5)))

    return models


def evaluate_regression_models(X, y):
    for name, model in get_regression_models():
        score = cross_val_by_mse(model, X, y)
        print_formatted_score(name, "MSE", score)


def evaluate_classification_models(X, y):
    for name, model in get_classification_models():
        score = cross_val_by_accuracy(model, X, y)
        print_formatted_score(name, "Accuracy Score", score)
