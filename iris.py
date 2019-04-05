from sklearn.datasets import load_iris

from evaluate import evaluate_classification_models
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

print("{:^28}".format("*IRIS DATASET*"))
print("-----------------------------")

iris = load_iris()

X = pd.DataFrame(iris["data"], columns=iris["feature_names"])
y = iris["target"]

evaluate_classification_models(X, y)
