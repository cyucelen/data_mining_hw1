import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from evaluate import evaluate_classification_models
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

print("{:^28}".format("*DIGITS DATASET*"))
print("-----------------------------")

wine = load_wine()

X = pd.DataFrame(wine["data"], columns=wine["feature_names"])
y = wine["target"]

evaluate_classification_models(X, y)
