import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from evaluate import evaluate_classification_models
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

print("{:^28}".format("*DIGITS DATASET*"))
print("-----------------------------")

digits = load_digits()

X = digits["data"]
y = digits["target"]

evaluate_classification_models(X, y)
