from sklearn.datasets import load_diabetes
from evaluate import evaluate_regression_models
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

print("{:^28}".format("*DIABETES DATASET*"))
print("-----------------------------")

diabetes = load_diabetes()

X = pd.DataFrame(diabetes["data"], columns=diabetes["feature_names"])
y = diabetes["target"]


evaluate_regression_models(X, y)
