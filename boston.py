from sklearn.datasets import load_boston
from evaluate import evaluate_regression_models
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

print("{:^28}".format("*BOSTON DATASET*"))
print("-----------------------------")

boston = load_boston()

X = pd.DataFrame(boston["data"], columns=boston["feature_names"])
y = boston["target"]

evaluate_regression_models(X, y)
