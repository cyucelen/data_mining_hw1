from sklearn.datasets import load_breast_cancer

from evaluate import evaluate_classification_models
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

print("{:^28}".format("*BREAST CANCER DATASET*"))
print("-----------------------------")

breast_cancer = load_breast_cancer()

X = pd.DataFrame(breast_cancer["data"], columns=breast_cancer["feature_names"])
y = breast_cancer["target"]

evaluate_classification_models(X, y)
