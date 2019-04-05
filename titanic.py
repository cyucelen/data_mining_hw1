import pandas as pd
import numpy as np
from evaluate import evaluate_classification_models

print("{:^28}".format("*TITANIC DATASET*"))
print("-----------------------------")

title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2,
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3, "Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona": 3, "Mme": 3, "Capt": 3, "Sir": 3}

sex_mapping = {"male": 0, "female": 1}

embarked_mapping = {"S": 0, "C": 1, "Q": 2}

cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}

family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}


def extract_titles(dataframe):
    return dataframe["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)


titanic_df = pd.read_csv("./titanic_data/train.csv")

titanic_df["Title"] = extract_titles(titanic_df)
titanic_df["Title"] = titanic_df["Title"].map(title_mapping)

titanic_df = titanic_df.drop(["Name"], axis=1)

titanic_df["Sex"] = titanic_df["Sex"].map(sex_mapping)

# replace NaN age with median value of Title group
titanic_df["Age"] = titanic_df["Age"].fillna(titanic_df.groupby("Title")["Age"].transform("median"))

titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")
titanic_df["Embarked"] = titanic_df["Embarked"].map(embarked_mapping)

titanic_df["Cabin"] = titanic_df["Cabin"].str[:1]
titanic_df["Cabin"] = titanic_df["Cabin"].map(cabin_mapping)
titanic_df["Cabin"] = titanic_df["Cabin"].fillna(titanic_df.groupby("Pclass")["Cabin"].transform("median"))

titanic_df["FamilySize"] = titanic_df["SibSp"] + titanic_df["Parch"] + 1
titanic_df["FamilySize"] = titanic_df["FamilySize"].map(family_mapping)

titanic_df.loc[titanic_df['Fare'] <= 17, 'Fare'] = 0,
titanic_df.loc[(titanic_df['Fare'] > 17) & (titanic_df['Fare'] <= 30), 'Fare'] = 1,
titanic_df.loc[(titanic_df['Fare'] > 30) & (titanic_df['Fare'] <= 100), 'Fare'] = 2,
titanic_df.loc[titanic_df['Fare'] > 100, 'Fare'] = 3

y = titanic_df["Survived"]
X = titanic_df.drop(["Ticket", "SibSp", "Parch", "PassengerId", "Survived"], axis=1)


# print(titanic_df.head(10))
evaluate_classification_models(X, y)
