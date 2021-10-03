import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv('500training.csv')

X = df.drop(columns=["LetterLabel"])
X = df.iloc[: , 2:]
y = df["LetterLabel"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

models = [
    (LogisticRegression(),"lr"),
    (LinearDiscriminantAnalysis(),"lda"),
    (DecisionTreeClassifier(),"tree"),
    (RandomForestClassifier(),"rf"),
    (GaussianNB(),"gnb")
]

for model, model_name in models:
    model.fit(X_train.values, y_train.values)
    print("=" * 48)
    print(f"EVALUATING MODEL {model_name}")
    print("=" * 48)
    print(classification_report(y_test , model.predict(X_test.values)))
    joblib.dump(model, f'models/{model_name}_model.joblib')


# Mapping of labels

# letterToId = {}
# idToLetter = {}
# for i, letter in enumerate(df["LetterLabel"].unique()):
#     idToLetter[i] = letter
#     letterToId[letter] = i

# print(idToLetter)

# X = df.drop(columns=["LetterLabel"])
# X = df.iloc[: , 2:]
# y = df["LetterLabel"]
# y = y.replace(letterToId)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# KNN_model = KNeighborsClassifier(n_neighbors=5)
# #KNN_model = DecisionTreeClassifier()
# KNN_model.fit(X_train, y_train)

# KNN_prediction = KNN_model.predict(X_test)

# print(accuracy_score(KNN_prediction, y_test))

# joblib.dump(KNN_model, 'hand_model.joblib')