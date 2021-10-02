import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv('500training.csv')
X = df.drop(columns=["LetterLabel"])
Y = df["LetterLabel"].values
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

