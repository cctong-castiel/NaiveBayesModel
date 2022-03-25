import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from lib.naivebayes import _NaiveBayesClassifier
from dotenv import load_dotenv
import os

load_dotenv()
seed = int(os.getenv("seed"))
np.random.seed(seed)

def main():

    # load_data
    X, y = load_breast_cancer(return_X_y=True)

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # train model
    nb_skl = MultinomialNB()
    nb_ctm = _NaiveBayesClassifier()

    # fit
    nb_skl.fit(X_train, y_train)
    nb_ctm.fit(X_train, y_train)

    # predict
    y_pred_skl = nb_skl.predict(X_test)
    y_pred_ctm = nb_ctm.predict(X_test)

    # predict proba
    y_pred_proba_skl = nb_skl.predict_proba(X_test)
    y_pred_proba_ctm = nb_ctm.predict_proba(X_test)

    print(f"y_pred_proba_skl: {y_pred_proba_skl}")
    print(f"y_pred_proba_ctm: {y_pred_proba_ctm}")

    # classification report
    print(f"classification report of sklearn naive bayes: \n{classification_report(y_test, y_pred_skl)}\n")
    print(f"classification report of custom naive bayes: \n{classification_report(y_test, y_pred_ctm)}\n")


if __name__ == "__main__":
    main()