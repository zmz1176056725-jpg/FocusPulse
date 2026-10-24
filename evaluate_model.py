# -*- coding: utf-8 -*-
"""Model evaluation for FocusPulse.

Loads the (synthetic) training data, runs stratified k-fold cross-validation
for the KNN classifier, and compares against random-guessing baselines.

NOTE: the dataset is small (~60 samples) and synthetic, so these numbers are
a sanity check only, not a clinical claim.
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

DATA = "attention_learning_data.csv"
FEATURES = ["mean_bpm", "max_bpm", "min_bpm", "avg_diff", "diff_std",
            "up_count", "down_count", "slope"]


def main():
    df = pd.read_csv(DATA)
    X = df[FEATURES].values
    y = df["label"].values

    print("=" * 50)
    print(f"samples        : {len(y)}")
    counts = dict(zip(*np.unique(y, return_counts=True)))
    print(f"class counts   : {counts}  (0=unfocused, 1=focused)")

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn_scores = cross_val_score(knn, X, y, cv=kf, scoring="accuracy")
    print("=" * 50)
    print(f"KNN(k=5) accuracy : {knn_scores.mean():.3f} +/- {knn_scores.std():.3f}  (5-fold CV)")

    dummy_strat = DummyClassifier(strategy="stratified", random_state=42)
    d_scores = cross_val_score(dummy_strat, X, y, cv=kf, scoring="accuracy")
    print(f"stratified-random : {d_scores.mean():.3f} +/- {d_scores.std():.3f}")

    dummy_maj = DummyClassifier(strategy="most_frequent")
    m_scores = cross_val_score(dummy_maj, X, y, cv=kf, scoring="accuracy")
    print(f"majority-class    : {m_scores.mean():.3f}")

    y_pred = cross_val_predict(knn, X, y, cv=kf)
    print("=" * 50)
    print("confusion matrix (rows=true, cols=predicted):")
    print(confusion_matrix(y, y_pred))
    print("-" * 50)
    print(classification_report(y, y_pred, target_names=["unfocused", "focused"], digits=3))
    print(f"overall accuracy  : {accuracy_score(y, y_pred):.3f}")
    print("=" * 50)
    print("NOTE: small + synthetic dataset -> indicative only.")


if __name__ == "__main__":
    main()
