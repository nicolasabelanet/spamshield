from sklearn.metrics import average_precision_score


def ap_on_spam(estimator, X, y) -> float:
    """
    Custom scorer for GridSearchCV: Average Precision using P(spam|x).
    Works with string labels ('ham','spam'); no relabeling required.
    """
    proba = estimator.predict_proba(X)
    spam_idx = list(estimator.classes_).index("spam")
    return float(
        average_precision_score(y_true=y, y_score=proba[:, spam_idx], pos_label="spam")  # type: ignore
    )
