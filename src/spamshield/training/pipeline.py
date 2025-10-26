from typing import TYPE_CHECKING, Any

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline

from spamshield.training import scoring

if TYPE_CHECKING:
    import pandas as pd


def small_grid() -> dict[str, Any]:
    """
    Return a small hyperparameter grid for GridSearchCV over the
    TF-IDF + LogisticRegression spam classifier pipeline.

    Hyperparameters
    ---------------
    tfidf__min_df:
        Ignore tokens that appear in fewer than `min_df` training documents.
        Filters out extremely rare tokens / typos.

    tfidf__ngram_range:
        Controls whether we include just unigrams (1-word terms) or
        unigrams + bigrams (2-word phrases). Bigrams capture patterns like
        "click here", which are useful for spam.

    clf__C:
        Inverse regularization strength for LogisticRegression.
        Smaller C -> stronger regularization -> simpler model.
        Larger C -> weaker regularization -> more flexible model.

    clf__penalty:
        Regularization type. We use "l2" (ridge-style), which smoothly
        shrinks large weights instead of zeroing them out.

    clf__solver:
        Optimization algorithm. "lbfgs" is stable and handles
        multiclass / dense features well.
    """
    return {
        "tfidf__min_df": [1, 2, 3],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__C": [0.5, 1.0, 2.0],
        "clf__penalty": ["l2"],
        "clf__solver": ["lbfgs"],
    }


def train_pipeline(
    X_train: pd.Series, y_train: pd.Series
) -> tuple[Pipeline, dict[str, Any]]:
    """
    Train and return a calibrated spam text classification pipeline.

    Steps
    -----
    1. Builds a base Pipeline with:
        - "tfidf": TfidfVectorizer()
        - "clf": LogisticRegression(max_iter=400, class_weight="balanced")
            * class_weight="balanced" -> reweights classes by inverse frequency
            (helps with spam vs ham imbalance)

    2. Runs GridSearchCV:
        - param_grid = small_grid() (simple hyperparameter grid)
        - scoring = scoring.ap_on_spam (average precision on spam text labels)
        - cv = 5 (5 fold cross validation)
        Produces `best_estimator_`, the best-performing pipeline.

    3. Freezes the best classifier:
        - FrozenEstimator(best_estimator["clf"]) wraps the trained classifier
        so it cannot be refit.

    4. Calibrates probabilities:
        - CalibratedClassifierCV(..., method="sigmoid") learns a sigmoid mapping
        (Platt scaling) from the frozen classifier's logits to
        well-calibrated probabilities.

    5. Builds the final inference pipeline:
        - final_pipe = make_pipeline(best tfidf, calibrated classifier)
        - final_pipe.fit(X_train, y_train) fits:
            * the TfidfVectorizer vocabulary / IDF weights
            * the calibration layer
        The frozen classifier's learned weights are NOT changed here.

    Parameters
    ----------
    X_train : pd.Series
        The spam training messages (text).
    y_train : pd.Series
        The corresponding spam/ham labels for the `X_train` messages.

    Returns
    -------
    final_pipe:
        A ready-to-use Pipeline that does TF-IDF -> calibrated classifier. Use this
        for infererence.
    best_params:
        The best hyperparameters found by GridSearchCV.
    """

    # This pipeline ensures the entire preprocessing (vectorization + classification)
    # can be treated as a single model object.
    base_pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression(max_iter=400, class_weight="balanced")),
        ]
    )

    grid_search = GridSearchCV(
        base_pipeline,
        param_grid=small_grid(),
        scoring=scoring.ap_on_spam,
        cv=5,
        n_jobs=-1,
        verbose=0,
    )

    # Fit the grid search and extract the best-performing pipeline.
    grid_search.fit(X_train, y_train)
    best_estimator = grid_search.best_estimator_

    # Make the best classifier read-only so it will not be retrained during calibration.
    frozen_clf = FrozenEstimator(best_estimator["clf"])

    # Learn a sigmoid mapping to turn raw scores into calibrated probabilities.
    calibrated = CalibratedClassifierCV(frozen_clf, method="sigmoid")

    # Build the final pipeline: vectorizer -> calibrated classifier.
    final_pipe = make_pipeline(best_estimator["tfidf"], calibrated)

    # Fit the vectorizer vocab/IDF and the calibration layer.
    # The frozen classifier itself is not refit here.
    final_pipe.fit(X_train, y_train)

    return final_pipe, grid_search.best_params_
