import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


class SimpleCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._punct = re.compile(r"[^\w\s]", flags=re.UNICODE)
        self._spaces = re.compile(r"\s+", flags=re.UNICODE)
        self.stopwords = set(self._default_stopwords())

    def _default_stopwords(self) -> List[str]:
        return [
            "i", "a", "ali", "pa", "te", "ni", "niti", "ili", "da", "jer", "dok", "ne", "je", "su", "smo", "sam", "ste",
            "u", "na", "o", "od", "do", "sa", "za", "po", "pri", "bez", "ka", "kod", "preko", "posle", "nakon", "kroz",
            "se", "će", "ću", "ćeš", "ćemo", "ćete", "bi", "bih", "bismo", "biste", "bili", "bila", "bilo",
            "taj", "ta", "to", "ovaj", "ova", "ovo", "onaj", "ona", "ono", "koji", "koja", "koje"
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        s = pd.Series(X).astype(str)
        s = s.str.lower()
        s = s.apply(lambda x: self._punct.sub(" ", x))
        s = s.apply(lambda x: self._spaces.sub(" ", x).strip())
        s = s.apply(lambda x: " ".join(w for w in x.split() if w not in self.stopwords))
        return s


def load_train_json(path: Path) -> pd.DataFrame:
    df = pd.read_json(path)
    need = {"Naslov", "Kategorija"}
    if not need.issubset(df.columns):
        raise ValueError("Train JSON mora imati kolone 'Naslov' i 'Kategorija'.")
    return df


def load_test_json(path: Path) -> pd.DataFrame:
    df = pd.read_json(path)
    if "Naslov" not in df.columns:
        raise ValueError("Test JSON mora imati kolonu 'Naslov'.")
    return df


def build_search_space() -> List[Tuple[str, Pipeline, Dict[str, List[Any]]]]:
    cleaner = SimpleCleaner()
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=20000,
        lowercase=False,
        norm="l2"
    )

    svc_pipe = Pipeline([
        ("clean", cleaner),
        ("tfidf", tfidf),
        ("clf", LinearSVC())
    ])
    svc_grid = {
        "tfidf__min_df": [2, 3, 5],
        "tfidf__max_df": [0.9, 1.0],
        "clf__C": [0.5, 1.0, 2.0],
    }

    logreg_pipe = Pipeline([
        ("clean", cleaner),
        ("tfidf", tfidf),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs", penalty="l2"))
    ])
    logreg_grid = {
        "tfidf__min_df": [2, 3, 5],
        "tfidf__max_df": [0.9, 1.0],
        "clf__C": [0.5, 1.0, 2.0],
    }

    return [
        ("LinearSVC", svc_pipe, svc_grid),
        ("LogisticRegression", logreg_pipe, logreg_grid),
    ]


def run_training(train_df: pd.DataFrame, cv_splits: int = 5, random_state: int = 42) -> Pipeline:
    X = train_df["Naslov"]
    y = train_df["Kategorija"]

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    best_name, best_cv, best_est = None, -np.inf, None
    for name, pipe, grid in build_search_space():
        search = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            scoring="f1_macro",
            cv=skf,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X, y)
        if search.best_score_ > best_cv:
            best_name, best_cv, best_est = name, search.best_score_, search.best_estimator_

    print(f"[INFO] Najbolji model (CV f1_macro): {best_name} = {best_cv:.4f}", file=sys.stderr)

    best_est.fit(X, y)
    return best_est

def run_prediction(model: Pipeline, test_df: pd.DataFrame) -> None:
    if "Kategorija" not in test_df.columns:
        raise ValueError("Za računanje macro F1 potrebna je kolona 'Kategorija' u test skupu.")

    y_true = test_df["Kategorija"]
    y_pred = model.predict(test_df["Naslov"])

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"{macro_f1:.6f}")

def local_eval(train_df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> None:
    # stratifikovano zbog balansa klasa
    X_tr, X_val = train_test_split(
        train_df,
        test_size=test_size,
        random_state=random_state,
        stratify=train_df["Kategorija"]
    )
    model = run_training(X_tr, cv_splits=5, random_state=random_state)

    y_true = X_val["Kategorija"]
    y_pred = model.predict(X_val["Naslov"])
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, digits=3)
    print("\n=== EVALUACIJA (hold-out) ===", file=sys.stderr)
    print(report, file=sys.stderr)
    print(f"Macro F1: {macro_f1:.4f}", file=sys.stderr)

    # Ispiši predikcije za val deo na STDOUT (radi forme/konzistentnosti)
    run_prediction(model, X_val)


if __name__ == '__main__':
    if len(sys.argv) == 3:
        train_path = Path(sys.argv[1])
        test_path = Path(sys.argv[2])

        train_df = load_train_json(train_path)
        test_df = load_test_json(test_path)

        model = run_training(train_df)
        run_prediction(model, test_df)
    elif len(sys.argv) == 2:
        train_path = Path(sys.argv[1])
        train_df = load_train_json(train_path)
        local_eval(train_df, test_size=0.2, random_state=42)
