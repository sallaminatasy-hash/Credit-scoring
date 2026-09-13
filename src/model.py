"""
Model training module for credit scoring.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV


def get_models():
    """Return a dict of models to compare."""
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        ),
    }
    return models


def train_model(model, X_train, y_train):
    """Fit a model."""
    model.fit(X_train, y_train)
    return model


def tune_xgboost(X_train, y_train):
    """Grid search for XGBoost."""
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
    }
    grid = GridSearchCV(
        XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
        param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_
