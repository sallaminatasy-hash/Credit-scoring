"""
Preprocessing module for credit scoring dataset.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


def load_data(path: str = "data/german_credit.csv") -> pd.DataFrame:
    """Load the German Credit dataset."""
    return pd.read_csv(path)


def split_features_target(df: pd.DataFrame, target_col: str = "default"):
    """Split into X and y."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a preprocessing pipeline for numeric + categorical features."""
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def prepare_data(df: pd.DataFrame, target_col: str = "default",
                 test_size: float = 0.2, random_state: int = 42):
    """Full pipeline: split, preprocess, and apply SMOTE on training set."""
    X, y = split_features_target(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    preprocessor = build_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Handle class imbalance
    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train_processed, y_train)

    return X_train_res, X_test_processed, y_train_res, y_test, preprocessor