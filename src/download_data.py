"""
Télécharge et convertit le German Credit Data en CSV.
"""
import pandas as pd
import urllib.request
import os


# Colonnes du dataset German Credit (UCI)
COLUMNS = [
    "checking_account_status", "duration_months", "credit_history", "purpose",
    "credit_amount", "savings_account", "employment_since", "installment_rate",
    "personal_status_sex", "other_debtors", "residence_since", "property",
    "age", "other_installment_plans", "housing", "existing_credits",
    "job", "num_dependents", "telephone", "foreign_worker", "default"
]

# URL du dataset
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"


def download_and_convert():
    """Télécharge le fichier et le convertit en CSV."""
    os.makedirs("data", exist_ok=True)

    print("Téléchargement du dataset...")
    urllib.request.urlretrieve(URL, "data/german.data")

    print("Conversion en CSV...")
    df = pd.read_csv("data/german.data", sep=" ", header=None, names=COLUMNS)

    # La variable cible : 1 = bon crédit, 2 = mauvais crédit
    # On la convertit en 0/1 (1 = défaut)
    df["default"] = df["default"].map({1: 0, 2: 1})

    df.to_csv("data/german_credit.csv", index=False)
    print(f"Dataset sauvegardé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    print(df.head())
    print(f"\nDistribution de la cible :\n{df['default'].value_counts(normalize=True)}")


if __name__ == "__main__":
    download_and_convert()