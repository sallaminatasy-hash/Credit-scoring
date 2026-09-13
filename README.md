# Credit Scoring - Prédiction du risque de défaut

## Objectif
Prédire la probabilité qu'un client fasse défaut sur son crédit à partir de données démographiques et financières.

## Dataset
[German Credit Data - UCI](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))  
- 1000 clients
- 20 features (statut du compte, durée du crédit, historique, montant, âge, etc.)
- Variable cible binaire : 0 = bon crédit (70%), 1 = défaut (30%)

## Stack technique
- **Python** : pandas, numpy, scikit-learn, XGBoost
- **ML** : Logistic Regression, Random Forest, XGBoost
- **Gestion du déséquilibre** : SMOTE (imbalanced-learn)
- **Interprétabilité** : SHAP, Feature Importance
- **Visualisation** : matplotlib, seaborn

## Résultats

| Modèle | ROC-AUC |
|---|---|
| Logistic Regression | **0.7967** |
| XGBoost | 0.7723 |
| XGBoost Tuned | 0.7668 |

**Meilleurs paramètres XGBoost** : `learning_rate=0.1, max_depth=7, n_estimators=100`

### Interprétation
Sur ce dataset de petite taille (1000 lignes), le **Logistic Regression** obtient les meilleures performances. C'est un résultat classique : les modèles complexes comme XGBoost ont tendance à sur-apprendre quand les données sont limitées.


## Installation
```bash
git clone https://github.com/sallaminatasy-hash/Credit-scoring.git
cd Credit-scoring
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/01_credit_scoring.ipynb


