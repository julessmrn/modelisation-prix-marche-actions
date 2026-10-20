# Modélisation du prix de marché d'actions : économétrie et machine learning

Prédiction et explication du prix de marché de 25 000 entreprises à partir de variables financières, de marché, de risque et macroéconomiques. Le projet compare la régression linéaire (OLS) à des méthodes d'apprentissage statistique, avec Python.

Projet réalisé en équipe avec Vianney Mathon et Dorian Fernandez, dans le cadre du Master Ingénierie des Risques Économiques et Financiers – Economic Risk and Data Sciences (Université de Bordeaux).

## Objectif

- Identifier les déterminants du prix de marché : rentabilité, risque, secteur, notation.
- Déterminer quel modèle prédit le mieux le prix sur des données qu'il n'a jamais vues.

## Données

25 000 entreprises et 30 variables explicatives, dont 5 qualitatives, avec environ 8 % de valeurs manquantes sur chaque variable (voir `data/README.md`).

## Méthodologie

### 1. Traitement des valeurs manquantes

Le traitement s'appuie sur la logique financière plutôt que sur une moyenne globale :
- **Identités comptables :** les actifs, la dette et les capitaux propres manquants sont reconstitués par l'identité actifs = capitaux propres + dette. Le ROA et le ROE sont reconstitués l'un par l'autre, via le résultat net.
- **Comparables sectoriels et de taille :** le chiffre d'affaires et le cash-flow sont imputés par un ratio sur actifs médian du secteur, ou du décile de taille si le secteur est inconnu. Le bêta, le dividende, l'EPS et le P/E ratio sont imputés par la médiane du secteur.
- **Variables de marché et de risque :**
  - les variables de marché sont imputées par la médiane des entreprises du même secteur et de la même taille ;
  - la VaR et l'expected shortfall par la médiane des entreprises au bêta comparable ;
  - le spread de crédit par la médiane des entreprises au levier comparable.
- **Pas de fuite de la cible :** aucune imputation n'utilise le prix de marché. L'identité PER = prix / EPS a volontairement été écartée, car elle aurait fait entrer la cible dans les variables explicatives.

### 2. Protocole d'évaluation

- **Découpage unique :** 80 % des données pour l'apprentissage et 20 % pour le test, avec une graine fixée.
- **Décisions sur l'apprentissage :** la sélection de variables et le réglage des hyperparamètres (validation croisée à 5 plis) se font uniquement sur l'échantillon d'apprentissage.
- **Même test pour tous :** les trois modèles sont comparés sur le même échantillon de test.

### 3. Modèles

- **OLS complet :** 46 variables après encodage des variables qualitatives, avec diagnostics (Durbin-Watson, Jarque-Bera, conditionnement).
- **Sélection de variables :** forward et backward pilotés par les p-values, et Lasso avec validation croisée. Le modèle final est un OLS à 17 variables.
- **KNN et arbre de décision,** réglés par validation croisée (GridSearchCV).

## Résultats sur l'échantillon de test

| Modèle | R² test | RMSE test |
|---|---|---|
| **OLS (17 variables)** | **0,695** | **0,0671** |
| Arbre de décision (profondeur 5) | 0,688 | 0,0679 |
| KNN (K = 12) | 0,547 | 0,0818 |

- **L'OLS est le meilleur modèle,** à la fois pour la prédiction et pour l'interprétation. Sur 5 découpages aléatoires différents, il fait au moins aussi bien que l'arbre à chaque fois.
- **Le modèle généralise bien :** son R² passe de 0,713 sur l'apprentissage à 0,695 sur le test.
- **Les principaux déterminants du prix sont :**
  - l'indice de volatilité, à effet négatif, de loin le plus fort ;
  - la rentabilité (EPS, ROE, ROA) ;
  - le spread de crédit, à effet négatif ;
  - le momentum à 12 mois ;
  - le secteur, avec une prime pour la technologie et la santé ;
  - la notation, avec une prime pour AAA et AA et une décote pour BB.
- **Les variables macroéconomiques** (inflation, taux, croissance) n'ont pas d'effet significatif en coupe instantanée.

## Limites

- Les médianes d'imputation sont calculées sur l'ensemble des données, avant le découpage. Elles n'utilisent jamais la cible, mais une approche entièrement rigoureuse les calculerait sur l'apprentissage seul.
- Les résidus ne suivent pas une loi normale (queues épaisses) : les tests de significativité sont à interpréter avec prudence.

## Reproduire l'analyse

```bash
pip install -r requirements.txt
jupyter notebook modelisation_prix_marche.ipynb
```

Placer au préalable les données dans `data/` (voir `data/README.md`).

## Structure du dépôt

```
├── modelisation_prix_marche.ipynb   # Notebook complet
├── requirements.txt                 # Dépendances Python
└── data/
    └── README.md                    # Description des données
```

## Outils

Python, pandas, NumPy, statsmodels, scikit-learn, Matplotlib, seaborn.

## Auteurs

- **Jules Marin** – [LinkedIn](https://www.linkedin.com/in/jules-marin-64611035a/)
- **Vianney Mathon**
- **Dorian Fernandez**
