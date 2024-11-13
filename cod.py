# Importation des bibliothèques nécessaires
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt

# Chargement des données
cod_data = pd.read_csv('cod.csv')

# Création d'une colonne cible binaire 'high_prestige' pour indiquer si le prestige est supérieur à 50
cod_data['high_prestige'] = (cod_data['prestige'] > 50).astype(int)

# Sélection des caractéristiques pertinentes pour l'arbre de décision
features = cod_data[['kills', 'kdRatio', 'timePlayed', 'level']]
target = cod_data['high_prestige']

# Traitement des valeurs manquantes dans 'kdRatio' et 'timePlayed' en les remplaçant par la médiane
features.loc[:, 'kdRatio'] = features['kdRatio'].fillna(features['kdRatio'].mean())
features.loc[:, 'timePlayed'] = features['timePlayed'].fillna(features['timePlayed'].mean())



# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Création et entraînement du modèle d'arbre de décision
clf_cod = DecisionTreeClassifier(max_depth=3, random_state=42)  # Limiter la profondeur pour simplifier l'interprétation
clf_cod.fit(X_train, y_train)

# Visualisation de l'arbre de décision
plt.figure(figsize=(12, 8))
tree.plot_tree(clf_cod, feature_names=features.columns, class_names=["Low Prestige", "High Prestige"], filled=True)
plt.show()
