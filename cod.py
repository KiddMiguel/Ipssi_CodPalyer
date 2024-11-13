# Importation des bibliothèques nécessaires
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement des données
cod_data = pd.read_csv('cod.csv')

# Création d'une colonne cible binaire 'high_prestige' pour indiquer si le prestige est supérieur à 50
cod_data['high_prestige'] = (cod_data['prestige'] > 50).astype(int)

# Sélection des caractéristiques pertinentes pour l'arbre de décision
features = cod_data[['kills', 'kdRatio', 'timePlayed', 'level']]
target = cod_data['high_prestige']

# Création d'un DataFrame combiné pour éviter SettingWithCopyWarning
features_with_target = features.copy()
features_with_target['high_prestige'] = target

# Traitement des valeurs manquantes dans 'kdRatio' et 'timePlayed' en les remplaçant par la moyenne
features_with_target.loc[:, 'kdRatio'] = features_with_target['kdRatio'].fillna(features_with_target['kdRatio'].mean())
features_with_target.loc[:, 'timePlayed'] = features_with_target['timePlayed'].fillna(features_with_target['timePlayed'].mean())

# Matrice de graphiques de dispersion avec seaborn
sns.pairplot(features_with_target, hue='high_prestige', palette="coolwarm")
plt.suptitle("Matrice de Graphiques de Dispersion par Niveau de Prestige", y=1.02)
plt.show()

# Création et entraînement du modèle d'arbre de décision sur l’ensemble complet
clf_cod = DecisionTreeClassifier(max_depth=3, random_state=42)
clf_cod.fit(features, target)

# Visualisation de l'arbre de décision
plt.figure(figsize=(12, 8))
tree.plot_tree(clf_cod, feature_names=features.columns, class_names=["Low Prestige", "High Prestige"], filled=True)
plt.show()
