""" import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Charger les données JSON
file_path = 'DATA.json'  # Remplacez par votre chemin
with open(file_path, 'r') as file:
    data = json.load(file)

# Convertir les collections en DataFrame
medicine = pd.json_normalize(data['Medicine'])
composition = pd.json_normalize(data['Medicine_composition'])
family = pd.json_normalize(data['Medicine_family'])
price = pd.json_normalize(data['Medicine_price'])

# Vérifier que les colonnes nécessaires existent
required_columns = ['name', 'valuable_pricing', 'hospital_pricing']
for col in required_columns:
    if col not in price.columns:
        raise KeyError(f"La colonne '{col}' est manquante dans le DataFrame 'price'")

# Fusionner les DataFrames sur le 'name'
merged_data = pd.merge(medicine, composition, on='name', how='inner')
merged_data = pd.merge(merged_data, family, on='name', how='inner')
merged_data = pd.merge(merged_data, price, on='name', how='inner')

# Gérer les colonnes de type liste : composition et family
merged_data['composition'] = merged_data['composition'].apply(lambda x: ', '.join([str(i) for i in x if i is not None]) if isinstance(x, list) else x)
merged_data['family'] = merged_data['family'].apply(lambda x: ', '.join([str(i) for i in x if i is not None]) if isinstance(x, list) else x)

# Préparer les données pour l'entraînement
X = merged_data[['dose', 'family', 'composition', 'presentation']]
y = merged_data['valuable_pricing']

# Encoder les variables catégorielles
label_encoders = {}
for column in ['family', 'composition', 'presentation']:
    le = LabelEncoder()
    X.loc[:, column] = le.fit_transform(X[column])
    label_encoders[column] = le

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner le modèle
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Faire des prédictions
y_pred = model.predict(X_test)

# Évaluer le modèle
print("\nRapport de classification :\n", classification_report(y_test, y_pred))
print("Précision : ", accuracy_score(y_test, y_pred))

# Sauvegarder le modèle pour une utilisation future
joblib.dump(model, 'classification_model.pkl')
print("\nModèle sauvegardé sous 'classification_model.pkl'")

# Exemple de prédiction
sample_data = pd.DataFrame([[250, 0, 1, 2]], columns=['dose', 'family', 'composition', 'presentation'])
prediction = model.predict(sample_data)
print("\nPrédiction sur un exemple :", prediction) """

import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
import re

# Charger les données JSON
file_path = 'DATA.json'  # Remplacez par votre chemin
with open(file_path, 'r') as file:
    data = json.load(file)

# Convertir les collections en DataFrame
medicine = pd.json_normalize(data['Medicine'])
composition = pd.json_normalize(data['Medicine_composition'])
family = pd.json_normalize(data['Medicine_family'])
price = pd.json_normalize(data['Medicine_price'])

# Vérifier que les colonnes nécessaires existent
required_columns = ['name', 'valuable_pricing', 'hospital_pricing']
for col in required_columns:
    if col not in price.columns:
        raise KeyError(f"La colonne '{col}' est manquante dans le DataFrame 'price'")

# Fusionner les DataFrames sur le 'name'
merged_data = pd.merge(medicine, composition, on='name', how='inner')
merged_data = pd.merge(merged_data, family, on='name', how='inner')
merged_data = pd.merge(merged_data, price, on='name', how='inner')

# Gérer les colonnes de type liste : composition et family
merged_data['composition'] = merged_data['composition'].apply(lambda x: ', '.join([str(i) for i in x if i is not None]) if isinstance(x, list) else x)
merged_data['family'] = merged_data['family'].apply(lambda x: ', '.join([str(i) for i in x if i is not None]) if isinstance(x, list) else x)

# Nettoyer et convertir la colonne 'dose'
def clean_dose(value):
    if pd.isnull(value):
        return 0.0  # Valeur par défaut pour les doses manquantes
    match = re.search(r"(\d+(\.\d+)?)", str(value))  # Extraire le nombre
    return float(match.group(1)) if match else 0.0

merged_data['dose'] = merged_data['dose'].apply(clean_dose)

# Convertir la cible (y) en classes discrètes
def classify_pricing(value):
    if value < 100:
        return "Low"
    elif 100 <= value <= 500:
        return "Medium"
    else:
        return "High"

merged_data['valuable_pricing'] = merged_data['valuable_pricing'].apply(classify_pricing)

# Supprimer les lignes avec des NaN dans les colonnes importantes
merged_data = merged_data.dropna(subset=['dose', 'valuable_pricing', 'family', 'composition'])

# Préparer les données pour l'entraînement
X = merged_data[['dose', 'family', 'composition', 'presentation']]
y = merged_data['valuable_pricing']

# Encoder les variables catégorielles
label_encoders = {}
for column in ['family', 'composition', 'presentation']:
    le = LabelEncoder()
    X.loc[:, column] = le.fit_transform(X[column].astype(str))  # Utiliser loc pour éviter les warnings
    label_encoders[column] = le

# Encoder la variable cible (y)
le_y = LabelEncoder()
y = le_y.fit_transform(y)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner le modèle
model = RandomForestClassifier(n_estimators=100, random_state=42,class_weight='balanced')
model.fit(X_train, y_train)

# Évaluer le modèle
y_pred = model.predict(X_test)
print("\nRapport de classification :\n", classification_report(y_test, y_pred, target_names=le_y.classes_))
print("Précision : ", accuracy_score(y_test, y_pred))

# Sauvegarder le modèle pour une utilisation future
joblib.dump(model, 'classification_model.pkl')
print("\nModèle de classification sauvegardé sous 'classification_model.pkl'")


test_data = pd.DataFrame([{
    "dose": 300,  # Exemple de dose
    "family": "Analgésique",  # Remplacez par une valeur de votre dataset
    "composition": "Paracétamol",  # Remplacez par une valeur de votre dataset
    "presentation": "Boite de 10"  # Remplacez par une valeur de votre dataset
}])

# Encoder les colonnes de test avec les encodeurs existants
for column in ['family', 'composition', 'presentation']:
    if column in test_data.columns and column in label_encoders:
        test_data[column] = label_encoders[column].transform(test_data[column])

# Faire une prédiction
test_prediction = model.predict(test_data)
test_class = le_y.inverse_transform(test_prediction)
print("\nPrédiction pour l'exemple donné :", test_class[0])