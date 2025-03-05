import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

###############CHARGEMENT DES DONNEES##################################
def load_data():
    # Charger le fichier CSV localement
    datafile=pd.read_csv('MathE_dataset.csv', delimiter=";", encoding='cp1252')
    return datafile
datafile = load_data()

###############PRERAITEMENT DES DONNEES##################################
datafile['réussite'] = datafile['Type of Answer']

#ENCODAGE DES VARIABLES CATEGORIELLES (NON ORDONNEES: ONE HOT ENCODING)
datafile = pd.get_dummies(datafile, columns=['Student Country'], drop_first=True)

#ENCODAGE DES VARIABLES CATEGORIELLES (ORDONNEES: LABEL ENCODING)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
datafile['Question Level'] = le.fit_transform(datafile['Question Level'])
datafile['Topic'] = le.fit_transform(datafile['Topic'])
datafile['Subtopic'] = le.fit_transform(datafile['Subtopic'])

#Séparation des variables explicatives (X) et de la variable cible (y)
X = datafile.drop(['Student ID','Question ID','Type of Answer', 'Keywords','réussite'], axis=1) 
# y : Variable cible
y = datafile['réussite']

#Standardisation des données 
from sklearn.preprocessing import StandardScaler

#Séparation des données en ensembles d’entraînement et de test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X[['Topic', 'Subtopic']] = scaler.fit_transform(X[['Topic', 'Subtopic']])

#Entrainement du modèle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Entraîner le modèle
model = LogisticRegression()
model.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
y_pred = model.predict(X_test)

# Évaluer les performances du modèle
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')


###############INTERFACE DE SAISIE DES DONNEES##################################
st.sidebar.title("Prédiction")
menu = st.sidebar.radio("Sélectionner une option", ["ACCEUIL", "PREDICTION", "A PROPOS DE NOUS"])
if menu == "ACCEUIL":
    st.title("ACCEUIL (visualisation du jeu de données")
    st.write("Bienvenue sur notre API de prédiction des especes de fleurs entrainée avec le jeu de donées <<MathE_Dataset.csv>>")
    st.write("")
    st.subheader("Voici un aperçu du jeu de données MathE dataset (les 05 premières lignes) :")
    st.dataframe(datafile.head()) 
    st.subheader("Description Générale :")
    st.dataframe(datafile.describe()) 
    st.write("")
    st.write("Rendez-vous au menu préddiction, entrer les dimension que vous voulez, et notre plateforme se chargera de vous prédire de quelle espèces cette fleurs appatiens.")
    
# Formulaire pour saisir les dimensions
elif menu == "PREDICTION":
    st.title("PREDICTION DES NOTES DES ELEVES")
    # Affichage de la précision du modèle
    
# Prédiction du type de fleur
    if st.button("Prédire"):
        # Créer un DataFrame à partir des entrées de l'utilisateur
        user_input = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], 
                              columns=["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"])
    # Prédire le type de fleur
     

elif menu == "A PROPOS DE NOUS":
    st.write("Etudiant à l'ENS de Yaoundé, filière informatique")
    st.write("@Waffo")
