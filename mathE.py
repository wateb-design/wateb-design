import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

def load_data():
    # Charger le fichier CSV localement
    datafile=pd.read_csv('MathE_dataset.csv', delimiter=";", encoding='cp1252')
    return datafile
datafile = load_data()

st.write('hello')

#INTERFACE DE SAISIE DES DONNEES

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
    st.subheader("Description Générale :")
    st.dataframe(datafile.info()) 
    st.write("")
    st.write("Rendez-vous au menu préddiction, entrer les dimension que vous voulez, et notre plateforme se chargera de vous prédire de quelle espèces cette fleurs appatiens.")
    
# Formulaire pour saisir les dimensions
elif menu == "PREDICTION":
    st.title("PREDICTION DES FLEURS")
    # Affichage de la précision du modèle
    st.write(f"Précision du modèle : {accuracy:.2f}")
    st.write("Entrez les dimensions de la fleur :")

# Prédiction du type de fleur
    if st.button("Prédire"):
        # Créer un DataFrame à partir des entrées de l'utilisateur
        user_input = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], 
                              columns=["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"])
    # Prédire le type de fleur
     

elif menu == "A PROPOS DE NOUS":
    st.write("Etudiant à l'ENS de Yaoundé, filière informatique")
    st.write("@Waffo")
