import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

def load_data():
    # Charger le fichier CSV localement
    datafile=pd.read_csv('MathE_dataset.csv', delimiter=";", encoding='cp1252')
    return df
datafile = load_data()

st.write('hello')
