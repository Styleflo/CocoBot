# Projet Final - Apprentissage Automatique pour Données Massives

### Système de recommandation et de recherche sémantique pour les produits Chanel, utilisant des embeddings de texte et d'image.

Ce projet a été réalisé dans le cadre du cours d'Apprentissage automatique pour les données massives. Il propose une application interactive permettant de rechercher et de recommander des produits de luxe (Chanel) en utilisant des techniques avancées de traitement du signal (images) et du langage naturel (texte).

## 🚀 Fonctionnalités

L'application Streamlit propose trois modes de recherche :
1. **Recherche par image** : Téléchargez une image pour trouver des produits visuellement similaires grâce à l'extracteur de caractéristiques **ResNet50**.
2. **Recherche par texte** : Entrez une description pour trouver des produits correspondants via des embeddings textuels générés par **Sentence-Transformers**.
3. **Recherche combinée** : Fusionne les informations visuelles et textuelles pour une recommandation ultra-précise.

## 🛠️ Technologies Utilisées

- **Python 3.x**
- **Streamlit** : Interface utilisateur web.
- **TensorFlow / Keras** : Utilisation du modèle pré-entraîné **ResNet50** pour les images.
- **Sentence-Transformers** : Modèle `all-MiniLM-L6-v2` pour les descriptions textuelles.
- **Pandas & NumPy** : Manipulation des données et calculs matriciels.
- **Scipy** : Calcul de la similarité cosinus.

## 📂 Structure du Projet

- `StreamlitApp.py` : Code principal de l'application web.
- `ProjetFinal_PF.ipynb` : Notebook de préparation des données, extraction des embeddings et analyses.
- `data_emb.csv` : Embeddings textuels pré-calculés.
- `resnet50_image_embeddings.parquet` : Embeddings d'images pré-calculés.
- `requirements.txt` : Liste des dépendances Python.

## ⚙️ Installation et Utilisation

1. **Cloner le dépôt** :
   ```bash
   git clone <url-du-depot>
   cd ProjetFinal
   ```

2. **Installer les dépendances** :
   ```bash
   pip install -r requirements.txt
   ```

3. **Lancer l'application** :
   ```bash
   streamlit run StreamlitApp.py
   ```

## 👥 Équipe
- Yann
- Alix
- Florian
- Thomas

---
*Projet réalisé dans le cadre du module Apprentissage automatique pour les données massives (UQAC).*
