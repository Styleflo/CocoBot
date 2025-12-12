import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import ast
import requests
from io import BytesIO

# Chargement des données
articles = pd.read_csv("data_emb.csv")
# Convertir les embeddings de str -> np.array
articles['embeddings'] = articles['embeddings'].apply(lambda x: np.array(ast.literal_eval(x)))

# Fonction de similarité
def compute_similarity(query_emb, df, top_k=10):
    query_emb = np.array(query_emb).flatten()
    df = df.copy()

    # Calcul de la similarité
    df['similarity'] = df['embeddings'].apply(lambda emb: 1 - cosine(query_emb, np.array(emb).flatten()))

    # Trier par similarité décroissante
    df_sorted = df.sort_values("similarity", ascending=False)

    # Garder seulement le premier article unique par product_code
    df_unique = df_sorted.drop_duplicates(subset='product_code', keep='first')

    # Sélectionner les top_k
    df_topk = df_unique.head(top_k)

    sims = df_topk['similarity'].values
    order = df_topk.index.tolist()
    return sims, order


def encode_image(image: Image.Image):
    return np.random.rand(512)

def encode_text(text: str):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(text, show_progress_bar=True)
    return embeddings


# Fonction utilitaire pour charger image depuis URL
def load_image_from_url(url):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        return img
    except:
        st.error(f"Impossible de charger l'image depuis {url}")
        return None


# Interface Streamlit

st.markdown(
    """
    <h1 style='text-align: center;'>
        Plateforme de recherche de similarité Channel
    </h1>
    """,
    unsafe_allow_html=True
)

option = st.sidebar.selectbox(
    "Choisir une option",
    ["Recherche par image", "Recherche par texte", "Recherche combinée"]
)

# Recherche par image
if option == "Recherche par image":

    st.header("Recherche basée sur une image")
    uploaded_file = st.file_uploader("Charge une image", type=["jpg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Image fournie", width=300)

        query_emb = encode_image(img)
        sims, order = compute_similarity(query_emb, articles)

        st.subheader("🔝 Top 10 articles visuellement similaires")
        for idx in order[:10]:
            row = articles.iloc[idx]
            col1, col2 = st.columns([1, 3])
            with col1:
                img_row = load_image_from_url(row["imageurl"])
                if img_row:
                    st.image(img_row, width=100)
            with col2:
                st.write(f"**{row['title']}**")
                st.write(f"*Code du produit : {row['product_code']}*")
                st.write(f"Catégorie: {row['category1_code']} / {row['category2_code']}")
                st.write(f"Prix : {row['price']} €")


# Recherche par texte
elif option == "Recherche par texte":

    st.header("Recherche basée sur un texte")
    query = st.text_input("Entrez une description")

    if query:
        query_emb = encode_text(query)
        sims, order = compute_similarity(query_emb, articles)

        st.subheader("Top 10 articles similaires en termes de description")
        for idx in order[:10]:
            row = articles.iloc[idx]
            col1, col2 = st.columns([1, 3])
            with col1:
                img_row = load_image_from_url(row["imageurl"])
                if img_row:
                    st.image(img_row, width=100)
            with col2:
                st.write(f"**{row['title']}**")
                st.write(f"*Code du produit : {row['product_code']}*")
                st.write(f"Catégorie: {row['category1_code']} / {row['category2_code']}")
                st.write(f"Prix : {row['price']} €")


# Recherche combinée
elif option == "Recherche combinée":

    st.header("Recherche combinée image + texte")
    uploaded_file = st.file_uploader("Charge une image", type=["jpg", "png"])
    query_text = st.text_input("Entrez une description")

    if uploaded_file and query_text:
        img = Image.open(uploaded_file)
        st.image(img, width=300)

        emb_img = encode_image(img)
        emb_txt = encode_text(query_text)
        query_emb = 0.5 * np.array(emb_img) + 0.5 * np.array(emb_txt)

        sims, order = compute_similarity(query_emb, articles)

        st.subheader("Top 10 articles combinés")
        for idx in order[:50]:
            row = articles.iloc[idx]
            col1, col2 = st.columns([1, 3])
            with col1:
                img_row = load_image_from_url(row["imageurl"])
                if img_row:
                    st.image(img_row, width=100)
            with col2:
                st.write(f"**{row['title']}**")
                st.write(f"*Code du produit : {row['product_code']}*")
                st.write(f"Catégorie: {row['category1_code']} / {row['category2_code']}")
                st.write(f"Prix : {row['price']} €")