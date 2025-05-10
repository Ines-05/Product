import streamlit as st
from dotenv import load_dotenv
import os
import agent

# Charger les variables d'environnement
load_dotenv()

# Vérifier que la clé API est disponible
if not os.getenv("GEMINI_API_KEY"):
    st.error("La clé API Gemini n'est pas définie. Veuillez créer un fichier .env avec GEMINI_API_KEY=votre_clé_api")
    st.stop()

# Configuration de la page Streamlit
st.set_page_config(page_title="Assistant de Recommandation Produits", layout="wide")
st.title("Assistant de Recommandation Produits")

# Initialisation de la session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "model_initialized" not in st.session_state:
    st.session_state.model_initialized = False

# Initialiser le modèle au premier chargement
if not st.session_state.model_initialized:
    with st.spinner("Initialisation du système de recommandation..."):
        try:
            success = agent.initialize()
            if success:
                st.session_state.model_initialized = True
                st.success("Système prêt!")
            else:
                st.error("Erreur lors de l'initialisation du système")
                st.stop()
        except Exception as e:
            st.error(f"Erreur: {str(e)}")
            st.stop()

# Charger les données
df_sample = agent.load_data()

# Interface utilisateur pour le chat
st.subheader("Posez votre question sur les produits")

# Affichage de l'historique des conversations
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
        # Si la réponse contient des produits recommandés, afficher les détails
        if message["role"] == "assistant" and "product_ids" in message:
            product_ids = message["product_ids"]
            if product_ids:
                with st.expander("Voir les détails des produits"):
                    for pid in product_ids:
                        product_info = df_sample[df_sample["ProductId"] == pid]
                        if not product_info.empty:
                            st.write(f"**Produit ID:** {pid}")
                            if "Text" in product_info.columns:
                                st.write(f"**Description:** {product_info['Text'].values[0]}")
                            st.write("---")

# Zone de saisie utilisateur
user_input = st.chat_input("Quelle recommandation de produit recherchez-vous ?")

if user_input:
    # Afficher la question de l'utilisateur
    with st.chat_message("user"):
        st.write(user_input)
    
    # Ajouter à l'historique
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    # Obtenir la réponse du système
    with st.spinner("Recherche en cours..."):
        try:
            answer, product_ids = agent.process_query(user_input)
        except Exception as e:
            answer = f"Erreur lors du traitement de votre requête: {str(e)}"
            product_ids = []
    
    # Afficher la réponse
    with st.chat_message("assistant"):
        st.write(answer)
        
        # Si des produits ont été trouvés, afficher les détails
        if product_ids:
            with st.expander("Voir les détails des produits recommandés"):
                for pid in product_ids:
                    product_info = df_sample[df_sample["ProductId"] == pid]
                    if not product_info.empty:
                        st.write(f"**Produit ID:** {pid}")
                        if "Text" in product_info.columns:
                            st.write(f"**Description:** {product_info['Text'].values[0]}")
                        st.write("---")
    
    # Ajouter la réponse à l'historique avec les IDs des produits
    st.session_state.chat_history.append({
        "role": "assistant", 
        "content": answer,
        "product_ids": product_ids
    })

# Afficher des informations supplémentaires
with st.sidebar:
    st.header("À propos")
    st.info(
        """
        Cet assistant utilise la technologie RAG (Retrieval-Augmented Generation) 
        pour recommander des produits en fonction de votre demande.
        
        Il analyse votre requête, recherche les produits pertinents dans sa base 
        de données et formule une réponse naturelle.
        """
    )
    
    # Bouton pour effacer l'historique
    if st.button("Effacer l'historique"):
        st.session_state.chat_history = []
        st.rerun()
    
    # Bouton pour réinitialiser le modèle (utile pendant le développement)
    if st.button("Réinitialiser le modèle"):
        st.session_state.model_initialized = False
        st.rerun()