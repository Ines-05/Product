import streamlit as st
import pandas as pd
import base64
from io import BytesIO
from dotenv import load_dotenv
import os
import agent
from PIL import Image
import io

# Charger les variables d'environnement
load_dotenv()

# Vérifier que la clé API est disponible
if not os.getenv("GEMINI_API_KEY"):
    st.error("La clé API Gemini n'est pas définie. Veuillez créer un fichier .env avec GEMINI_API_KEY=votre_clé_api")
    st.stop()
# Configuration de la page Streamlit
st.set_page_config( 
    page_title="Assistant Produits IA",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fonctions d'utilitaires
def get_image_html(img_base64):
    """Génère le code HTML pour afficher une image encodée en base64"""
    html = f"""
    <img src="data:image/png;base64,{img_base64}" 
         style="width:100%; max-width:200px; border-radius:10px; border:1px solid #ddd; padding:3px;">
    """
    return html

def display_chat_message(message, is_user=False):
    """Affiche un message de conversation avec le style approprié"""
    if is_user:
        st.markdown(
            f"""
            <div style="display: flex; justify-content: flex-end; margin-bottom: 10px;">
                <div style="background-color:#01050a; padding: 10px; border-radius: 10px; max-width: 80%;">
                    <p style="margin: 0;">{message}</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style="display: flex; justify-content: flex-start; margin-bottom: 10px;">
                <div style="background-color: #01050a; padding: 10px; border-radius: 10px; max-width: 80%;">
                    <p style="margin: 0;">{message}</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# Initialisation des variables de session
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    
if 'conversation_mode' not in st.session_state:
    st.session_state.conversation_mode = True

if 'model_initialized' not in st.session_state:
    st.session_state.model_initialized = False

# Initialiser les variables pour stocker les produits
if 'product_ids' not in st.session_state:
    st.session_state.product_ids = []

if 'product_images' not in st.session_state:
    st.session_state.product_images = {}

if 'product_details' not in st.session_state:
    st.session_state.product_details = {}

# Titre principal de l'application
st.title("🛍️ Assistant Produits Intelligent")

# Afficher des informations supplémentaires
with st.sidebar:
    st.header("Options")
    
    # Option pour activer/désactiver le mode conversation
    conversation_mode = st.toggle(
        "Mode conversation",
        value=st.session_state.conversation_mode,
        help="Active la mémoire de conversation pour des échanges plus naturels"
    )
    
    if conversation_mode != st.session_state.conversation_mode:
        st.session_state.conversation_mode = conversation_mode
        st.success(f"Mode conversation {'activé' if conversation_mode else 'désactivé'}")
    
    # Filtres de sentiment
    st.subheader("Filtres de sentiment")
    sentiment_options = ["Automatique", "Positif", "Négatif", "Neutre"]
    selected_sentiment = st.selectbox(
        "Privilégier les avis:",
        sentiment_options,
        index=0,
        help="Détermine quels types d'avis seront privilégiés dans les résultats"
    )
    
    # Convertir le sentiment sélectionné au format attendu par l'agent
    sentiment_mapping = {
        "Automatique": None,
        "Positif": "positive",
        "Négatif": "negative",
        "Neutre": "neutral"
    }
    sentiment_filter = sentiment_mapping[selected_sentiment]
    
    # Score minimum de polarité (si applicable)
    min_polarity = None
    if selected_sentiment != "Automatique":
        min_polarity = st.slider(
            "Score minimum de polarité:",
            min_value=-1.0,
            max_value=1.0,
            value=0.2 if selected_sentiment == "Positif" else (-0.2 if selected_sentiment == "Négatif" else 0.0),
            step=0.1
        )
    
    # Bouton pour effacer l'historique
    if st.button("Effacer l'historique"):
        st.session_state.chat_history = []
        agent.reset_memory()
        st.rerun()
    
    # Bouton pour réinitialiser le modèle (utile pendant le développement)
    if st.button("Réinitialiser le modèle"):
        st.session_state.model_initialized = False
        st.rerun()
    
    st.header("À propos")
    st.info(
        """
        Cet assistant utilise la technologie RAG (Retrieval-Augmented Generation)
        pour recommander des produits en fonction de votre demande.
        
        Il analyse votre requête, recherche les produits pertinents dans sa base
        de données et formule une réponse naturelle.
        
        Les images des produits sont générées par IA à partir des descriptions.
        
        En mode conversation, l'assistant peut poser des questions supplémentaires
        pour mieux comprendre vos préférences.
        """
    )

# Zone principale de l'application
main_col1, main_col2 = st.columns([2, 1])

# Afficher l'historique des messages
with main_col1:
    st.subheader("💬 Conversation")
    
    # Conteneur pour l'historique des messages
    chat_container = st.container()
    
    with chat_container:
        # Afficher les messages précédents
        for message in st.session_state.chat_history:
            display_chat_message(message["content"], message["is_user"])
    
    # Zone de saisie pour la question de l'utilisateur
    user_input = st.chat_input("Posez votre question sur les produits...")

# Zone des résultats de produits
with main_col2:
    st.subheader("🏷️ Produits suggérés")
    # Ajout d'un conteneur dédié aux produits pour une meilleure structure
    products_container = st.container()
    
    with products_container:
        if 'product_ids' in st.session_state and st.session_state.product_ids:
            for product_id in st.session_state.product_ids:
                product_image = st.session_state.product_images.get(product_id, "")
                product_details = st.session_state.product_details.get(product_id, {})
                
                with st.container():
                    st.markdown(f"""<div style="background-color: #f9f9f9;border-radius: 10px;
                            padding: 15px;margin-bottom: 15px;border-left: 5px solid #1E88E5;box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                                </div>"""
                            , unsafe_allow_html=True)
                    
                    # Affichage de l'image du produit
                    if product_image:
                        image_data = base64.b64decode(product_image)
                        image = Image.open(io.BytesIO(image_data))
                        st.image(image, caption=f"Produit {product_id}", width=150)
                    
                    # Nom du produit
                    st.markdown(f"### {product_details.get('name', f'Produit {product_id}')}")
                    
                    # Sentiment avec coloration
                    sentiment = product_details.get('sentiment', 'neutral')
                    sentiment_text = {
                        'positive': 'Positif',
                        'negative': 'Négatif',
                        'neutral': 'Neutre'
                    }.get(sentiment, 'Non spécifié')
                    
                    sentiment_colors = {
                        'positive': 'green',
                        'negative': 'red',
                        'neutral': 'gray'
                    }
                    color = sentiment_colors.get(sentiment, 'blue')
                    
                    st.markdown(f"""
                        <div style="display: inline-block; background-color: {color}; 
                                color: white; padding: 2px 8px; border-radius: 10px; 
                                font-size: 0.8em; margin-bottom: 10px;">
                            {sentiment_text}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Ajout du dropdown pour afficher les détails
                    with st.expander("Voir détails"):
                        st.markdown(product_details.get('features', 'Aucun détail disponible'), unsafe_allow_html=True)
                        
                        # Afficher le score de polarité si disponible
                        if 'polarity_score' in product_details:
                            st.progress(
                                (product_details['polarity_score'] + 1) / 2,  # Normaliser entre 0 et 1
                                text=f"Score de polarité: {product_details['polarity_score']:.2f}"
                            )
        else:
            st.info("Posez une question pour voir les produits recommandés.")

# Initialiser le modèle si ce n'est pas déjà fait
if not st.session_state.model_initialized:
    with st.spinner("Initialisation du moteur d'IA..."):
        try:
            agent.initialize()
            st.session_state.model_initialized = True
            st.success("Modèle initialisé avec succès!")
        except Exception as e:
            st.error(f"Erreur lors de l'initialisation du modèle: {str(e)}")

# Traitement de la requête utilisateur
if user_input and st.session_state.model_initialized:
    # Ajouter la question de l'utilisateur à l'historique
    st.session_state.chat_history.append({"content": user_input, "is_user": True})
    
    # Afficher un indicateur de chargement
    with main_col1:
        with st.spinner("Recherche des produits..."):
            # Traiter la question avec l'agent
            answer, product_ids, product_images, product_details = agent.process_query(
                user_input,
                use_conversation=st.session_state.conversation_mode,
                sentiment_filter=sentiment_filter,
                min_polarity=min_polarity
            )
    
    # Mettre à jour les variables de session avec les résultats
    st.session_state.product_ids = product_ids
    st.session_state.product_images = product_images
    st.session_state.product_details = product_details
    
    # Ajouter la réponse à l'historique
    st.session_state.chat_history.append({"content": answer, "is_user": False})
    
    # Forcer le rafraîchissement de l'interface pour afficher les nouveaux messages
    st.rerun()