import faiss
import numpy as np
import pandas as pd
import os
import base64
from textblob import TextBlob
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
import io
from PIL import Image, ImageDraw

# Variables globales pour stocker les objets réutilisables
df_sample = None
vector_store = None
qa_chain = None
memory = None
llm = None
embedding_model = None


def load_data():
    """Charger le dataset et le retourner"""
    global df_sample
    
    if df_sample is None:
        df_sample = pd.read_csv('df_sample.csv')
    
    return df_sample
def initialize():
    """Initialise le modèle, les embeddings et la base de données vectorielle"""
    global df_sample, vector_store, qa_chain, memory, llm, embedding_model
    
    load_dotenv()
    
    # Charger le dataset (à adapter selon votre structure de données)
    df_sample = load_data()
    
    # Initialiser la mémoire de conversation
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="result"
    )
    
    # Initialiser le modèle LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=3
    )
    
    # Initialiser le modèle d'embeddings Gemini
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Préparer les textes et générer les embeddings
    texts = df_sample["clean_text"].tolist()
    
    # Analyse de sentiment pour chaque texte
    sentiments = []
    for text in texts:
        # TextBlob retourne une polarité entre -1 (négatif) et 1 (positif)
        polarity = TextBlob(text).sentiment.polarity
        
        # Classification en catégories
        if polarity > 0.1:
            sentiment = "positive"
        elif polarity < -0.1:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        # Stockage de la polarité numérique et de la catégorie
        sentiments.append({
            "polarity_score": polarity,
            "sentiment": sentiment
        })
    
    # Mise à jour des métadonnées avec le sentiment
    metadatas = []
    for i, row in df_sample.iterrows():
        metadata = {
            "product_id": row["ProductId"],
            "sentiment": sentiments[i]["sentiment"],
            "polarity_score": sentiments[i]["polarity_score"]
        }
        metadatas.append(metadata)
    
    # Générer les embeddings avec Gemini
    embeddings_array = np.array(embedding_model.embed_documents(texts))
    
    # Créer les documents LangChain
    documents = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(texts, metadatas)
    ]
    
    # Créer l'index FAISS
    embedding_dim = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings_array)
    
    # Créer le docstore et le mapping
    docstore = InMemoryDocstore()
    index_to_docstore_id = {}
    
    for i, doc in enumerate(documents):
        doc_id = str(i)
        docstore.add({doc_id: doc})
        index_to_docstore_id[i] = doc_id
    
    # Créer le VectorStore FAISS
    vector_store = FAISS(
        embedding_function=embedding_model,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )
    
    # Créer le retriever par défaut (sans filtre)
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.5}
    )
    
    # Créer la chaîne QA
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )
    
    return True

def reset_memory():
    """Réinitialise la mémoire de conversation"""
    global memory
    if memory:
        memory.clear()

def get_retriever(sentiment_filter=None, min_polarity=None):
    """
    Crée un retriever qui peut filtrer par sentiment ou score de polarité
    
    Args:
        sentiment_filter (str, optional): 'positive', 'negative', ou 'neutral'
        min_polarity (float, optional): Score minimum de polarité (entre -1 et 1)
    
    Returns:
        Un retriever configuré avec les filtres demandés
    """
    global vector_store
    
    if vector_store is None:
        raise ValueError("Le VectorStore n'est pas initialisé.")
    
    search_kwargs = {"k": 5, "fetch_k": 10, "lambda_mult": 0.5}
    
    # Ajout du filtre de métadonnées si spécifié
    if sentiment_filter or min_polarity is not None:
        filter_dict = {}
        if sentiment_filter:
            filter_dict["sentiment"] = sentiment_filter
        
        search_kwargs["filter"] = filter_dict
    
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs
    )
    
    # Si on a besoin de filtrer par score de polarité, on crée une sous-classe
    if min_polarity is not None:
        from langchain.schema import BaseRetriever
        from typing import List
        from langchain.schema import Document
        
        class FilteredRetriever(BaseRetriever):
            def get_relevant_documents(self, query: str) -> List[Document]:
                docs = retriever.get_relevant_documents(query)
                return [doc for doc in docs if doc.metadata.get("polarity_score", -1) >= min_polarity]
            
            async def aget_relevant_documents(self, query: str) -> List[Document]:
                docs = await retriever.aget_relevant_documents(query)
                return [doc for doc in docs if doc.metadata.get("polarity_score", -1) >= min_polarity]
        
        return FilteredRetriever()
    
    return retriever

def get_product_images(product_ids):
    """
    Récupère les images des produits (pour cette démo, nous simulons des images encodées en base64)
    Dans une application réelle, vous récupéreriez les images depuis une base de données ou une API.
    """
    product_images = {}
    
    # Simuler des images pour la démo (rectangles colorés avec ID du produit)
    for pid in product_ids:
        # Cette fonction simule une image encodée en base64
        # Dans une vraie application, vous récupéreriez l'image réelle du produit
        product_images[pid] = generate_mock_image(pid)
    
    return product_images

def generate_mock_image(product_id):
    """Génère une image factice encodée en base64 pour la démonstration"""
    import random
    
    # Créer une image avec un fond de couleur aléatoire
    width, height = 200, 150
    color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
    
    image = Image.new("RGB", (width, height), color)
    draw = ImageDraw.Draw(image)
    
    # Dessiner un cadre
    border_color = (50, 50, 50)
    border_width = 5
    draw.rectangle([(0, 0), (width-1, height-1)], outline=border_color, width=border_width)
    
    # Ajouter l'ID du produit comme texte
    draw.text((width//2 - 20, height//2 - 10), f"ID: {product_id}", fill=(0, 0, 0))
    
    # Convertir l'image en base64
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return img_str

def extract_product_details(source_documents):
    """
    Extrait les détails des produits à partir des documents sources
    """
    product_details = {}
    product_ids = []
    
    for doc in source_documents:
        product_id = doc.metadata.get("product_id")
        sentiment = doc.metadata.get("sentiment", "non spécifié")
        polarity_score = doc.metadata.get("polarity_score", 0)
        
        if product_id and product_id not in product_ids:
            product_ids.append(product_id)
            
            # Extraire quelques détails du texte pour la démonstration
            text = doc.page_content
            
            # Simuler l'extraction d'un nom de produit (à adapter selon vos données)
            # Dans une vraie application, vous récupéreriez le nom réel du produit
            name = f"Produit {product_id}"
            
            # Extraire quelques caractéristiques du texte
            features = f"""
            **Sentiment**: {sentiment.capitalize()} (score: {polarity_score:.2f})
            
            **Résumé**: {text[:150]}...
            """
            
            product_details[str(product_id)] = {
                "name": name,
                "features": features,
                "sentiment": sentiment,
                "polarity_score": polarity_score
            }
    
    return product_ids, product_details

def process_query(query, use_conversation=True, sentiment_filter=None, min_polarity=None):
    """
    Traite une requête utilisateur et retourne une réponse avec les produits recommandés
    
    Args:
        query (str): La question ou la requête de l'utilisateur
        use_conversation (bool): Utiliser l'historique de conversation
        sentiment_filter (str, optional): Filtre de sentiment ('positive', 'negative', 'neutral')
        min_polarity (float, optional): Score minimum de polarité
    
    Returns:
        tuple: (réponse, list_product_ids, dict_product_images, dict_product_details)
    """
    global qa_chain, memory
    
    if qa_chain is None:
        raise ValueError("Le système n'est pas initialisé. Appelez initialize() d'abord.")
    
    # Déterminer s'il s'agit d'une requête qui devrait privilégier les avis positifs ou négatifs
    # (Analyse simple basée sur les mots clés)
    query_lower = query.lower()
    
    # Si aucun filtre n'est spécifié, essayer de détecter le sentiment de la requête
    if sentiment_filter is None:
        if any(word in query_lower for word in ["meilleur", "excellent", "recommander", "avantage", "top", "positif"]):
            sentiment_filter = "positive"
            min_polarity = 0.2  # Privilégier les avis plutôt positifs
        elif any(word in query_lower for word in ["problème", "défaut", "mauvais", "éviter", "négatif", "plainte"]):
            sentiment_filter = "negative"
            min_polarity = -0.2  # Privilégier les avis plutôt négatifs
    
    # Créer un retriever avec le filtre de sentiment approprié
    custom_retriever = get_retriever(sentiment_filter=sentiment_filter, min_polarity=min_polarity)
    
    # Mise à jour dynamique du retriever dans la chaîne QA
    qa_chain.retriever = custom_retriever
    
    try:
        # Exécuter la requête
        if use_conversation:
            result = qa_chain({"query": query})
        else:
            # Créer une chaîne QA temporaire sans mémoire pour une requête ponctuelle
            temp_qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=custom_retriever,
                return_source_documents=True
            )
            result = temp_qa_chain({"query": query})
        
        # Extraire la réponse et les documents sources
        answer = result["result"]
        source_documents = result["source_documents"]
        
        # Extraire les IDs des produits et leurs détails
        product_ids, product_details = extract_product_details(source_documents)
        
        # Récupérer les images des produits
        product_images = get_product_images(product_ids)
        
        # Ajouter des informations sur le filtrage par sentiment dans la réponse
        if sentiment_filter:
            sentiment_type = {
                "positive": "positifs",
                "negative": "négatifs",
                "neutral": "neutres"
            }.get(sentiment_filter, "")
            
            #answer += f"\n\n_J'ai privilégié les avis {sentiment_type} pour cette recherche._"
        
        return answer, product_ids, product_images, product_details
        
    except Exception as e:
        print(f"Erreur lors du traitement de la requête: {str(e)}")
        return f"Je suis désolé, une erreur est survenue lors du traitement de votre requête: {str(e)}", [], {}, {}