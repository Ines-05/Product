import os
import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Récupérer la clé API Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("La clé API Gemini n'a pas été trouvée. Assurez-vous de l'avoir définie dans le fichier .env")

# Variables globales pour stocker le modèle et ses composants
qa_chain = None
df_sample = None
vector_store = None

def load_data():
    """Charger le dataset et le retourner"""
    global df_sample
    
    if df_sample is None:
        df_sample = pd.read_csv('df_sample.csv')
    
    return df_sample

def create_embeddings_and_index():
    """Créer les embeddings et l'index FAISS une seule fois"""
    global vector_store, df_sample
    
    # S'assurer que les données sont chargées
    if df_sample is None:
        df_sample = load_data()
    
    # Si le vector_store existe déjà, le retourner directement
    if vector_store is not None:
        return vector_store
    
    # Sinon, créer les embeddings et l'index
    
    # 1. Chargement du modèle d'embeddings
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY
    )
    
    # 2. Préparation des textes et des métadonnées
    texts = df_sample["clean_text"].tolist()
    metadatas = [{"product_id": pid} for pid in df_sample["ProductId"]]
    
    # 3. Génération des embeddings
    embeddings_array = np.array(embedding_model.embed_documents(texts))
    
    # 4. Création des documents LangChain
    documents = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(texts, metadatas)
    ]
    
    # 5. Création de l'index FAISS
    embedding_dim = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings_array)
    
    # 6. Création du docstore et du mapping
    docstore = InMemoryDocstore()
    index_to_docstore_id = {}
    for i, doc in enumerate(documents):
        doc_id = str(i)
        docstore.add({doc_id: doc})
        index_to_docstore_id[i] = doc_id
    
    # 7. Création du VectorStore FAISS avec LangChain
    vector_store = FAISS(
        embedding_function=embedding_model,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )
    
    return vector_store

def get_qa_chain():
    """Récupérer la chaîne QA ou la créer si elle n'existe pas encore"""
    global qa_chain, vector_store
    
    # Si la chaîne QA existe déjà, la retourner directement
    if qa_chain is not None:
        return qa_chain
    
    # Sinon, créer la chaîne QA
    
    # S'assurer que le vector_store est créé
    if vector_store is None:
        vector_store = create_embeddings_and_index()
    
    # Configuration du retriever
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.5}
    )
    
    # Configuration du LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=3
    )
    
    # Définition du prompt personnalisé pour guider les réponses du modèle
    custom_prompt_template = """
    Tu es un assistant de recommandation de produits amical et serviable.
    
    Basé sur les documents de contexte suivants, réponds à la question de l'utilisateur de manière naturelle et conversationnelle.
    
    Si tu ne trouves pas de produits pertinents à recommander en fonction de la requête, réponds simplement "Je n'ai pas de suggestions de produits pour cette demande" sans t'excuser ni donner d'explications techniques sur pourquoi tu ne peux pas répondre.
    
    Contexte: {context}
    
    Question: {question}
    
    Réponse:
    """
    
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )
    
    # Création de la chaîne QA avec le prompt personnalisé
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain

def initialize():
    """Initialiser les composants du système de recommandation"""
    load_data()
    create_embeddings_and_index()
    get_qa_chain()
    return True

def process_query(query):
    """Traiter une requête utilisateur et retourner la réponse et les produits recommandés"""
    qa_chain = get_qa_chain()
    response = qa_chain({"query": query})
    
    answer = response["result"]
    source_docs = response.get("source_documents", [])
    
    # Extraire les IDs des produits recommandés
    product_ids = []
    for doc in source_docs:
        if "product_id" in doc.metadata:
            product_ids.append(doc.metadata["product_id"])
    
    return answer, product_ids