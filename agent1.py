import os
import base64
import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Récupérer la clé API Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("La clé API Gemini n'a pas été trouvée. Assurez-vous de l'avoir définie dans le fichier .env")

# Variables globales pour stocker le modèle et ses composants
conversation_chain = None
qa_chain = None
df_sample = None
vector_store = None
llm = None
llm_image_gen = None
memory = None

def load_data():
    """Charger le dataset et le retourner"""
    global df_sample
    
    if df_sample is None:
        df_sample = pd.read_csv('df_sample.csv')
    
    return df_sample

def image_to_base64(image_path):
    """
    Convertit une image depuis un chemin de fichier en chaîne base64
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_string
    except Exception as e:
        print(f"Erreur lors de la conversion de l'image en base64: {e}")
        return None
    

def get_llm():
    """Initialiser et retourner le modèle LLM"""
    global llm
    
    if llm is None:
        # Configuration du LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GEMINI_API_KEY,
            temperature=0.2,  # Un peu de créativité pour les questions de suivi
            max_tokens=None,
            timeout=None,
            max_retries=3
        )
    
    return llm

def get_image_gen_llm():
    """Initialiser et retourner le modèle LLM pour la génération d'images"""
    global llm_image_gen
    
    if llm_image_gen is None:
        try:
            # Configuration du LLM pour la génération d'images
            llm_image_gen = ChatGoogleGenerativeAI(
                model="models/gemini-2.0-flash-exp-image-generation",
                google_api_key=GEMINI_API_KEY,
                temperature=0.7,  # Plus de créativité pour les images
                max_tokens=None,
                timeout=None,
                max_retries=3
            )
        except Exception as e:
            print(f"Erreur lors de l'initialisation du modèle de génération d'images: {e}")
            llm_image_gen = None
    
    return llm_image_gen

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

def get_memory():
    """Initialiser et retourner la mémoire de conversation"""
    global memory
    
    if memory is None:
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
    
    return memory

def get_conversation_chain():
    """Récupérer la chaîne de conversation ou la créer si elle n'existe pas encore"""
    global conversation_chain, vector_store, memory
    
    # Si la chaîne de conversation existe déjà, la retourner directement
    if conversation_chain is not None:
        return conversation_chain
    
    # Sinon, créer la chaîne de conversation
    
    # S'assurer que le vector_store est créé
    if vector_store is None:
        vector_store = create_embeddings_and_index()
    
    # S'assurer que la mémoire est initialisée
    if memory is None:
        memory = get_memory()
    
    # Configuration du retriever
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.5}
    )
    
    # Obtenir le LLM
    llm = get_llm()
    
    # Définition du prompt personnalisé pour guider les réponses du modèle
    condense_question_prompt_template = """
    Étant donné l'historique de conversation suivant et la question actuelle de l'utilisateur, 
    reformule la question pour qu'elle soit autonome.
    
    Si la question est vague ou concerne un nouveau sujet, conserve la formulation originale.
    Si l'utilisateur indique qu'il n'est pas expert ou qu'il a peu d'expérience, intègre cette 
    information dans la question reformulée.
    
    Historique de conversation:
    {chat_history}
    
    Question actuelle: {question}
    
    Question reformulée:
    """
    
    condense_question_prompt = PromptTemplate(
        template=condense_question_prompt_template,
        input_variables=["chat_history", "question"]
    )
    
    qa_prompt_template = """
    Tu es un assistant de recommandation de produits expert, amical et conversationnel.
    
    Basé sur les documents de contexte suivants et sur l'historique de la conversation, 
    réponds à la question de l'utilisateur de manière naturelle et conversationnelle.
    
    INSTRUCTIONS IMPORTANTES:
    1. Si l'utilisateur indique directement ou indirectement qu'il n'est pas expert, qu'il a peu d'expérience, ou qu'il demande des conseils généraux, PROPOSE-LUI IMMÉDIATEMENT DES PRODUITS SPÉCIFIQUES sans poser de questions supplémentaires. Pars du principe qu'il a besoin de ton expertise pour faire un choix.

    2. Si l'utilisateur ne donne pas de critères spécifiques mais semble avoir besoin de conseils généraux (par exemple "je n'ai pas vraiment d'expérience", "je ne sais pas quoi choisir", etc.), recommande-lui directement 2 à 3 produits différents en expliquant pourquoi ils pourraient lui convenir.
    
    3. Seulement si l'utilisateur pose une question très précise ou spécifique, tu peux alors poser une question de suivi pour clarifier ses besoins, mais limite-toi à UNE SEULE question à la fois.
    
    4. Présente toujours tes recommandations avec:
       - Le nom du produit
       - Une brève description
       - Ses avantages principaux
    
    5. Donne toujours des recommandations concrètes, pas des conseils généraux.
    
    Si tu ne trouves pas de produits pertinents à recommander en fonction de la requête, 
    réponds simplement "Je n'ai pas de suggestions de produits pour cette demande" sans t'excuser 
    ni donner d'explications techniques sur pourquoi tu ne peux pas répondre.
    
    Contexte: {context}
    
    Question: {question}
    
    Réponse:
    """
    
    qa_prompt = PromptTemplate(
        template=qa_prompt_template,
        input_variables=["context", "question"]
    )
    
    # Création de la chaîne de conversation
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        condense_question_prompt=condense_question_prompt,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        return_source_documents=True,
        verbose=True
    )
    
    return conversation_chain

def get_qa_chain():
    """Récupérer la chaîne QA standard pour les requêtes simples"""
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
    llm = get_llm()
    
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

import os
import base64
import mimetypes
from google import genai
from google.genai import types


def save_binary_file(file_path, data):
    with open(file_path, "wb") as f:
        f.write(data)
    print(f"Image enregistrée à : {file_path}")
    return file_path


def generate_product_image(product_description, file_name="image_produit"):
    """Générer une image photo-réaliste d’un produit à partir de sa description"""

    try:
        client = genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY"),
        )

        model = "gemini-2.0-flash-preview-image-generation"
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=f"""generate an image of a product based on the following description: {product_description}"""),
                    
                ],
            )
        ]
        generate_content_config = types.GenerateContentConfig(
            response_modalities=["image", "text"],
            response_mime_type="text/plain",
        )

        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            if (
                chunk.candidates is None
                or chunk.candidates[0].content is None
                or chunk.candidates[0].content.parts is None
            ):
                continue

            part = chunk.candidates[0].content.parts[0]
            if part.inline_data:
                inline_data = part.inline_data
                data_buffer = inline_data.data
                file_extension = mimetypes.guess_extension(inline_data.mime_type)
                file_path = f"{file_name}{file_extension}"
                return save_binary_file(file_path, data_buffer)

            elif hasattr(chunk, "text"):
                print("Texte généré :", chunk.text)

        print("Aucune image générée.")
        return None

    except Exception as e:
        print(f"Erreur lors de la génération d'image: {e}")
        return None


def initialize():
    """Initialiser les composants du système de recommandation"""
    load_data()
    create_embeddings_and_index()
    get_conversation_chain()
    get_qa_chain()
    #get_image_gen_llm()  # Tente d'initialiser le modèle de génération d'images
    return True

def reset_memory():
    """Réinitialiser la mémoire de conversation"""
    global memory
    memory = None
    get_memory()
    return True

def get_description_llm():
    """
    Renvoie l'instance LLM à utiliser pour la reformulation des descriptions
    Utilise le même modèle Gemini que pour les requêtes principales
    """
    # Réutiliser le même modèle LLM que pour les requêtes
    return get_llm()

def enhance_product_description(description):
    """
    Reformuler ou résumer la description du produit en anglais correct
    pour une meilleure génération d'image
    """
    llm = get_description_llm()
    
    prompt = f"""
    Summarize and enhance the following product description for image generation.
    Create a clear, concise description in proper English that highlights key visual aspects.
    Keep it under 75 words and focus on appearance, color, shape, and distinctive features.
    
    Original description: {description}
    
    Enhanced description:
    """
    
    # Obtenir la réponse du LLM via l'invocation directe de Gemini
    try:
        response = llm.invoke(prompt).content
        # Nettoyer la réponse si nécessaire
        enhanced_description = response.strip()
        return enhanced_description
    except Exception as e:
        print(f"Erreur lors de la reformulation de la description: {e}")
        # En cas d'erreur, retourner la description originale
        return description

def image_to_base64(image_path):
    """
    Convertit une image depuis un chemin de fichier en chaîne base64
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_string
    except Exception as e:
        print(f"Erreur lors de la conversion de l'image en base64: {e}")
        return None

def verify_image_format(image_path):
    """
    Vérifie que l'image existe et est dans un format valide
    Retourne le chemin si valide, None sinon
    """
    try:
        from PIL import Image
        # Vérifier que le fichier existe
        if not os.path.exists(image_path):
            print(f"Le fichier image n'existe pas: {image_path}")
            return None
        
        # Vérifier que c'est une image valide
        try:
            with Image.open(image_path) as img:
                format = img.format
                print(f"Format d'image détecté: {format}")
                # Enregistrer l'image au format PNG si ce n'est pas déjà le cas
                if format != 'PNG' and format != 'JPEG':
                    new_path = f"{os.path.splitext(image_path)[0]}.png"
                    img.save(new_path, format='PNG')
                    print(f"Image convertie et enregistrée en PNG: {new_path}")
                    return new_path
                return image_path
        except Exception as e:
            print(f"Fichier non valide comme image: {image_path}, erreur: {e}")
            return None
    except ImportError:
        print("La bibliothèque PIL (Pillow) n'est pas installée.")
        # Si PIL n'est pas disponible, on suppose que l'image est valide
        return image_path if os.path.exists(image_path) else None

def process_query(query, use_conversation=True):
    """Traiter une requête utilisateur et retourner la réponse, les produits recommandés et leurs images"""
    if use_conversation:
        chain = get_conversation_chain()
    else:
        chain = get_qa_chain()
    
    response = chain({"question": query})
    
    answer = response["answer"]
    source_docs = response.get("source_documents", [])
    
    # Extraire les IDs des produits recommandés et les dédupliquer
    product_ids = []
    seen_product_ids = set()
    
    for doc in source_docs:
        if "product_id" in doc.metadata:
            product_id = doc.metadata["product_id"]
            # Éviter les doublons
            if product_id not in seen_product_ids:
                product_ids.append(product_id)
                seen_product_ids.add(product_id)
    
    # Générer des images pour les produits s'ils existent
    product_images = {}  # Dictionnaire qui contiendra les images en base64
    
    if product_ids:
        df = load_data()
        for pid in product_ids[:3]:  # Limiter à 3 produits pour des raisons de performance
            product_info = df[df["ProductId"] == pid]
            if not product_info.empty:
                original_description = product_info["clean_text"].values[0]
                print(f"Description originale: {original_description}")
                
                # Reformuler la description pour la génération d'image
                enhanced_description = enhance_product_description(original_description)
                print(f"Description améliorée: {enhanced_description}")
                
                # Utiliser la description améliorée pour la génération d'image
                image_filename = f"product_{pid}"
                image_path = generate_product_image(enhanced_description, file_name=image_filename)
                
                if image_path:
                    # Vérifier et standardiser le format de l'image
                    verified_image_path = verify_image_format(image_path)
                    if verified_image_path:
                        # Convertir l'image en base64 pour l'affichage dans Streamlit
                        base64_image = image_to_base64(verified_image_path)
                        if base64_image:
                            product_images[pid] = base64_image
                            print(f"Image générée et convertie en base64 pour le produit {pid}")
                        else:
                            print(f"Échec de la conversion en base64 pour le produit {pid}")
                    else:
                        print(f"Format d'image non valide pour le produit {pid}")
                else:
                    print(f"Pas d'image générée pour le produit {pid}")
    
    return answer, product_ids, product_images