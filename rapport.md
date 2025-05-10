# Résumé du Processus RAG dans le système d'analyse de produits

## Architecture RAG (Retrieval-Augmented Generation)

Le système d'analyse de produits implémente un modèle RAG sophistiqué qui permet de générer des réponses précises et contextuelles aux requêtes des utilisateurs en s'appuyant sur une base de données d'avis produits. Voici le processus détaillé :

## 1. Préparation des données

- **Chargement des données** : Le système charge un dataset d'avis clients depuis un fichier CSV (`df_sample.csv`).
- **Analyse de sentiment** : Chaque avis est analysé avec TextBlob pour déterminer :
  - Un score de polarité (entre -1 et 1)
  - Une classification en catégories (positif, neutre, négatif)
- **Enrichissement des métadonnées** : Les avis sont enrichis avec des métadonnées incluant l'ID du produit et les informations de sentiment.

## 2. Création de la base de données vectorielle

- **Génération d'embeddings** : Les textes des avis sont convertis en embeddings vectoriels à l'aide du modèle Gemini (`GoogleGenerativeAIEmbeddings`).
- **Indexation FAISS** : Un index FAISS est créé pour permettre une recherche vectorielle rapide et efficace.
- **Stockage des documents** : Les documents originaux et leurs métadonnées sont stockés dans un `InMemoryDocstore` avec un mapping entre l'index FAISS et les IDs de documents.

## 3. Processus de requête et de récupération

- **Analyse d'intention** : Le système analyse la requête de l'utilisateur pour détecter si elle privilégie les avis positifs ou négatifs.
- **Configuration du retriever** : Un retriever personnalisé est créé avec les filtres appropriés :
  - Filtres de sentiment (`positive`, `negative`, `neutral`)
  - Filtres de polarité minimale pour affiner les résultats
- **Recherche sémantique** : Le système utilise l'algorithme MMR (Maximal Marginal Relevance) pour équilibrer pertinence et diversité des résultats.
- **Paramètres de recherche** :
  - `k=5` : Nombre de documents à récupérer
  - `fetch_k=10` : Nombre de documents candidats à considérer
  - `lambda_mult=0.5` : Facteur d'équilibre entre pertinence et diversité

## 4. Génération de réponses

- **Chaîne de traitement QA** : Utilisation de `RetrievalQA` de LangChain pour intégrer les documents récupérés dans la génération de réponse.
- **Méthode de chaînage** : Utilisation de la stratégie `"stuff"` qui injecte tous les documents pertinents dans le prompt du LLM.
- **Mémoire de conversation** : Utilisation de `ConversationBufferMemory` pour maintenir le contexte à travers les échanges.
- **Génération de réponse** : Le modèle Gemini (`ChatGoogleGenerativeAI`) génère une réponse cohérente basée sur les documents récupérés et le contexte de la conversation.

## 5. Extraction des informations produits

- **Extraction des IDs de produits** des documents récupérés
- **Compilation des détails produits** à partir des avis et des métadonnées
- **Génération d'images de démonstration** pour représenter visuellement les produits
- **Construction d'un profil produit** comprenant :
  - Nom du produit
  - Sentiment général
  - Score de polarité
  - Résumé des avis

## Optimisations du système RAG

- **Filtrage dynamique** : Adaptation des filtres en fonction du contexte de la requête
- **Sous-classes de retriever personnalisées** pour les filtres complexes de polarité
- **Recherche hybride MMR** pour maximiser la pertinence tout en assurant la diversité
- **Mémoire contextuelle** permettant des conversations multi-tours
- **Réinitialisation de la mémoire** pour démarrer de nouvelles sessions

## Avantages du modèle RAG implémenté

1. **Précision améliorée** : Les réponses sont ancrées dans des données réelles d'avis clients
2. **Pertinence contextuelle** : Le système priorise les avis positifs ou négatifs selon l'intention détectée
3. **Transparence** : Les sources (avis) sont récupérables et peuvent être présentées à l'utilisateur
4. **Flexibilité** : Le système peut être facilement adapté à différents types de produits ou de requêtes
5. **Évolutivité** : L'architecture permet d'ajouter facilement de nouvelles données ou de nouveaux filtres

Cette implémentation RAG offre un équilibre optimal entre la puissance du LLM (Gemini) et l'ancrage factuel dans une base de données d'avis réels, permettant des réponses précises et contextuelles aux questions des utilisateurs.