# Assistant d'Analyse de Produits

## Présentation

L'Assistant d'Analyse de Produits est une application web interactive qui permet aux utilisateurs d'interroger en langage naturel une base de données d'avis sur des produits. L'application utilise des technologies d'intelligence artificielle pour analyser et récupérer des informations pertinentes à partir d'un large ensemble d'avis clients.


## Fonctionnalités principales

- **Recherche en langage naturel** : Posez des questions sur des produits comme vous le feriez à un conseiller humain
- **Analyse de sentiment** : Les avis sont automatiquement analysés (positif, neutre, négatif) 
- **Filtrage intelligent** : Le système détecte les intentions de recherche et privilégie les avis positifs ou négatifs selon le contexte
- **Interface conversationnelle** : Gardez une trace de votre historique de conversation
- **Visualisation des produits** : Consultez les produits recommandés avec leurs détails via un menu déroulant
- **Mémoire de conversation** : Le système se souvient du contexte des échanges précédents

## Architecture technique

L'application est construite autour de plusieurs composants clés :

- **Frontend** : Interface utilisateur développée avec Streamlit
- **Moteur de recherche vectorielle** : FAISS pour des recherches sémantiques rapides et efficaces
- **Modèles d'intelligence artificielle** :
  - Gemini (Google) pour la génération de réponses et l'embeddings de texte
  - TextBlob pour l'analyse de sentiment
- **Système de filtrage** : Filtre dynamique des documents selon l'intention détectée

## Prérequis

- Python 3.8+
- Connexion Internet (pour l'API Google Gemini)
- Clé API Google Gemini

## Installation

1. Clonez ce dépôt :
```bash
git clone https://github.com/Ines-05/Product.git
```

2. Créez et activez un environnement virtuel (recommandé) :
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows : venv\Scripts\activate
```

3. Installez les dépendances :
```bash
pip install -r requirements.txt
```

4. Créez un fichier `.env` à la racine du projet avec votre clé API :
```
GOOGLE_API_KEY=votre_clé_api_google
```

## Structure des fichiers

```
assistant-analyse-produits/
├── agent.py               # Module principal avec les fonctions de traitement
├── stream.py                # Interface Streamlit
├── df_sample.csv         # Données d'exemple (avis produits)
├── .env                  # Variables d'environnement (non versionné)
├── requirements.txt      # Dépendances du projet
└── README.md             # Ce fichier
```

## Utilisation

1. Lancez l'application :
```bash
streamlit run stream.py
```

2. Ouvrez votre navigateur à l'adresse indiquée (généralement http://localhost:8501)

3. Commencez à poser des questions comme :
   - "Je souhaite avoir des recommandations sur des produits cosmétiques pour rendre la peau lisse "
   - "Quelles sont les meilleurs produits cosmétiques??"

## Personnalisation des données

Par défaut, l'application utilise un échantillon de données préchargé (`df_sample.csv`). Pour utiliser vos propres données :

1. Préparez un fichier CSV avec au moins les colonnes suivantes :
   - `ProductId` : Identifiant unique du produit
   - `clean_text` : Texte de l'avis nettoyé

2. Remplacez le chemin dans la fonction `load_data()` du fichier `agent.py`

## Fonctionnalités avancées

### Filtrage par sentiment

Le système peut détecter automatiquement si votre requête privilégie les avis positifs ou négatifs en analysant les mots-clés de votre question.

### Détails des produits

Pour chaque produit suggéré, vous pouvez cliquer sur "Voir détails" pour afficher des informations supplémentaires comme :
- Le score de sentiment précis
- Un résumé des avis
- D'autres caractéristiques pertinentes

### Nouvelle conversation

Utilisez le bouton "Effacer l'historique" pour réinitialiser la mémoire et commencer une nouvelle session.

## Limitations actuelles

- Le système utilise des images de produits simulées pour la démonstration
- La base de connaissances est limitée aux données présentes dans le fichier CSV
- L'analyse de sentiment est basique et pourrait être améliorée

## Développement futur

- [ ] Intégration avec des APIs e-commerce pour obtenir des données produits en temps réel
- [ ] Amélioration de l'analyse de sentiment avec des modèles plus sophistiqués
- [ ] Ajout de comparaisons visuelles entre produits
- [ ] Support multilingue
- [ ] Export des résultats de recherche

## Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.

## Contact

Pour toute question ou suggestion d'amélioration, veuillez ouvrir une issue sur GitHub ou contacter l'équipe de développement à l'adresse email@example.com.

---

Développé avec ❤️ par [Votre Nom/Organisation]