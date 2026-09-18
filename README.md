# 📊 Analyse Big Data des Avis Clients

Projet académique de **Master 1 Intelligence Artificielle** réalisé dans le cadre de l'étude du **Big Data, NLP, Machine Learning et analyse prédictive**.

L'objectif du projet est d'analyser un très grand volume d'avis clients afin d'identifier les tendances de satisfaction, les sentiments exprimés, les thèmes récurrents et les produits les plus appréciés. Le projet intègre également des modèles de Machine Learning permettant de prédire le sentiment et la note d'un nouvel avis, ainsi qu'un système de prévision de la satisfaction future.

---

## 🎯 Objectif du projet

Les avis clients représentent une source importante d'informations pour comprendre la satisfaction des consommateurs.

Cependant, lorsqu'un grand nombre d'avis est disponible, une analyse manuelle devient impossible.

Ce projet propose donc une solution permettant de :

* analyser automatiquement les avis clients ;
* nettoyer et préparer les données ;
* analyser les sentiments des clients ;
* identifier les principaux thèmes abordés ;
* prédire le sentiment d'un nouvel avis ;
* prédire la note attribuée à un produit ;
* analyser l'évolution de la satisfaction dans le temps ;
* utiliser PySpark pour le traitement Big Data ;
* prévoir l'évolution future de la satisfaction ;
* exposer les modèles via une API REST ;
* visualiser les résultats dans un dashboard interactif ;
* analyser de nouveaux avis en temps réel.

---

## 📌 Problématique

**Comment exploiter un grand volume d'avis clients afin d'identifier automatiquement la satisfaction des utilisateurs, comprendre leurs principales préoccupations et prévoir l'évolution future de cette satisfaction ?**

---

## 🗂️ Dataset

Le projet utilise le dataset :

**Amazon Fine Food Reviews**

Le dataset original contient :

* **568 454 avis**
* **10 colonnes**
* des informations sur les produits, utilisateurs, notes et commentaires.

Après nettoyage :

* **568 453 avis exploitables**

Le dataset contient notamment :

```text
Id
ProductId
UserId
ProfileName
Score
Time
Date
Summary
Text
HelpfulnessNumerator
HelpfulnessDenominator
```

### ⚠️ Dataset volumineux

Le fichier nettoyé `reviews_clean.csv` fait plus de **500 MB**.

Pour cette raison, il n'est pas stocké dans le dépôt GitHub.

Il doit être placé localement dans :

```text
data/processed/reviews_clean.csv
```

Le dataset brut est également exclu du dépôt GitHub.

---

# 🏗️ Architecture du projet

```text
                         ┌─────────────────────┐
                         │ Amazon Fine Food    │
                         │ Reviews Dataset     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Nettoyage des       │
                         │ données             │
                         │ Python / Pandas     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Analyse exploratoire│
                         │ + NLP               │
                         └──────────┬──────────┘
                                    │
                   ┌────────────────┼────────────────┐
                   │                │                │
                   ▼                ▼                ▼
             Machine Learning   PySpark        Analyse temporelle
                   │                │                │
                   ▼                ▼                ▼
             Prédiction         Big Data       Prévision
             sentiment/note                    satisfaction
                   │                                 │
                   └──────────────┬──────────────────┘
                                  │
                                  ▼
                         ┌─────────────────────┐
                         │      FastAPI        │
                         │    REST API         │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │     Streamlit       │
                         │     Dashboard       │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Analyse des avis    │
                         │ en temps réel       │
                         └─────────────────────┘
```

---

# 🛠️ Technologies utilisées

## Langage

* Python 3
* SQL

## Analyse de données

* Pandas
* NumPy
* Matplotlib
* Seaborn

## NLP

* NLTK
* TF-IDF
* WordCloud
* traitement et nettoyage du texte

## Machine Learning

* Scikit-learn
* Logistic Regression
* TF-IDF Vectorizer

## Big Data

* Apache Spark
* PySpark
* Hadoop / Winutils pour l'environnement Windows

## Prévision

* Statsmodels
* Holt-Winters / Exponential Smoothing

## API

* FastAPI
* Uvicorn
* Pydantic

## Dashboard

* Streamlit
* Plotly

## Base de données

* PostgreSQL
* psycopg2 (pool de connexions)
* python-dotenv

## Gestion du projet

* Git
* GitHub

---

# 📁 Structure du projet

```text
big-data-analyse-avis-clients/
│
├── data/
│   ├── raw/
│   │   └── reviews.csv
│   │
│   ├── processed/
│   │   └── reviews_clean.csv
│   │
│   └── sample/
│
├── notebooks/
│
├── src/
│   ├── 01_verification.py
│   ├── 02_nettoyage.py
│   ├── 03_analyse.py
│   ├── 04_nlp.py
│   ├── 05_prediction_sentiment.py
│   ├── 06_prediction_note.py
│   ├── 07_prediction_complete.py
│   ├── 08_analyse_temporelle.py
│   ├── 09_prevision_satisfaction.py
│   ├── 10_pyspark_analyse.py
│   ├── 11_preparation_dashboard.py
│   └── 12_simulateur_avis.py
│
├── models/
│   ├── sentiment_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── rating_model.pkl
│   └── rating_tfidf_vectorizer.pkl
│
├── results/
│   ├── figures/
│   └── predictions/
│
├── dashboard/
│   ├── data/
│   ├── figures/
│   └── app.py
│
├── api/
│   ├── __init__.py
│   ├── database.py
│   ├── models.py
│   └── main.py
│
├── requirements.txt
├── .env
├── README.md
└── .gitignore
```

---

# 🔄 Pipeline de traitement

Le projet est organisé en plusieurs étapes.

## 1. Vérification des données

Script :

```text
src/01_verification.py
```

Cette étape permet de vérifier :

* le nombre de lignes ;
* le nombre de colonnes ;
* les valeurs manquantes ;
* les doublons ;
* les types de données.

---

## 2. Nettoyage des données

Script :

```text
src/02_nettoyage.py
```

Les opérations principales sont :

* suppression des données invalides ;
* traitement des valeurs manquantes ;
* nettoyage du texte ;
* conversion des dates ;
* création des variables `Annee` et `Mois` ;
* création de la variable `Sentiment`.

La classification utilisée est :

```text
Score 1-2 → negatif
Score 3   → neutre
Score 4-5 → positif
```

---

# 📊 Analyse exploratoire

Script :

```text
src/03_analyse.py
```

L'analyse permet d'étudier :

* la distribution des notes ;
* le nombre d'avis par année ;
* la note moyenne ;
* la répartition des sentiments ;
* l'évolution des avis dans le temps.

Les graphiques sont enregistrés dans :

```text
results/figures/
```

---

# 🧠 Analyse NLP

Script :

```text
src/04_nlp.py
```

L'analyse NLP permet d'identifier :

* les mots les plus fréquents ;
* les mots associés aux avis positifs ;
* les mots associés aux avis négatifs ;
* les principaux thèmes abordés par les clients.

### Principaux mots identifiés

Parmi les mots les plus fréquents :

```text
like
good
taste
great
coffee
flavor
tea
love
price
buy
best
```

### Principaux thèmes

Les thèmes détectés sont notamment :

* Qualité
* Goût
* Prix
* Quantité
* Emballage
* Livraison
* Service

---

# 🤖 Machine Learning — Prédiction du sentiment

Script :

```text
src/05_prediction_sentiment.py
```

Le modèle utilisé est une **régression logistique multiclasses** avec une représentation textuelle **TF-IDF**.

Configuration principale :

```text
200 000 avis utilisés pour l'apprentissage
160 000 avis pour l'entraînement
40 000 avis pour le test
30 000 caractéristiques TF-IDF
N-grams : 1 à 2
```

## Résultats

### Accuracy

**89,73 %**

### F1-score

```text
Négatif  : 78,44 %
Neutre   : 42,70 %
Positif  : 95,05 %
```

### F1-score pondéré

**88,75 %**

Le modèle reconnaît particulièrement bien les avis positifs, qui constituent la majorité du dataset.

---

# ⭐ Machine Learning — Prédiction de la note

Script :

```text
src/06_prediction_note.py
```

Le modèle tente de prédire directement la note :

```text
1 → 5
```

à partir du texte de l'avis.

## Résultat

Accuracy :

**77,48 %**

F1-score pondéré :

**75,58 %**

Le modèle obtient de meilleurs résultats sur les notes très fréquentes, notamment la note **5**.

---

# 🔮 Analyse complète d'un nouvel avis

Script :

```text
src/07_prediction_complete.py
```

Ce module permet de fournir un texte et d'obtenir :

```text
Sentiment
Confiance du sentiment
Note prédite
Confiance de la prédiction
```

Exemple :

```text
Avis :
"The product tastes great and I really love it."

Résultat :
Sentiment : positif
Note prédite : 5
```

---

# 📅 Analyse temporelle

Script :

```text
src/08_analyse_temporelle.py
```

Cette étape étudie l'évolution :

* du nombre d'avis ;
* de la note moyenne ;
* des sentiments ;
* des notes mensuelles.

Les données montrent une forte augmentation du volume d'avis au fil des années.

---

# 📈 Prévision de la satisfaction

Script :

```text
src/09_prevision_satisfaction.py
```

Le modèle utilisé est :

**Holt-Winters / Exponential Smoothing**

avec :

* tendance additive ;
* tendance amortie ;
* saisonnalité additive ;
* saisonnalité de 12 mois.

## Résultats

Satisfaction moyenne historique :

**4,31 / 5**

MAE :

**0,1359**

RMSE :

**0,1548**

Satisfaction moyenne prévue sur les 12 prochains mois :

**4,09 / 5**

Variation prévue :

**-4,91 %**

Cette prévision représente une **tendance estimée par le modèle et non une certitude**.

---

# ⚡ Analyse Big Data avec PySpark

Script :

```text
src/10_pyspark_analyse.py
```

L'objectif est de démontrer l'utilisation d'un framework Big Data capable de traiter efficacement un grand volume de données.

Technologie :

```text
PySpark 4.0.1
```

Les analyses réalisées comprennent :

* nombre d'avis par année ;
* note moyenne par année ;
* répartition des sentiments ;
* produits les plus évalués ;
* avis les plus utiles.

### Exemple

Le produit :

```text
B007JFMH8M
```

possède :

```text
913 avis
Note moyenne : 4,58 / 5
```

---

# 📊 Dashboard Streamlit

Le dashboard est disponible dans :

```text
dashboard/app.py
```

Il permet de visualiser plusieurs parties du projet.

### Pages disponibles

```text
🏠 Vue générale
📈 Évolution temporelle
📦 Produits
💬 Thèmes
🤖 IA & Prévision
📡 Avis en temps réel
📊 Analyse en temps réel
```

Le dashboard permet notamment de consulter :

* les statistiques générales ;
* la distribution des notes ;
* l'évolution temporelle ;
* les produits ;
* les thèmes ;
* les prédictions IA ;
* les prévisions ;
* les nouveaux avis enregistrés.

---

# 🌐 API FastAPI

L'API se trouve dans :

```text
api/
```

Elle permet d'exposer les modèles Machine Learning sous forme d'API REST.

## Démarrage

Depuis la racine du projet :

```powershell
python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

API :

```text
http://127.0.0.1:8000
```

Documentation interactive :

```text
http://127.0.0.1:8000/docs
```

---

# 🔌 Principaux endpoints

## Vérification de l'API

```http
GET /
```

## Health check

```http
GET /api/health
```

## Récupérer les avis

```http
GET /api/reviews
```

## Statistiques

```http
GET /api/reviews/stats
```

## Ajouter un nouvel avis

```http
POST /api/reviews
```

Exemple :

```json
{
    "product_id": "TEST001",
    "user_name": "Utilisateur",
    "text": "This product is excellent and tastes great.",
    "score": 5
}
```

L'API retourne notamment :

```text
sentiment
sentiment_confidence
predicted_score
rating_confidence
```

---

# 📡 Analyse en temps réel

Le système permet d'ajouter automatiquement de nouveaux avis.

Script :

```text
src/12_simulateur_avis.py
```

Le simulateur génère des avis en anglais afin de rester cohérent avec le vocabulaire utilisé lors de l'entraînement des modèles.

Chaque avis est envoyé à l'API :

```text
Simulateur
     ↓
FastAPI
     ↓
Modèle Sentiment
     ↓
Modèle Note
     ↓
PostgreSQL
     ↓
Dashboard Streamlit
```

Les statistiques sont ensuite actualisées dans le dashboard.

---

# 🗄️ Base de données

Les avis générés en temps réel sont enregistrés dans **PostgreSQL**, via un pool de connexions (`api/database.py`).

La connexion est configurée par variables d'environnement, à définir dans un fichier `.env` à la racine du projet :

```text
DB_HOST=127.0.0.1
DB_PORT=5433
DB_NAME=bigdata_reviews
DB_USER=postgres
DB_PASSWORD=votre_mot_de_passe
```

`DB_PASSWORD` est obligatoire : au démarrage, l'API refuse de se lancer si cette variable n'est pas définie, plutôt que d'échouer plus tard avec une erreur peu claire.

La table `reviews` contient notamment :

```text
id
product_id
user_name
text
score
sentiment
sentiment_confidence
predicted_score
rating_confidence
created_at
```

Le fichier `.env` est exclu du dépôt GitHub car il contient des identifiants de connexion.

---

# 🚀 Installation

## 1. Cloner le projet

```powershell
git clone https://github.com/Ranto-nyaina/Big-data-analyse-avis-clients.git
```

Puis :

```powershell
cd Big-data-analyse-avis-clients
```

---

## 2. Créer l'environnement virtuel

Windows :

```powershell
python -m venv venv
```

Activer l'environnement :

```powershell
.\venv\Scripts\Activate.ps1
```

---

## 3. Installer les dépendances

```powershell
python -m pip install -r requirements.txt
```

---

## 4. Configurer les variables d'environnement

Créer un fichier `.env` à la racine du projet :

```text
DB_HOST=127.0.0.1
DB_PORT=5433
DB_NAME=bigdata_reviews
DB_USER=postgres
DB_PASSWORD=votre_mot_de_passe
```

Optionnel — si l'API ou les fichiers du dashboard ne sont pas à leur emplacement par défaut :

```text
DASHBOARD_API_URL=http://127.0.0.1:8000
DASHBOARD_DATA_DIR=chemin/vers/dashboard/data
DASHBOARD_FIGURES_DIR=chemin/vers/dashboard/figures
DASHBOARD_MODELS_DIR=chemin/vers/models
DASHBOARD_RESULTS_DIR=chemin/vers/results
```

Sans ces variables optionnelles, le dashboard utilise les mêmes chemins par défaut qu'auparavant.

---

# 📂 Ajouter le dataset

Le dataset Amazon Fine Food Reviews n'est pas inclus dans GitHub en raison de sa taille.

Le fichier nettoyé doit être placé ici :

```text
data/processed/reviews_clean.csv
```

Le fichier original peut être placé dans :

```text
data/raw/reviews.csv
```

---

# ⚙️ Configuration PySpark sous Windows

Le projet utilise :

```text
Java 17
PySpark 4.0.1
Hadoop / Winutils
```

Exemple de configuration :

```powershell
$env:JAVA_HOME="C:\Program Files\Java\jdk-17.0.12"
$env:SPARK_HOME="$env:VIRTUAL_ENV\Lib\site-packages\pyspark"
$env:HADOOP_HOME="C:\hadoop"
$env:hadoop_home_dir="C:\hadoop"

$env:Path="$env:JAVA_HOME\bin;$env:SPARK_HOME\bin;$env:HADOOP_HOME\bin;$env:Path"
```

Vérification :

```powershell
python -c "from pyspark.sql import SparkSession; spark=SparkSession.builder.master('local[*]').getOrCreate(); print(spark.version); spark.stop()"
```

---

# ▶️ Exécution du projet

## Analyse des données

```powershell
python src/01_verification.py
python src/02_nettoyage.py
python src/03_analyse.py
python src/04_nlp.py
```

## Machine Learning

```powershell
python src/05_prediction_sentiment.py
python src/06_prediction_note.py
python src/07_prediction_complete.py
```

## Analyse temporelle et prévision

```powershell
python src/08_analyse_temporelle.py
python src/09_prevision_satisfaction.py
```

## PySpark

```powershell
python src/10_pyspark_analyse.py
```

## Préparation du dashboard

```powershell
python src/11_preparation_dashboard.py
```

---

# 🌐 Lancer l'API

Terminal 1 :

```powershell
python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

---

# 📊 Lancer le dashboard

Terminal 2 :

```powershell
streamlit run dashboard/app.py
```

Streamlit ouvre automatiquement le dashboard dans le navigateur.

---

# 📡 Lancer le simulateur

Terminal 3 :

```powershell
python src/12_simulateur_avis.py
```

Le simulateur envoie alors automatiquement de nouveaux avis vers l'API.

---

# 📊 Résultats principaux

| Indicateur               |     Résultat |
| ------------------------ | -----------: |
| Avis analysés            |  **568 453** |
| Note moyenne             | **4,18 / 5** |
| Avis positifs            |  **443 776** |
| Avis négatifs            |   **82 037** |
| Avis neutres             |   **42 640** |
| Taux positif             |  **78,07 %** |
| Taux négatif             |  **14,43 %** |
| Taux neutre              |   **7,50 %** |
| Accuracy sentiment       |  **89,73 %** |
| Accuracy prédiction note |  **77,48 %** |
| Satisfaction historique  | **4,31 / 5** |
| Satisfaction prévue      | **4,09 / 5** |
| Variation prévue         |  **-4,91 %** |

---

# 🔍 Interprétation

L'analyse montre que la majorité des avis sont positifs.

La note moyenne globale est de :

```text
4,18 / 5
```

Les avis positifs représentent environ :

```text
78,07 %
```

Le modèle de sentiment atteint une accuracy de :

```text
89,73 %
```

ce qui montre une bonne capacité à distinguer les sentiments exprimés dans les avis.

La prédiction directe de la note est plus difficile, avec une accuracy de :

```text
77,48 %
```

Cela s'explique notamment par la difficulté à distinguer certaines notes proches comme 3 et 4 ou 4 et 5 uniquement à partir du texte.

Enfin, le modèle temporel prévoit une satisfaction moyenne d'environ :

```text
4,09 / 5
```

sur les 12 mois suivants, contre une moyenne historique de :

```text
4,31 / 5
```

Il indique donc une tendance potentielle à la baisse de la satisfaction.

---

# ⚠️ Limites du projet

Le projet présente plusieurs limites :

* les données proviennent d'Amazon et représentent un contexte particulier ;
* le modèle de sentiment est principalement entraîné sur des avis en anglais ;
* la classe neutre est moins bien reconnue que les classes positive et négative ;
* la prédiction exacte d'une note est plus difficile que la prédiction du sentiment ;
* la détection des thèmes repose en partie sur des mots-clés ;
* les prévisions temporelles sont basées sur les tendances historiques ;
* la prévision ne constitue pas une certitude sur le comportement futur des clients ;
* le dashboard utilise actuellement une architecture locale.

---

# 🚀 Perspectives d'amélioration

Plusieurs améliorations sont possibles :

### Machine Learning

* utiliser des modèles Transformer ;
* tester BERT ou RoBERTa ;
* améliorer la classification des avis neutres ;
* effectuer une optimisation plus poussée des hyperparamètres.

### NLP

* améliorer la détection automatique des thèmes ;
* utiliser LDA ou BERTopic ;
* détecter les aspects précis d'un produit ;
* réaliser une analyse d'opinion par aspect.

### Big Data

* utiliser un cluster Spark distribué ;
* utiliser HDFS ;
* intégrer Kafka pour le streaming ;
* traiter les avis en continu.

### Architecture

* déployer FastAPI sur un serveur ;
* déployer Streamlit dans le cloud ;
* utiliser Docker ;
* mettre en place une pipeline CI/CD.

### Prédiction

* prédire les tendances de satisfaction ;
* identifier les produits susceptibles de recevoir davantage d'avis négatifs ;
* recommander des améliorations de produits ;
* détecter automatiquement les problèmes récurrents.

---

# 🔐 Données et confidentialité

Les fichiers volumineux et les fichiers générés localement sont volontairement exclus du dépôt GitHub.

Le fichier `.gitignore` exclut notamment :

```text
venv/
data/raw/
data/processed/reviews_clean.csv
.env
results/pyspark/
```

Les modèles Machine Learning sont conservés dans le dépôt car leur taille totale reste raisonnable et ils permettent d'exécuter directement les prédictions sans réentraîner les modèles.

---

# 📚 Compétences mises en œuvre

Ce projet permet de mettre en pratique :

* Big Data
* Data Engineering
* Data Analysis
* Natural Language Processing
* Machine Learning
* Classification
* Time Series Forecasting
* PySpark
* API REST
* FastAPI
* Streamlit
* PostgreSQL
* Git
* GitHub
* Visualisation de données

---

# 👨‍🎓 Contexte académique

**Formation :** Master 1 Intelligence Artificielle
**Projet :** Analyse Big Data des avis clients
**Domaine :** Big Data / IA / NLP / Machine Learning

---

# 📌 Conclusion

Ce projet propose une chaîne complète d'analyse Big Data appliquée aux avis clients.

À partir de plus de **568 000 avis**, le système permet de nettoyer les données, analyser les textes, identifier les sentiments et les thèmes, effectuer des prédictions avec Machine Learning, traiter les données avec PySpark et prévoir l'évolution future de la satisfaction.

L'intégration de **FastAPI** et **Streamlit** permet ensuite de transformer les modèles et analyses en une application interactive capable de traiter également de nouveaux avis en temps réel.

Le projet constitue ainsi une démonstration complète d'une chaîne **Data → NLP → Machine Learning → Big Data → API → Dashboard → Temps réel**.
