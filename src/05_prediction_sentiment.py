import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/processed/reviews_clean.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

# Nombre maximum d'avis utilisés
MAX_REVIEWS = 200000

# ============================================================
# 1. CHARGEMENT
# ============================================================

print("=" * 60)
print("ÉTAPE 05 - PRÉDICTION DU SENTIMENT")
print("=" * 60)

print("\n1. Chargement des données...")

df = pd.read_csv(
    INPUT_FILE,
    usecols=["TextClean", "Sentiment"]
)

print(f"Nombre total d'avis disponibles : {len(df)}")

# ============================================================
# 2. NETTOYAGE
# ============================================================

print("\n2. Préparation des données...")

df = df.dropna(subset=["TextClean", "Sentiment"])

df = df[
    df["TextClean"].str.strip() != ""
]

print(f"Avis disponibles après nettoyage : {len(df)}")

# ============================================================
# 3. ÉCHANTILLONNAGE
# ============================================================

if len(df) > MAX_REVIEWS:

    print(
        f"\nLe dataset est important ({len(df)} avis)."
    )

    print(
        f"Échantillonnage de {MAX_REVIEWS} avis..."
    )

    df = df.sample(
        n=MAX_REVIEWS,
        random_state=42
    )

print(f"Nombre d'avis utilisés : {len(df)}")

print("\nRépartition des sentiments :")

print(
    df["Sentiment"]
    .value_counts()
)

# ============================================================
# 4. VARIABLES
# ============================================================

X = df["TextClean"]

y = df["Sentiment"]

# ============================================================
# 5. TRAIN / TEST
# ============================================================

print("\n3. Séparation entraînement / test...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Entraînement : {len(X_train)} avis")
print(f"Test : {len(X_test)} avis")

# ============================================================
# 6. TF-IDF
# ============================================================

print("\n4. Transformation TF-IDF...")

vectorizer = TfidfVectorizer(
    max_features=30000,
    min_df=3,
    max_df=0.95,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)

print(
    f"Dimension TF-IDF entraînement : "
    f"{X_train_tfidf.shape}"
)

print("Transformation du jeu de test...")

X_test_tfidf = vectorizer.transform(X_test)

print(
    f"Dimension TF-IDF test : "
    f"{X_test_tfidf.shape}"
)

# ============================================================
# 7. ENTRAÎNEMENT
# ============================================================

print("\n5. Entraînement du modèle...")

model = LogisticRegression(
    max_iter=1000,
    C=2.0,
    solver="lbfgs"
)

model.fit(
    X_train_tfidf,
    y_train
)

print("Entraînement terminé.")

# ============================================================
# 8. PRÉDICTION
# ============================================================

print("\n6. Prédiction sur le jeu de test...")

y_pred = model.predict(
    X_test_tfidf
)

# ============================================================
# 9. ÉVALUATION
# ============================================================

print("\n" + "=" * 60)
print("RÉSULTATS DU MODÈLE")
print("=" * 60)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(
    f"\nAccuracy : {accuracy:.4f}"
)

print(
    f"Accuracy en pourcentage : "
    f"{accuracy * 100:.2f}%"
)

print("\nClassification Report :")

print(
    classification_report(
        y_test,
        y_pred,
        digits=4
    )
)

# ============================================================
# 10. MATRICE DE CONFUSION
# ============================================================

print("\nMatrice de confusion :")

labels = [
    "negatif",
    "neutre",
    "positif"
]

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

print(cm_df)

# ============================================================
# 11. SAUVEGARDE DU MODÈLE
# ============================================================

print("\n7. Sauvegarde du modèle...")

model_path = os.path.join(
    MODEL_DIR,
    "sentiment_model.pkl"
)

vectorizer_path = os.path.join(
    MODEL_DIR,
    "tfidf_vectorizer.pkl"
)

joblib.dump(
    model,
    model_path
)

joblib.dump(
    vectorizer,
    vectorizer_path
)

print(
    f"Modèle sauvegardé : {model_path}"
)

print(
    f"Vectoriseur sauvegardé : {vectorizer_path}"
)

# ============================================================
# 12. TEST SUR DE NOUVEAUX AVIS
# ============================================================

print("\n" + "=" * 60)
print("TEST SUR DE NOUVEAUX AVIS")
print("=" * 60)

nouveaux_avis = [
    "This product is excellent, I really love it.",
    "Very bad product, I am disappointed.",
    "The product is okay, nothing special."
]

nouveaux_tfidf = vectorizer.transform(
    nouveaux_avis
)

predictions = model.predict(
    nouveaux_tfidf
)

probabilites = model.predict_proba(
    nouveaux_tfidf
)

for i, avis in enumerate(nouveaux_avis):

    sentiment = predictions[i]

    confiance = np.max(
        probabilites[i]
    )

    print("\nAvis :")
    print(avis)

    print(
        f"Sentiment prédit : {sentiment}"
    )

    print(
        f"Confiance : {confiance * 100:.2f}%"
    )

# ============================================================
# FIN
# ============================================================

print("\n" + "=" * 60)
print("ÉTAPE 05 TERMINÉE")
print("=" * 60)