import joblib
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

SENTIMENT_MODEL = "models/sentiment_model.pkl"
SENTIMENT_VECTORIZER = "models/tfidf_vectorizer.pkl"

RATING_MODEL = "models/rating_model.pkl"
RATING_VECTORIZER = "models/rating_tfidf_vectorizer.pkl"


# ============================================================
# 1. CHARGEMENT DES MODÈLES
# ============================================================

print("=" * 60)
print("ÉTAPE 07 - PRÉDICTION COMPLÈTE")
print("=" * 60)

print("\nChargement des modèles...")

sentiment_model = joblib.load(
    SENTIMENT_MODEL
)

sentiment_vectorizer = joblib.load(
    SENTIMENT_VECTORIZER
)

rating_model = joblib.load(
    RATING_MODEL
)

rating_vectorizer = joblib.load(
    RATING_VECTORIZER
)

print("Modèles chargés avec succès.")


# ============================================================
# 2. FONCTION DE PRÉDICTION
# ============================================================

def analyser_avis(avis):

    # --------------------------------------------------------
    # SENTIMENT
    # --------------------------------------------------------

    texte_sentiment = sentiment_vectorizer.transform(
        [avis]
    )

    sentiment = sentiment_model.predict(
        texte_sentiment
    )[0]

    probabilites_sentiment = (
        sentiment_model.predict_proba(
            texte_sentiment
        )[0]
    )

    confiance_sentiment = np.max(
        probabilites_sentiment
    )


    # --------------------------------------------------------
    # NOTE
    # --------------------------------------------------------

    texte_rating = rating_vectorizer.transform(
        [avis]
    )

    note = rating_model.predict(
        texte_rating
    )[0]

    probabilites_rating = (
        rating_model.predict_proba(
            texte_rating
        )[0]
    )

    confiance_note = np.max(
        probabilites_rating
    )


    return (
        sentiment,
        confiance_sentiment,
        note,
        confiance_note
    )


# ============================================================
# 3. TESTS AUTOMATIQUES
# ============================================================

print("\n" + "=" * 60)
print("TESTS AUTOMATIQUES")
print("=" * 60)


avis_tests = [

    "This product is absolutely amazing. "
    "The taste is excellent and I love it.",

    "This product is terrible. "
    "The taste is horrible and I will never buy it again.",

    "The product is okay. "
    "It is not bad but nothing special.",

    "I really like this product. "
    "It works perfectly and I will buy it again.",

    "Very disappointing product. "
    "The quality is poor and it is too expensive."
]


for numero, avis in enumerate(
    avis_tests,
    start=1
):

    sentiment, conf_sentiment, note, conf_note = (
        analyser_avis(avis)
    )

    print("\n" + "-" * 60)

    print(f"Test {numero}")

    print("\nAvis :")
    print(avis)

    print(
        f"\nSentiment prédit : {sentiment}"
    )

    print(
        f"Confiance sentiment : "
        f"{conf_sentiment * 100:.2f}%"
    )

    print(
        f"Note prédite : {note}/5"
    )

    print(
        f"Confiance note : "
        f"{conf_note * 100:.2f}%"
    )


# ============================================================
# 4. MODE INTERACTIF
# ============================================================

print("\n" + "=" * 60)
print("MODE INTERACTIF")
print("=" * 60)

print(
    "\nÉcris un avis client pour obtenir une prédiction."
)

print(
    "Tape 'exit' pour quitter."
)


while True:

    avis = input(
        "\nNouvel avis : "
    )

    if avis.lower() == "exit":
        break

    if not avis.strip():
        print("Veuillez saisir un avis.")
        continue

    sentiment, conf_sentiment, note, conf_note = (
        analyser_avis(avis)
    )

    print("\nRésultat")
    print("-" * 40)

    print(
        f"Sentiment : {sentiment}"
    )

    print(
        f"Confiance sentiment : "
        f"{conf_sentiment * 100:.2f}%"
    )

    print(
        f"Note prédite : {note}/5"
    )

    print(
        f"Confiance note : "
        f"{conf_note * 100:.2f}%"
    )


print("\n" + "=" * 60)
print("ÉTAPE 07 TERMINÉE")
print("=" * 60)