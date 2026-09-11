from pathlib import Path

import joblib
import numpy as np

from fastapi import FastAPI, HTTPException

from fastapi.middleware.cors import CORSMiddleware

from api.database import (
    init_database,
    insert_review,
    get_recent_reviews,
    get_statistics
)

from api.models import ReviewCreate


# ==========================================================
# CHEMINS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"


SENTIMENT_MODEL_FILE = (
    MODELS_DIR / "sentiment_model.pkl"
)

SENTIMENT_VECTORIZER_FILE = (
    MODELS_DIR / "tfidf_vectorizer.pkl"
)

RATING_MODEL_FILE = (
    MODELS_DIR / "rating_model.pkl"
)

RATING_VECTORIZER_FILE = (
    MODELS_DIR / "rating_tfidf_vectorizer.pkl"
)


# ==========================================================
# CHARGEMENT DES MODELES
# ==========================================================

MODELS_LOADED = False

sentiment_model = None
sentiment_vectorizer = None

rating_model = None
rating_vectorizer = None


try:

    sentiment_model = joblib.load(
        SENTIMENT_MODEL_FILE
    )

    sentiment_vectorizer = joblib.load(
        SENTIMENT_VECTORIZER_FILE
    )

    rating_model = joblib.load(
        RATING_MODEL_FILE
    )

    rating_vectorizer = joblib.load(
        RATING_VECTORIZER_FILE
    )

    MODELS_LOADED = True

    print("")
    print("==========================================")
    print("      MODELES IA CHARGES AVEC SUCCES")
    print("==========================================")
    print("")


except Exception as e:

    print("")
    print("==========================================")
    print("       ERREUR CHARGEMENT MODELES")
    print("==========================================")
    print(e)
    print("")


# ==========================================================
# APPLICATION FASTAPI
# ==========================================================

app = FastAPI(

    title="API Avis Clients - Big Data IA",

    description=(
        "API temps réel pour l'analyse "
        "des avis clients avec Machine Learning"
    ),

    version="1.0.0"
)


# ==========================================================
# CORS
# ==========================================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]
)


# ==========================================================
# INITIALISATION
# ==========================================================

@app.on_event("startup")
def startup_event():

    init_database()

    print("")
    print("==========================================")
    print("       BASE SQLITE INITIALISEE")
    print("==========================================")
    print(
        f"Base : {BASE_DIR / 'data' / 'realtime_reviews.db'}"
    )
    print("")


# ==========================================================
# ROUTE RACINE
# ==========================================================

@app.get("/")
def root():

    return {

        "message":
            "API Avis Clients opérationnelle",

        "version":
            "1.0",

        "models_loaded":
            MODELS_LOADED
    }


# ==========================================================
# HEALTH CHECK
# ==========================================================

@app.get("/api/health")
def health():

    return {

        "status": "ok",

        "models_loaded":
            MODELS_LOADED
    }


# ==========================================================
# PREDICTION SENTIMENT
# ==========================================================

def predict_sentiment(text):

    vector = sentiment_vectorizer.transform(
        [text]
    )

    prediction = sentiment_model.predict(
        vector
    )[0]

    probabilities = sentiment_model.predict_proba(
        vector
    )[0]

    confidence = float(
        np.max(probabilities)
    )

    return (
        str(prediction),
        confidence
    )


# ==========================================================
# PREDICTION NOTE
# ==========================================================

def predict_rating(text):

    vector = rating_vectorizer.transform(
        [text]
    )

    prediction = rating_model.predict(
        vector
    )[0]

    probabilities = rating_model.predict_proba(
        vector
    )[0]

    confidence = float(
        np.max(probabilities)
    )

    try:

        prediction = int(
            round(float(prediction))
        )

    except Exception:

        prediction = 3

    prediction = max(
        1,
        min(5, prediction)
    )

    return (
        prediction,
        confidence
    )


# ==========================================================
# POST /api/reviews
# ==========================================================

@app.post("/api/reviews")
def create_review(
    review: ReviewCreate
):

    # ------------------------------------------------------
    # VERIFICATION MODELES
    # ------------------------------------------------------

    if not MODELS_LOADED:

        raise HTTPException(

            status_code=500,

            detail=(
                "Les modèles IA ne sont pas chargés. "
                "Vérifiez le dossier models."
            )
        )

    # ------------------------------------------------------
    # NETTOYAGE
    # ------------------------------------------------------

    text = review.text.strip()

    if not text:

        raise HTTPException(

            status_code=400,

            detail=(
                "Le texte de l'avis est obligatoire."
            )
        )

    # ------------------------------------------------------
    # SENTIMENT
    # ------------------------------------------------------

    sentiment, sentiment_confidence = (
        predict_sentiment(text)
    )

    # ------------------------------------------------------
    # NOTE
    # ------------------------------------------------------

    predicted_score, rating_confidence = (
        predict_rating(text)
    )

    # ------------------------------------------------------
    # SQLITE
    # ------------------------------------------------------

    review_id = insert_review(

        product_id=review.product_id,

        user_name=review.user_name,

        text=text,

        score=review.score,

        sentiment=sentiment,

        sentiment_confidence=(
            sentiment_confidence
        ),

        predicted_score=predicted_score,

        rating_confidence=(
            rating_confidence
        )
    )

    # ------------------------------------------------------
    # REPONSE
    # ------------------------------------------------------

    return {

        "message":
            "Avis enregistré avec succès",

        "id":
            review_id,

        "product_id":
            review.product_id,

        "user_name":
            review.user_name,

        "text":
            text,

        "score":
            review.score,

        "sentiment":
            sentiment,

        "sentiment_confidence":
            round(
                sentiment_confidence * 100,
                2
            ),

        "predicted_score":
            predicted_score,

        "rating_confidence":
            round(
                rating_confidence * 100,
                2
            )
    }


# ==========================================================
# GET /api/reviews
# ==========================================================

@app.get("/api/reviews")
def reviews(
    limit: int = 20
):

    if limit < 1:

        limit = 1

    if limit > 100:

        limit = 100

    data = get_recent_reviews(
        limit
    )

    return {

        "count":
            len(data),

        "reviews":
            data
    }


# ==========================================================
# GET /api/reviews/stats
# ==========================================================

@app.get("/api/reviews/stats")
def review_statistics():

    return get_statistics()


# ==========================================================
# EXECUTION DIRECTE
# ==========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        "api.main:app",

        host="127.0.0.1",

        port=8000,

        reload=True
    )