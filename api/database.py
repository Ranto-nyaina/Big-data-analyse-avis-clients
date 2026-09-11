import os

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv


# ==========================================================
# CHARGEMENT DES VARIABLES D'ENVIRONNEMENT
# ==========================================================

load_dotenv()


# ==========================================================
# CONFIGURATION POSTGRESQL
# ==========================================================

DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "5433")
DB_NAME = os.getenv("DB_NAME", "bigdata_reviews")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD")


# ==========================================================
# CONNEXION POSTGRESQL
# ==========================================================

def get_connection():

    connection = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

    return connection


# ==========================================================
# INITIALISATION DE LA BASE
# ==========================================================

def init_database():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reviews (

            id SERIAL PRIMARY KEY,

            product_id TEXT,

            user_name TEXT,

            text TEXT NOT NULL,

            score INTEGER,

            sentiment TEXT,

            sentiment_confidence DOUBLE PRECISION,

            predicted_score INTEGER,

            rating_confidence DOUBLE PRECISION,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

        )
    """)

    connection.commit()

    cursor.close()
    connection.close()


# ==========================================================
# AJOUT D'UN AVIS
# ==========================================================

def insert_review(
    product_id,
    user_name,
    text,
    score,
    sentiment,
    sentiment_confidence,
    predicted_score,
    rating_confidence
):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO reviews (

            product_id,
            user_name,
            text,
            score,
            sentiment,
            sentiment_confidence,
            predicted_score,
            rating_confidence

        )

        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)

        RETURNING id
        """,

        (
            product_id,
            user_name,
            text,
            score,
            sentiment,
            sentiment_confidence,
            predicted_score,
            rating_confidence
        )
    )

    review_id = cursor.fetchone()[0]

    connection.commit()

    cursor.close()
    connection.close()

    return review_id


# ==========================================================
# DERNIERS AVIS
# ==========================================================

def get_recent_reviews(limit=20):

    connection = get_connection()

    cursor = connection.cursor(
        cursor_factory=RealDictCursor
    )

    cursor.execute(
        """
        SELECT
            id,
            product_id,
            user_name,
            text,
            score,
            sentiment,
            sentiment_confidence,
            predicted_score,
            rating_confidence,
            created_at

        FROM reviews

        ORDER BY id DESC

        LIMIT %s
        """,

        (limit,)
    )

    rows = cursor.fetchall()

    cursor.close()
    connection.close()

    return [
        dict(row)
        for row in rows
    ]


# ==========================================================
# STATISTIQUES
# ==========================================================

def get_statistics():

    connection = get_connection()

    cursor = connection.cursor()

    # ======================================================
    # TOTAL
    # ======================================================

    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM reviews
        """
    )

    total = cursor.fetchone()[0]


    # ======================================================
    # POSITIFS
    # ======================================================

    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM reviews
        WHERE sentiment = 'positif'
        """
    )

    positif = cursor.fetchone()[0]


    # ======================================================
    # NEUTRES
    # ======================================================

    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM reviews
        WHERE sentiment = 'neutre'
        """
    )

    neutre = cursor.fetchone()[0]


    # ======================================================
    # NEGATIFS
    # ======================================================

    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM reviews
        WHERE sentiment = 'negatif'
        """
    )

    negatif = cursor.fetchone()[0]


    # ======================================================
    # NOTE MOYENNE
    # ======================================================

    cursor.execute(
        """
        SELECT AVG(score) AS moyenne
        FROM reviews
        WHERE score IS NOT NULL
        """
    )

    moyenne = cursor.fetchone()[0]


    cursor.close()
    connection.close()


    # ======================================================
    # VALEUR PAR DEFAUT
    # ======================================================

    if moyenne is None:
        moyenne = 0


    # ======================================================
    # POURCENTAGES
    # ======================================================

    if total > 0:

        pourcentage_positif = (
            positif / total
        ) * 100

        pourcentage_neutre = (
            neutre / total
        ) * 100

        pourcentage_negatif = (
            negatif / total
        ) * 100

    else:

        pourcentage_positif = 0
        pourcentage_neutre = 0
        pourcentage_negatif = 0


    # ======================================================
    # RESULTAT
    # ======================================================

    return {

        "total": total,

        "positif": positif,

        "neutre": neutre,

        "negatif": negatif,

        "pourcentage_positif":
            round(pourcentage_positif, 2),

        "pourcentage_neutre":
            round(pourcentage_neutre, 2),

        "pourcentage_negatif":
            round(pourcentage_negatif, 2),

        "note_moyenne":
            round(float(moyenne), 2)

    }