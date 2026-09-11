import sqlite3
from pathlib import Path


# ==========================================================
# CONFIGURATION
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_FILE = DATA_DIR / "realtime_reviews.db"


# ==========================================================
# CONNEXION SQLITE
# ==========================================================

def get_connection():

    connection = sqlite3.connect(
        DATABASE_FILE,
        check_same_thread=False
    )

    connection.row_factory = sqlite3.Row

    return connection


# ==========================================================
# INITIALISATION DE LA BASE
# ==========================================================

def init_database():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reviews (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            product_id TEXT,

            user_name TEXT,

            text TEXT NOT NULL,

            score INTEGER,

            sentiment TEXT,

            sentiment_confidence REAL,

            predicted_score INTEGER,

            rating_confidence REAL,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

        )
    """)

    connection.commit()

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

        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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

    connection.commit()

    review_id = cursor.lastrowid

    connection.close()

    return review_id


# ==========================================================
# DERNIERS AVIS
# ==========================================================

def get_recent_reviews(limit=20):

    connection = get_connection()

    cursor = connection.cursor()

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

        LIMIT ?
        """,

        (limit,)
    )

    rows = cursor.fetchall()

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

    # Total
    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM reviews
        """
    )

    total = cursor.fetchone()["total"]

    # Positifs
    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM reviews
        WHERE sentiment = 'positif'
        """
    )

    positif = cursor.fetchone()["total"]

    # Neutres
    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM reviews
        WHERE sentiment = 'neutre'
        """
    )

    neutre = cursor.fetchone()["total"]

    # Négatifs
    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM reviews
        WHERE sentiment = 'negatif'
        """
    )

    negatif = cursor.fetchone()["total"]

    # Note moyenne
    cursor.execute(
        """
        SELECT AVG(score) AS moyenne
        FROM reviews
        WHERE score IS NOT NULL
        """
    )

    moyenne = cursor.fetchone()["moyenne"]

    connection.close()

    if moyenne is None:
        moyenne = 0

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
            round(moyenne, 2)
    }