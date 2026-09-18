import os
import logging
from contextlib import contextmanager

import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv


# ==========================================================
# CHARGEMENT DES VARIABLES D'ENVIRONNEMENT
# ==========================================================

load_dotenv()

logger = logging.getLogger(__name__)


# ==========================================================
# CONFIGURATION POSTGRESQL
# ==========================================================

DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "5433")
DB_NAME = os.getenv("DB_NAME", "bigdata_reviews")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# Bornes du pool de connexions. À ajuster selon la charge réelle
# (nombre d'onglets/utilisateurs simultanés sur le dashboard).
POOL_MIN_CONNECTIONS = 1
POOL_MAX_CONNECTIONS = 10


# ==========================================================
# EXCEPTIONS
# ==========================================================

class DatabaseError(Exception):
    """Erreur générique remontée par la couche base de données."""


class ConfigurationError(DatabaseError):
    """Variable d'environnement manquante ou invalide."""


def _check_config():

    if not DB_PASSWORD:

        raise ConfigurationError(
            "DB_PASSWORD n'est pas défini. Vérifiez votre fichier .env "
            "ou les variables d'environnement du service avant de "
            "démarrer l'application."
        )


# ==========================================================
# POOL DE CONNEXIONS
# ==========================================================

_pool = None


def _get_pool():
    """
    Retourne le pool de connexions, en le créant au premier appel.
    Un seul pool est partagé par tout le processus.
    """

    global _pool

    if _pool is None:

        _check_config()

        try:

            _pool = pool.SimpleConnectionPool(
                POOL_MIN_CONNECTIONS,
                POOL_MAX_CONNECTIONS,
                host=DB_HOST,
                port=DB_PORT,
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD
            )

        except psycopg2.OperationalError as e:

            raise DatabaseError(
                f"Impossible de se connecter à PostgreSQL "
                f"({DB_HOST}:{DB_PORT}/{DB_NAME}) : {e}"
            ) from e

    return _pool


def close_pool():
    """
    Ferme proprement toutes les connexions du pool.
    À appeler à l'arrêt de l'application (shutdown FastAPI par exemple).
    """

    global _pool

    if _pool is not None:

        _pool.closeall()

        _pool = None


# ==========================================================
# CONNEXION DIRECTE (hors pool)
# ==========================================================

def get_connection():
    """
    Retourne une connexion PostgreSQL directe, hors pool.

    Réservée aux opérations ponctuelles (init_database, scripts de
    migration). L'appelant est responsable de fermer la connexion.
    Pour toute requête applicative, utilisez get_cursor() ci-dessous.
    """

    _check_config()

    try:

        return psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )

    except psycopg2.OperationalError as e:

        raise DatabaseError(
            f"Impossible de se connecter à PostgreSQL "
            f"({DB_HOST}:{DB_PORT}/{DB_NAME}) : {e}"
        ) from e


# ==========================================================
# CURSEUR VIA LE POOL (usage recommandé)
# ==========================================================

@contextmanager
def get_cursor(commit=False, dict_cursor=False):
    """
    Fournit un curseur PostgreSQL depuis le pool de connexions.

    Garanties :
    - la connexion est TOUJOURS remise dans le pool (jamais de fuite),
      même si la requête échoue ;
    - le curseur est toujours fermé ;
    - commit uniquement si commit=True et qu'aucune exception n'a été
      levée ; rollback automatique sinon ;
    - toute erreur psycopg2 est ré-emballée en DatabaseError, avec le
      détail original conservé via `raise ... from e`.

    Usage :
        with get_cursor(commit=True) as cursor:
            cursor.execute("INSERT INTO ...", (...))
    """

    conn = _get_pool().getconn()

    cursor = None

    try:

        cursor_factory = RealDictCursor if dict_cursor else None

        cursor = conn.cursor(cursor_factory=cursor_factory)

        yield cursor

        if commit:

            conn.commit()

    except Exception as e:

        conn.rollback()

        logger.error("Erreur base de données : %s", e)

        raise DatabaseError(str(e)) from e

    finally:

        if cursor is not None:

            cursor.close()

        _get_pool().putconn(conn)


# ==========================================================
# INITIALISATION DE LA BASE
# ==========================================================

def init_database():

    connection = get_connection()

    try:

        cursor = connection.cursor()

        try:

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

        finally:

            cursor.close()

    except Exception as e:

        connection.rollback()

        raise DatabaseError(
            f"Échec de l'initialisation de la base : {e}"
        ) from e

    finally:

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

    with get_cursor(commit=True) as cursor:

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

    return review_id


# ==========================================================
# DERNIERS AVIS
# ==========================================================

def get_recent_reviews(limit=20):

    with get_cursor(dict_cursor=True) as cursor:

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

    return [
        dict(row)
        for row in rows
    ]


# ==========================================================
# STATISTIQUES
# ==========================================================

def get_statistics():
    """
    Calcule toutes les statistiques en une seule requête (au lieu de
    4 allers-retours séparés) grâce à COUNT(*) FILTER.
    """

    with get_cursor(dict_cursor=True) as cursor:

        cursor.execute(
            """
            SELECT

                COUNT(*) AS total,

                COUNT(*) FILTER (
                    WHERE sentiment = 'positif'
                ) AS positif,

                COUNT(*) FILTER (
                    WHERE sentiment = 'neutre'
                ) AS neutre,

                COUNT(*) FILTER (
                    WHERE sentiment = 'negatif'
                ) AS negatif,

                AVG(score) FILTER (
                    WHERE score IS NOT NULL
                ) AS moyenne

            FROM reviews
            """
        )

        row = cursor.fetchone()


    # ======================================================
    # VALEURS PAR DEFAUT
    # ======================================================

    total = row["total"]

    positif = row["positif"]

    neutre = row["neutre"]

    negatif = row["negatif"]

    moyenne = row["moyenne"]

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