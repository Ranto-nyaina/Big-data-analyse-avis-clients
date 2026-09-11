from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    avg,
    count,
    sum,
    when,
    desc
)
from pyspark.sql.types import (
    IntegerType,
    DoubleType
)
import os


# ==========================================================
# CONFIGURATION
# ==========================================================

INPUT_FILE = "data/processed/reviews_clean.csv"

OUTPUT_DIR = "results/pyspark"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================================
# 1. DÉMARRAGE DE SPARK
# ==========================================================

print("=" * 70)
print("ÉTAPE 10 - ANALYSE BIG DATA AVEC PYSPARK")
print("=" * 70)

print("\n[1/7] Démarrage de Spark...")

spark = (
    SparkSession.builder
    .appName("AnalyseReviewsBigData")
    .master("local[*]")
    .config("spark.sql.ansi.enabled", "false")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

print("Spark démarré avec succès.")


# ==========================================================
# 2. CHARGEMENT DU DATASET
# ==========================================================

print("\n[2/7] Chargement du dataset...")

df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "false")
    .option("quote", '"')
    .option("escape", '"')
    .option("multiLine", "true")
    .option("mode", "PERMISSIVE")
    .csv(INPUT_FILE)
)

print("Nombre de lignes :", df.count())

print("\nStructure initiale du dataset :")
df.printSchema()


# ==========================================================
# 3. CONVERSION DES TYPES
# ==========================================================

print("\nConversion des colonnes numériques...")

df = (
    df
    .withColumn("Score", col("Score").cast(IntegerType()))
    .withColumn("Time", col("Time").cast(IntegerType()))
    .withColumn("Annee", col("Annee").cast(IntegerType()))
    .withColumn("Mois", col("Mois").cast(IntegerType()))
    .withColumn(
        "HelpfulnessNumerator",
        col("HelpfulnessNumerator").cast(IntegerType())
    )
    .withColumn(
        "HelpfulnessDenominator",
        col("HelpfulnessDenominator").cast(IntegerType())
    )
)

print("\nStructure après conversion :")
df.printSchema()


# ==========================================================
# 4. ANALYSE GLOBALE
# ==========================================================

print("\n[3/7] Analyse globale...")


# ----------------------------------------------------------
# Nombre d'avis par sentiment
# ----------------------------------------------------------

print("\nNombre d'avis par sentiment :")

sentiments = (
    df
    .filter(col("Sentiment").isin(
        "positif",
        "negatif",
        "neutre"
    ))
    .groupBy("Sentiment")
    .agg(count("*").alias("Nombre"))
    .orderBy(desc("Nombre"))
)

sentiments.show()


# ----------------------------------------------------------
# Note moyenne
# ----------------------------------------------------------

print("\nNote moyenne globale :")

note_moyenne = (
    df
    .filter(col("Score").isNotNull())
    .agg(
        avg("Score").alias("Note_Moyenne")
    )
)

note_moyenne.show()


# ----------------------------------------------------------
# Répartition des notes
# ----------------------------------------------------------

print("\nRépartition des notes :")

notes = (
    df
    .filter(col("Score").isNotNull())
    .groupBy("Score")
    .agg(count("*").alias("Nombre"))
    .orderBy("Score")
)

notes.show()


# ==========================================================
# 5. ANALYSE TEMPORELLE
# ==========================================================

print("\n[4/7] Analyse temporelle...")


# ----------------------------------------------------------
# Analyse par année
# ----------------------------------------------------------

analyse_annee = (
    df
    .filter(col("Annee").isNotNull())
    .groupBy("Annee")
    .agg(
        count("*").alias("Nombre_Avis"),
        avg("Score").alias("Note_Moyenne")
    )
    .orderBy("Annee")
)

print("\nAnalyse par année :")

analyse_annee.show(30)


# ----------------------------------------------------------
# Sentiment par année
# ----------------------------------------------------------

sentiments_annee = (
    df
    .filter(
        col("Annee").isNotNull()
        & col("Sentiment").isin(
            "positif",
            "negatif",
            "neutre"
        )
    )
    .groupBy("Annee", "Sentiment")
    .agg(
        count("*").alias("Nombre")
    )
    .orderBy("Annee", "Sentiment")
)

print("\nSentiments par année :")

sentiments_annee.show(50)


# ==========================================================
# 6. ANALYSE DE L'UTILITÉ DES AVIS
# ==========================================================

print("\n[5/7] Analyse de l'utilité des avis...")


df = df.withColumn(
    "TauxUtilite",
    when(
        col("HelpfulnessDenominator") > 0,
        col("HelpfulnessNumerator") /
        col("HelpfulnessDenominator")
    ).otherwise(0.0)
)


# ----------------------------------------------------------
# Top avis les plus utiles
# ----------------------------------------------------------

top_avis_utiles = (
    df
    .filter(
        col("HelpfulnessDenominator").isNotNull()
        & (col("HelpfulnessDenominator") > 0)
    )
    .select(
        "ProductId",
        "Score",
        "Sentiment",
        "Summary",
        "TauxUtilite",
        "HelpfulnessNumerator",
        "HelpfulnessDenominator"
    )
    .orderBy(
        desc("TauxUtilite"),
        desc("HelpfulnessDenominator")
    )
    .limit(100)
)

print("\nTop avis les plus utiles :")

top_avis_utiles.show(20, truncate=50)


# ==========================================================
# 7. PRODUITS LES PLUS COMMENTÉS
# ==========================================================

print("\n[6/7] Produits les plus commentés...")

top_produits = (
    df
    .filter(col("ProductId").isNotNull())
    .groupBy("ProductId")
    .agg(
        count("*").alias("Nombre_Avis"),
        avg("Score").alias("Note_Moyenne")
    )
    .orderBy(desc("Nombre_Avis"))
    .limit(20)
)

print("\nTop 20 produits par nombre d'avis :")

top_produits.show(20, truncate=False)


# ==========================================================
# SAUVEGARDE DES RÉSULTATS
# ==========================================================

print("\n[7/7] Sauvegarde des résultats...")


# ----------------------------------------------------------
# Sentiments
# ----------------------------------------------------------

sentiments.coalesce(1).write.mode("overwrite").option(
    "header", "true"
).csv(
    f"{OUTPUT_DIR}/sentiments_pyspark"
)


# ----------------------------------------------------------
# Notes
# ----------------------------------------------------------

notes.coalesce(1).write.mode("overwrite").option(
    "header", "true"
).csv(
    f"{OUTPUT_DIR}/notes_pyspark"
)


# ----------------------------------------------------------
# Analyse par année
# ----------------------------------------------------------

analyse_annee.coalesce(1).write.mode("overwrite").option(
    "header", "true"
).csv(
    f"{OUTPUT_DIR}/analyse_par_annee_pyspark"
)


# ----------------------------------------------------------
# Sentiments par année
# ----------------------------------------------------------

sentiments_annee.coalesce(1).write.mode("overwrite").option(
    "header", "true"
).csv(
    f"{OUTPUT_DIR}/sentiments_par_annee_pyspark"
)


# ----------------------------------------------------------
# Avis utiles
# ----------------------------------------------------------

top_avis_utiles.coalesce(1).write.mode("overwrite").option(
    "header", "true"
).csv(
    f"{OUTPUT_DIR}/top_avis_utiles_pyspark"
)


# ----------------------------------------------------------
# Produits
# ----------------------------------------------------------

top_produits.coalesce(1).write.mode("overwrite").option(
    "header", "true"
).csv(
    f"{OUTPUT_DIR}/top_produits_pyspark"
)


# ==========================================================
# FIN
# ==========================================================

print("\n" + "=" * 70)
print("ANALYSE PYSPARK TERMINÉE AVEC SUCCÈS")
print("=" * 70)

print("\nRésultats enregistrés dans :")
print(f"{OUTPUT_DIR}/")

spark.stop()