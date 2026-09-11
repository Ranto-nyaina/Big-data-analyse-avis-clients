import os
import shutil
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, "predictions")
PYSPARK_DIR = os.path.join(RESULTS_DIR, "pyspark")

DASHBOARD_DIR = os.path.join(BASE_DIR, "dashboard")
DATA_DIR = os.path.join(DASHBOARD_DIR, "data")


# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def creer_dossier(path):
    """Créer un dossier s'il n'existe pas."""
    os.makedirs(path, exist_ok=True)


def afficher_titre(titre):
    print()
    print("=" * 70)
    print(titre)
    print("=" * 70)


def sauvegarder(df, nom):
    """Sauvegarder un DataFrame dans dashboard/data."""
    chemin = os.path.join(DATA_DIR, nom)

    df.to_csv(
        chemin,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"  OK : {nom} ({len(df):,} lignes)")

    return chemin


def chercher_csv(dossier, prefixe):
    """
    Cherche un fichier CSV dans un dossier Spark.
    Spark crée généralement un dossier contenant part-xxxxx.csv.
    """
    if not os.path.exists(dossier):
        return None

    fichiers = []

    for fichier in os.listdir(dossier):
        if fichier.endswith(".csv"):
            fichiers.append(os.path.join(dossier, fichier))

    if not fichiers:
        return None

    fichiers.sort()

    return fichiers[0]


def lire_csv_spark(dossier, colonnes=None):
    """Lire un résultat CSV produit par Spark."""
    fichier = chercher_csv(dossier, "")

    if fichier is None:
        return None

    try:
        df = pd.read_csv(fichier)

        if colonnes:
            colonnes_existantes = [
                c for c in colonnes
                if c in df.columns
            ]

            if colonnes_existantes:
                df = df[colonnes_existantes]

        return df

    except Exception as e:
        print(f"  ERREUR lecture {fichier}: {e}")
        return None


# ============================================================
# DÉMARRAGE
# ============================================================

afficher_titre(
    "ÉTAPE 11 - PRÉPARATION DU DASHBOARD POWER BI"
)

print(f"Projet : {BASE_DIR}")
print(f"Destination : {DATA_DIR}")


# ============================================================
# CRÉATION DES DOSSIERS
# ============================================================

creer_dossier(DASHBOARD_DIR)
creer_dossier(DATA_DIR)

print("\nDossiers créés/vérifiés.")


# ============================================================
# 1. AVIS PAR ANNÉE
# ============================================================

afficher_titre("[1/8] Avis par année")

fichier = os.path.join(
    RESULTS_DIR,
    "avis_par_annee.csv"
)

if os.path.exists(fichier):

    df = pd.read_csv(fichier)

    print(df.head())

    sauvegarder(
        df,
        "avis_par_annee.csv"
    )

else:

    # Si le fichier n'existe pas, essayer de le reconstruire
    print("Fichier avis_par_annee.csv introuvable.")

    fichier_clean = os.path.join(
        BASE_DIR,
        "data",
        "processed",
        "reviews_clean.csv"
    )

    if os.path.exists(fichier_clean):

        df_reviews = pd.read_csv(
            fichier_clean,
            usecols=["Annee"]
        )

        df = (
            df_reviews
            .groupby("Annee")
            .size()
            .reset_index(name="Nombre_Avis")
        )

        sauvegarder(
            df,
            "avis_par_annee.csv"
        )

    else:
        print("Impossible de reconstruire les données.")


# ============================================================
# 2. NOTE MOYENNE PAR ANNÉE
# ============================================================

afficher_titre("[2/8] Note moyenne par année")

fichier = os.path.join(
    RESULTS_DIR,
    "note_par_annee.csv"
)

if os.path.exists(fichier):

    df = pd.read_csv(fichier)

    print(df.head())

    sauvegarder(
        df,
        "notes_par_annee.csv"
    )

else:

    print("Fichier note_par_annee.csv introuvable.")

    fichier_clean = os.path.join(
        BASE_DIR,
        "data",
        "processed",
        "reviews_clean.csv"
    )

    if os.path.exists(fichier_clean):

        df_reviews = pd.read_csv(
            fichier_clean,
            usecols=["Annee", "Score"]
        )

        df = (
            df_reviews
            .groupby("Annee")
            .agg(
                Nombre_Avis=("Score", "count"),
                Note_Moyenne=("Score", "mean")
            )
            .reset_index()
        )

        sauvegarder(
            df,
            "notes_par_annee.csv"
        )

    else:
        print("Impossible de reconstruire les données.")


# ============================================================
# 3. SENTIMENTS PAR ANNÉE
# ============================================================

afficher_titre("[3/8] Sentiments par année")

fichier = os.path.join(
    RESULTS_DIR,
    "sentiments_par_annee.csv"
)

if os.path.exists(fichier):

    df = pd.read_csv(fichier)

    print(df.head())

    sauvegarder(
        df,
        "sentiments_par_annee.csv"
    )

else:

    print("Fichier sentiments_par_annee.csv introuvable.")

    fichier_clean = os.path.join(
        BASE_DIR,
        "data",
        "processed",
        "reviews_clean.csv"
    )

    if os.path.exists(fichier_clean):

        df_reviews = pd.read_csv(
            fichier_clean,
            usecols=["Annee", "Sentiment"]
        )

        df = (
            df_reviews
            .groupby(["Annee", "Sentiment"])
            .size()
            .reset_index(name="Nombre")
        )

        sauvegarder(
            df,
            "sentiments_par_annee.csv"
        )

    else:
        print("Impossible de reconstruire les données.")


# ============================================================
# 4. RÉPARTITION DES NOTES
# ============================================================

afficher_titre("[4/8] Répartition des notes")

fichier = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "reviews_clean.csv"
)

if os.path.exists(fichier):

    df = pd.read_csv(
        fichier,
        usecols=["Score"]
    )

    repartition = (
        df
        .groupby("Score")
        .size()
        .reset_index(name="Nombre")
    )

    repartition["Pourcentage"] = (
        repartition["Nombre"]
        / repartition["Nombre"].sum()
        * 100
    )

    repartition["Pourcentage"] = (
        repartition["Pourcentage"]
        .round(2)
    )

    sauvegarder(
        repartition,
        "repartition_notes.csv"
    )

    print(repartition)

else:

    print("Dataset nettoyé introuvable.")


# ============================================================
# 5. PRODUITS LES PLUS COMMENTÉS
# ============================================================

afficher_titre("[5/8] Produits les plus commentés")

pyspark_produits = os.path.join(
    PYSPARK_DIR,
    "top_produits_pyspark"
)

df_produits = lire_csv_spark(
    pyspark_produits
)

if df_produits is not None:

    print(df_produits.head())

    # Trier par nombre d'avis
    if "Nombre_Avis" in df_produits.columns:

        df_produits = (
            df_produits
            .sort_values(
                "Nombre_Avis",
                ascending=False
            )
            .head(20)
        )

    sauvegarder(
        df_produits,
        "produits_top.csv"
    )

else:

    # Reconstruction avec Pandas
    fichier_clean = os.path.join(
        BASE_DIR,
        "data",
        "processed",
        "reviews_clean.csv"
    )

    if os.path.exists(fichier_clean):

        df_reviews = pd.read_csv(
            fichier_clean,
            usecols=[
                "ProductId",
                "Score"
            ]
        )

        produits = (
            df_reviews
            .groupby("ProductId")
            .agg(
                Nombre_Avis=("Score", "count"),
                Note_Moyenne=("Score", "mean")
            )
            .reset_index()
            .sort_values(
                "Nombre_Avis",
                ascending=False
            )
            .head(20)
        )

        produits["Note_Moyenne"] = (
            produits["Note_Moyenne"]
            .round(2)
        )

        sauvegarder(
            produits,
            "produits_top.csv"
        )


# ============================================================
# 6. AVIS LES PLUS UTILES
# ============================================================

afficher_titre("[6/8] Avis les plus utiles")

pyspark_utiles = os.path.join(
    PYSPARK_DIR,
    "top_avis_utiles_pyspark"
)

df_utiles = lire_csv_spark(
    pyspark_utiles
)

if df_utiles is not None:

    print(df_utiles.head())

    sauvegarder(
        df_utiles,
        "avis_utiles.csv"
    )

else:

    print(
        "Résultat PySpark introuvable. "
        "Reconstruction..."
    )

    fichier_clean = os.path.join(
        BASE_DIR,
        "data",
        "processed",
        "reviews_clean.csv"
    )

    if os.path.exists(fichier_clean):

        df_reviews = pd.read_csv(
            fichier_clean,
            usecols=[
                "ProductId",
                "Score",
                "Sentiment",
                "Summary",
                "HelpfulnessNumerator",
                "HelpfulnessDenominator"
            ]
        )

        df_reviews["TauxUtilite"] = np.where(
            df_reviews["HelpfulnessDenominator"] > 0,
            df_reviews["HelpfulnessNumerator"]
            / df_reviews["HelpfulnessDenominator"],
            0
        )

        df_utiles = (
            df_reviews
            .sort_values(
                "TauxUtilite",
                ascending=False
            )
            .head(20)
        )

        sauvegarder(
            df_utiles,
            "avis_utiles.csv"
        )


# ============================================================
# 7. PRÉVISION DE SATISFACTION
# ============================================================

afficher_titre("[7/8] Prévision de satisfaction")

fichier_prevision = os.path.join(
    PREDICTIONS_DIR,
    "prevision_satisfaction_12_mois.csv"
)

if os.path.exists(fichier_prevision):

    df_prevision = pd.read_csv(
        fichier_prevision
    )

    print(df_prevision)

    sauvegarder(
        df_prevision,
        "satisfaction_prevision.csv"
    )

else:

    print(
        "Fichier de prévision introuvable :"
    )

    print(fichier_prevision)


# ============================================================
# 8. THÈMES
# ============================================================

afficher_titre("[8/8] Thèmes des avis")

# Le script 04 produit normalement les résultats
# des thèmes sous forme d'analyse.
#
# On reconstruit ici les thèmes directement à partir
# des mots-clés afin d'avoir un fichier exploitable
# dans Power BI.

fichier_clean = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "reviews_clean.csv"
)

if os.path.exists(fichier_clean):

    print("Analyse des thèmes...")

    df_reviews = pd.read_csv(
        fichier_clean,
        usecols=["TextClean"]
    )

    df_reviews["TextClean"] = (
        df_reviews["TextClean"]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    themes = {

        "Qualité": [
            "quality",
            "product",
            "good",
            "bad",
            "best",
            "excellent",
            "poor"
        ],

        "Goût": [
            "taste",
            "flavor",
            "flavour",
            "delicious",
            "sweet",
            "bitter",
            "salty"
        ],

        "Prix": [
            "price",
            "expensive",
            "cheap",
            "cost",
            "value"
        ],

        "Quantité": [
            "amount",
            "quantity",
            "size",
            "small",
            "large",
            "portion"
        ],

        "Emballage": [
            "package",
            "packaging",
            "box",
            "bag",
            "package"
        ],

        "Livraison": [
            "shipping",
            "delivery",
            "arrived",
            "deliver"
        ],

        "Service": [
            "service",
            "seller",
            "customer",
            "support"
        ]
    }

    resultats_themes = []

    total_avis = len(df_reviews)

    for theme, mots in themes.items():

        masque = None

        for mot in mots:

            condition = (
                df_reviews["TextClean"]
                .str.contains(
                    mot,
                    regex=False,
                    na=False
                )
            )

            if masque is None:
                masque = condition
            else:
                masque = masque | condition

        nombre = int(masque.sum())

        pourcentage = (
            nombre / total_avis * 100
            if total_avis > 0
            else 0
        )

        resultats_themes.append(
            {
                "Theme": theme,
                "Nombre_Avis": nombre,
                "Pourcentage": round(
                    pourcentage,
                    2
                )
            }
        )

    df_themes = pd.DataFrame(
        resultats_themes
    )

    df_themes = (
        df_themes
        .sort_values(
            "Nombre_Avis",
            ascending=False
        )
        .reset_index(drop=True)
    )

    sauvegarder(
        df_themes,
        "themes.csv"
    )

    print(df_themes)

else:

    print(
        "Dataset nettoyé introuvable."
    )


# ============================================================
# TABLEAU DE SYNTHÈSE
# ============================================================

afficher_titre(
    "CRÉATION DU TABLEAU DE SYNTHÈSE"
)

fichier_clean = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "reviews_clean.csv"
)

if os.path.exists(fichier_clean):

    df_global = pd.read_csv(
        fichier_clean,
        usecols=[
            "Score",
            "Sentiment"
        ]
    )

    total_avis = len(df_global)

    note_moyenne = (
        df_global["Score"]
        .mean()
    )

    positif = (
        df_global["Sentiment"]
        .eq("positif")
        .sum()
    )

    negatif = (
        df_global["Sentiment"]
        .eq("negatif")
        .sum()
    )

    neutre = (
        df_global["Sentiment"]
        .eq("neutre")
        .sum()
    )

    synthese = pd.DataFrame(
        [
            {
                "Indicateur": "Nombre total d'avis",
                "Valeur": total_avis
            },
            {
                "Indicateur": "Note moyenne",
                "Valeur": round(
                    note_moyenne,
                    2
                )
            },
            {
                "Indicateur": "Avis positifs",
                "Valeur": positif
            },
            {
                "Indicateur": "Avis négatifs",
                "Valeur": negatif
            },
            {
                "Indicateur": "Avis neutres",
                "Valeur": neutre
            },
            {
                "Indicateur": "% avis positifs",
                "Valeur": round(
                    positif / total_avis * 100,
                    2
                )
            },
            {
                "Indicateur": "% avis négatifs",
                "Valeur": round(
                    negatif / total_avis * 100,
                    2
                )
            },
            {
                "Indicateur": "% avis neutres",
                "Valeur": round(
                    neutre / total_avis * 100,
                    2
                )
            }
        ]
    )

    sauvegarder(
        synthese,
        "synthese_dashboard.csv"
    )

    print()
    print(synthese)


# ============================================================
# COPIE DES GRAPHIQUES IMPORTANTS
# ============================================================

afficher_titre(
    "COPIE DES GRAPHIQUES"
)

figures_dashboard = os.path.join(
    DASHBOARD_DIR,
    "figures"
)

creer_dossier(figures_dashboard)

figures_a_copier = [
    "06_top_mots.png",
    "07_wordcloud_global.png",
    "08_wordcloud_positif.png",
    "09_wordcloud_negatif.png",
    "10_themes.png",
    "11_evolution_nombre_avis.png",
    "12_evolution_note_moyenne.png",
    "13_evolution_sentiments.png",
    "14_note_par_mois.png",
    "15_prevision_satisfaction.png",
    "16_evaluation_modele_temporel.png"
]

for nom in figures_a_copier:

    source = os.path.join(
        FIGURES_DIR,
        nom
    )

    destination = os.path.join(
        figures_dashboard,
        nom
    )

    if os.path.exists(source):

        shutil.copy2(
            source,
            destination
        )

        print(f"  OK : {nom}")

    else:

        print(f"  ABSENT : {nom}")


# ============================================================
# README DASHBOARD
# ============================================================

readme = os.path.join(
    DASHBOARD_DIR,
    "README_DASHBOARD.txt"
)

contenu_readme = """
============================================================
DASHBOARD - ANALYSE DES AVIS CLIENTS
============================================================

Projet :
Analyse Big Data des avis clients Amazon Fine Food Reviews.

Nombre d'avis :
568453

Note moyenne :
4.18 / 5

Sentiment :
- Positif : 443776
- Négatif : 82037
- Neutre   : 42640

============================================================
FICHIERS DISPONIBLES
============================================================

synthese_dashboard.csv
    Indicateurs principaux du projet.

avis_par_annee.csv
    Nombre d'avis par année.

notes_par_annee.csv
    Note moyenne par année.

sentiments_par_annee.csv
    Évolution des sentiments par année.

repartition_notes.csv
    Répartition des notes 1 à 5.

produits_top.csv
    Top 20 des produits les plus commentés.

avis_utiles.csv
    Avis ayant le meilleur taux d'utilité.

satisfaction_prevision.csv
    Prévision de satisfaction sur 12 mois.

themes.csv
    Principaux thèmes présents dans les avis.

============================================================
POWER BI
============================================================

Importer les fichiers CSV depuis :

dashboard/data/

Dashboard recommandé :

PAGE 1 - VUE GÉNÉRALE
    - Nombre total d'avis
    - Note moyenne
    - % positif
    - % négatif
    - % neutre
    - Répartition des notes

PAGE 2 - ANALYSE TEMPORELLE
    - Nombre d'avis par année
    - Note moyenne par année
    - Évolution des sentiments

PAGE 3 - ANALYSE PRODUITS
    - Top 20 produits
    - Nombre d'avis
    - Note moyenne

PAGE 4 - ANALYSE DES THÈMES
    - Goût
    - Qualité
    - Prix
    - Quantité
    - Emballage
    - Livraison
    - Service

PAGE 5 - PRÉVISION
    - Satisfaction historique
    - Satisfaction prévue
    - Évolution sur 12 mois

============================================================
"""


with open(
    readme,
    "w",
    encoding="utf-8"
) as f:

    f.write(contenu_readme)

print()
print(f"README créé : {readme}")


# ============================================================
# FIN
# ============================================================

afficher_titre(
    "ÉTAPE 11 TERMINÉE AVEC SUCCÈS"
)

print(
    f"""
Les données du dashboard sont disponibles dans :

{DATA_DIR}

Fichiers préparés pour Power BI :
"""

)

if os.path.exists(DATA_DIR):

    fichiers = sorted(
        os.listdir(DATA_DIR)
    )

    for fichier in fichiers:
        print(f"  - {fichier}")

print()
print("=" * 70)
print("PROCHAINE ÉTAPE : CRÉATION DU DASHBOARD POWER BI")
print("=" * 70)