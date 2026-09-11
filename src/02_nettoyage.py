import pandas as pd
import re
import os

# ==========================================================
# CONFIGURATION
# ==========================================================

FICHIER_ENTREE = "data/raw/reviews.csv"
FICHIER_SORTIE = "data/processed/reviews_clean.csv"

# Créer le dossier de sortie s'il n'existe pas
os.makedirs("data/processed", exist_ok=True)

# ==========================================================
# 1. CHARGEMENT
# ==========================================================

print("=" * 60)
print("CHARGEMENT DU DATASET")
print("=" * 60)

df = pd.read_csv(FICHIER_ENTREE)

print("Nombre de lignes :", len(df))
print("Nombre de colonnes :", len(df.columns))

# ==========================================================
# 2. SUPPRESSION DES LIGNES INUTILES
# ==========================================================

print("\n" + "=" * 60)
print("NETTOYAGE")
print("=" * 60)

# Supprimer les lignes sans texte d'avis
avant = len(df)

df = df.dropna(subset=["Text"])

print("Lignes sans texte supprimées :", avant - len(df))

# ==========================================================
# 3. SUPPRESSION DES DOUBLONS
# ==========================================================

avant = len(df)

df = df.drop_duplicates()

print("Doublons supprimés :", avant - len(df))

# ==========================================================
# 4. NETTOYAGE DU TEXTE
# ==========================================================

def nettoyer_texte(texte):

    texte = str(texte)

    # Convertir en minuscules
    texte = texte.lower()

    # Supprimer les balises HTML
    texte = re.sub(r"<.*?>", " ", texte)

    # Supprimer les URLs
    texte = re.sub(r"http\S+|www\S+", " ", texte)

    # Garder les lettres et les espaces
    texte = re.sub(r"[^a-zA-ZÀ-ÿ\s]", " ", texte)

    # Supprimer les espaces multiples
    texte = re.sub(r"\s+", " ", texte)

    # Supprimer les espaces au début et à la fin
    texte = texte.strip()

    return texte


print("\nNettoyage du texte en cours...")

df["TextClean"] = df["Text"].apply(nettoyer_texte)

# ==========================================================
# 5. SUPPRIMER LES AVIS VIDES APRÈS NETTOYAGE
# ==========================================================

avant = len(df)

df = df[df["TextClean"].str.len() > 0]

print("Avis vides après nettoyage :", avant - len(df))

# ==========================================================
# 6. TRAITEMENT DE LA NOTE
# ==========================================================

df["Score"] = pd.to_numeric(df["Score"], errors="coerce")

# Garder uniquement les notes valides
df = df[df["Score"].between(1, 5)]

# ==========================================================
# 7. CRÉATION DU SENTIMENT
# ==========================================================

def obtenir_sentiment(score):

    if score <= 2:
        return "negatif"

    elif score == 3:
        return "neutre"

    else:
        return "positif"


df["Sentiment"] = df["Score"].apply(obtenir_sentiment)

# ==========================================================
# 8. CONVERSION DE LA DATE
# ==========================================================

# Le champ Time est un timestamp Unix
df["Date"] = pd.to_datetime(
    df["Time"],
    unit="s",
    errors="coerce"
)

# ==========================================================
# 9. INFORMATIONS TEMPORELLES
# ==========================================================

df["Annee"] = df["Date"].dt.year
df["Mois"] = df["Date"].dt.month

# ==========================================================
# 10. SÉLECTION DES COLONNES
# ==========================================================

df_final = df[
    [
        "Id",
        "ProductId",
        "UserId",
        "ProfileName",
        "Score",
        "Time",
        "Date",
        "Annee",
        "Mois",
        "Summary",
        "Text",
        "TextClean",
        "Sentiment",
        "HelpfulnessNumerator",
        "HelpfulnessDenominator"
    ]
]

# ==========================================================
# 11. SAUVEGARDE
# ==========================================================

df_final.to_csv(
    FICHIER_SORTIE,
    index=False,
    encoding="utf-8-sig"
)

# ==========================================================
# 12. RÉSULTATS
# ==========================================================

print("\n" + "=" * 60)
print("NETTOYAGE TERMINÉ")
print("=" * 60)

print("Nombre de lignes finales :", len(df_final))
print("Nombre de colonnes finales :", len(df_final.columns))

print("\nRépartition des sentiments :")

print(
    df_final["Sentiment"]
    .value_counts()
)

print("\nMoyenne des notes :")

print(
    df_final["Score"].mean()
)

print("\nFichier créé :")

print(FICHIER_SORTIE)

print("\nPremières lignes :")

print(
    df_final[
        [
            "ProductId",
            "Score",
            "Date",
            "Sentiment",
            "TextClean"
        ]
    ].head()
)