import pandas as pd
import matplotlib.pyplot as plt
import os

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/processed/reviews_clean.csv"
OUTPUT_DIR = "results/figures"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("ÉTAPE 08 - ANALYSE TEMPORELLE")
print("=" * 60)

# ============================================================
# 1. CHARGEMENT
# ============================================================

print("\n1. Chargement des données...")

df = pd.read_csv(
    INPUT_FILE,
    usecols=[
        "Score",
        "Date",
        "Annee",
        "Mois",
        "Sentiment"
    ]
)

print(f"Nombre d'avis : {len(df)}")

# Conversion de la date
df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

df["Annee"] = pd.to_numeric(
    df["Annee"],
    errors="coerce"
)

df["Mois"] = pd.to_numeric(
    df["Mois"],
    errors="coerce"
)

df = df.dropna(
    subset=["Date", "Annee"]
)

# ============================================================
# 2. NOMBRE D'AVIS PAR ANNÉE
# ============================================================

print("\n" + "=" * 60)
print("NOMBRE D'AVIS PAR ANNÉE")
print("=" * 60)

avis_annee = (
    df.groupby("Annee")
    .size()
    .reset_index(name="NombreAvis")
)

print(avis_annee.to_string(index=False))

# Graphique

plt.figure(figsize=(12, 6))

plt.plot(
    avis_annee["Annee"],
    avis_annee["NombreAvis"],
    marker="o"
)

plt.xlabel("Année")
plt.ylabel("Nombre d'avis")
plt.title("Évolution du nombre d'avis par année")
plt.grid(True)
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/11_evolution_nombre_avis.png",
    dpi=150
)

plt.close()

# ============================================================
# 3. NOTE MOYENNE PAR ANNÉE
# ============================================================

print("\n" + "=" * 60)
print("NOTE MOYENNE PAR ANNÉE")
print("=" * 60)

note_annee = (
    df.groupby("Annee")["Score"]
    .agg(["mean", "count"])
    .reset_index()
)

note_annee.columns = [
    "Annee",
    "NoteMoyenne",
    "NombreAvis"
]

print(
    note_annee.to_string(index=False)
)

# Graphique

plt.figure(figsize=(12, 6))

plt.plot(
    note_annee["Annee"],
    note_annee["NoteMoyenne"],
    marker="o"
)

plt.axhline(
    y=4,
    linestyle="--"
)

plt.xlabel("Année")
plt.ylabel("Note moyenne")
plt.title("Évolution de la note moyenne")
plt.ylim(1, 5)
plt.grid(True)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/12_evolution_note_moyenne.png",
    dpi=150
)

plt.close()

# ============================================================
# 4. SENTIMENT PAR ANNÉE
# ============================================================

print("\n" + "=" * 60)
print("ÉVOLUTION DES SENTIMENTS")
print("=" * 60)

sentiment_annee = pd.crosstab(
    df["Annee"],
    df["Sentiment"],
    normalize="index"
) * 100

sentiment_annee = sentiment_annee.reset_index()

print(
    sentiment_annee.to_string(index=False)
)

# ============================================================
# 5. GRAPHIQUE DES SENTIMENTS
# ============================================================

plt.figure(figsize=(12, 6))

for sentiment in [
    "positif",
    "neutre",
    "negatif"
]:

    if sentiment in sentiment_annee.columns:

        plt.plot(
            sentiment_annee["Annee"],
            sentiment_annee[sentiment],
            marker="o",
            label=sentiment
        )

plt.xlabel("Année")
plt.ylabel("Pourcentage (%)")
plt.title("Évolution des sentiments par année")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/13_evolution_sentiments.png",
    dpi=150
)

plt.close()

# ============================================================
# 6. SATISFACTION PAR MOIS
# ============================================================

print("\n" + "=" * 60)
print("SATISFACTION PAR MOIS")
print("=" * 60)

note_mois = (
    df.groupby("Mois")["Score"]
    .mean()
    .reset_index()
)

note_mois.columns = [
    "Mois",
    "NoteMoyenne"
]

print(
    note_mois.to_string(index=False)
)

# Graphique

plt.figure(figsize=(12, 6))

plt.plot(
    note_mois["Mois"],
    note_mois["NoteMoyenne"],
    marker="o"
)

plt.xlabel("Mois")
plt.ylabel("Note moyenne")
plt.title("Note moyenne par mois")
plt.xticks(range(1, 13))
plt.ylim(1, 5)
plt.grid(True)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/14_note_par_mois.png",
    dpi=150
)

plt.close()

# ============================================================
# 7. TENDANCE GLOBALE
# ============================================================

print("\n" + "=" * 60)
print("ANALYSE DE LA TENDANCE")
print("=" * 60)

x = note_annee["Annee"].values
y = note_annee["NoteMoyenne"].values

if len(x) >= 2:

    coefficient = pd.Series(y).corr(
        pd.Series(x)
    )

    print(
        f"\nCorrélation année / note moyenne : "
        f"{coefficient:.4f}"
    )

    if coefficient > 0.3:

        print(
            "Tendance : la satisfaction semble augmenter."
        )

    elif coefficient < -0.3:

        print(
            "Tendance : la satisfaction semble diminuer."
        )

    else:

        print(
            "Tendance : satisfaction relativement stable."
        )

# ============================================================
# 8. MEILLEURE ET PLUS MAUVAISE ANNÉE
# ============================================================

print("\n" + "=" * 60)
print("ANNÉES EXTRÊMES")
print("=" * 60)

meilleure = note_annee.loc[
    note_annee["NoteMoyenne"].idxmax()
]

pire = note_annee.loc[
    note_annee["NoteMoyenne"].idxmin()
]

print(
    f"\nMeilleure année : "
    f"{int(meilleure['Annee'])}"
)

print(
    f"Note moyenne : "
    f"{meilleure['NoteMoyenne']:.2f}/5"
)

print(
    f"\nAnnée avec la plus faible satisfaction : "
    f"{int(pire['Annee'])}"
)

print(
    f"Note moyenne : "
    f"{pire['NoteMoyenne']:.2f}/5"
)

# ============================================================
# 9. EXPORT DES RÉSULTATS
# ============================================================

print("\n" + "=" * 60)
print("SAUVEGARDE")
print("=" * 60)

avis_annee.to_csv(
    "results/avis_par_annee.csv",
    index=False
)

note_annee.to_csv(
    "results/note_par_annee.csv",
    index=False
)

sentiment_annee.to_csv(
    "results/sentiments_par_annee.csv",
    index=False
)

note_mois.to_csv(
    "results/note_par_mois.csv",
    index=False
)

print("Résultats sauvegardés dans results/")

# ============================================================
# FIN
# ============================================================

print("\n" + "=" * 60)
print("ÉTAPE 08 TERMINÉE")
print("=" * 60)