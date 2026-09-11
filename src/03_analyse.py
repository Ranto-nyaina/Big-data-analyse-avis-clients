import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================================
# CONFIGURATION
# ==========================================================

FICHIER = "data/processed/reviews_clean.csv"

DOSSIER_RESULTATS = "results/figures"

os.makedirs(DOSSIER_RESULTATS, exist_ok=True)

# ==========================================================
# 1. CHARGEMENT
# ==========================================================

print("=" * 60)
print("ANALYSE EXPLORATOIRE DES AVIS CLIENTS")
print("=" * 60)

df = pd.read_csv(FICHIER)

print("\nNombre d'avis :", len(df))
print("Nombre de colonnes :", len(df.columns))

# ==========================================================
# 2. NOMBRE DE PRODUITS
# ==========================================================

nombre_produits = df["ProductId"].nunique()

print("\nNombre de produits :", nombre_produits)

# ==========================================================
# 3. NOMBRE DE CLIENTS
# ==========================================================

nombre_clients = df["UserId"].nunique()

print("Nombre de clients :", nombre_clients)

# ==========================================================
# 4. NOTE MOYENNE
# ==========================================================

note_moyenne = df["Score"].mean()

print("Note moyenne :", round(note_moyenne, 2), "/ 5")

# ==========================================================
# 5. RÉPARTITION DES NOTES
# ==========================================================

print("\n" + "=" * 60)
print("RÉPARTITION DES NOTES")
print("=" * 60)

repartition_notes = df["Score"].value_counts().sort_index()

print(repartition_notes)

pourcentage_notes = (
    df["Score"]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)

print("\nPourcentage :")

print(pourcentage_notes.round(2))

# ==========================================================
# GRAPHIQUE DES NOTES
# ==========================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Score"
)

plt.title("Répartition des notes clients")
plt.xlabel("Note")
plt.ylabel("Nombre d'avis")

plt.tight_layout()

plt.savefig(
    f"{DOSSIER_RESULTATS}/01_repartition_notes.png"
)

plt.close()

# ==========================================================
# 6. RÉPARTITION DES SENTIMENTS
# ==========================================================

print("\n" + "=" * 60)
print("RÉPARTITION DES SENTIMENTS")
print("=" * 60)

repartition_sentiments = df["Sentiment"].value_counts()

print(repartition_sentiments)

pourcentage_sentiments = (
    df["Sentiment"]
    .value_counts(normalize=True)
    * 100
)

print("\nPourcentage :")

print(pourcentage_sentiments.round(2))

# ==========================================================
# GRAPHIQUE DES SENTIMENTS
# ==========================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Sentiment",
    order=["negatif", "neutre", "positif"]
)

plt.title("Répartition des sentiments")
plt.xlabel("Sentiment")
plt.ylabel("Nombre d'avis")

plt.tight_layout()

plt.savefig(
    f"{DOSSIER_RESULTATS}/02_repartition_sentiments.png"
)

plt.close()

# ==========================================================
# 7. TOP 10 PRODUITS PAR NOMBRE D'AVIS
# ==========================================================

print("\n" + "=" * 60)
print("TOP 10 PRODUITS PAR NOMBRE D'AVIS")
print("=" * 60)

top_produits = (
    df["ProductId"]
    .value_counts()
    .head(10)
)

print(top_produits)

# ==========================================================
# GRAPHIQUE TOP PRODUITS
# ==========================================================

plt.figure(figsize=(10, 6))

top_produits.sort_values().plot(
    kind="barh"
)

plt.title("Top 10 produits par nombre d'avis")
plt.xlabel("Nombre d'avis")
plt.ylabel("Produit")

plt.tight_layout()

plt.savefig(
    f"{DOSSIER_RESULTATS}/03_top_produits.png"
)

plt.close()

# ==========================================================
# 8. NOTE MOYENNE PAR PRODUIT
# ==========================================================

note_par_produit = (
    df.groupby("ProductId")["Score"]
    .agg(["count", "mean"])
)

# Garder uniquement les produits ayant au moins 50 avis
produits_fiables = note_par_produit[
    note_par_produit["count"] >= 50
]

# ==========================================================
# TOP 10 MEILLEURS PRODUITS
# ==========================================================

meilleurs_produits = (
    produits_fiables
    .sort_values("mean", ascending=False)
    .head(10)
)

print("\n" + "=" * 60)
print("TOP 10 PRODUITS LES MIEUX NOTÉS")
print("=" * 60)

print(meilleurs_produits)

# ==========================================================
# TOP 10 PRODUITS LES MOINS BIEN NOTÉS
# ==========================================================

pires_produits = (
    produits_fiables
    .sort_values("mean", ascending=True)
    .head(10)
)

print("\n" + "=" * 60)
print("TOP 10 PRODUITS LES MOINS BIEN NOTÉS")
print("=" * 60)

print(pires_produits)

# ==========================================================
# 9. ÉVOLUTION DES AVIS PAR ANNÉE
# ==========================================================

avis_par_annee = (
    df.groupby("Annee")
    .size()
)

print("\n" + "=" * 60)
print("NOMBRE D'AVIS PAR ANNÉE")
print("=" * 60)

print(avis_par_annee)

# ==========================================================
# GRAPHIQUE ÉVOLUTION
# ==========================================================

plt.figure(figsize=(10, 5))

avis_par_annee.plot(
    kind="line",
    marker="o"
)

plt.title("Évolution du nombre d'avis par année")
plt.xlabel("Année")
plt.ylabel("Nombre d'avis")
plt.grid(True)

plt.tight_layout()

plt.savefig(
    f"{DOSSIER_RESULTATS}/04_evolution_avis.png"
)

plt.close()

# ==========================================================
# 10. SENTIMENT PAR ANNÉE
# ==========================================================

sentiment_annee = pd.crosstab(
    df["Annee"],
    df["Sentiment"],
    normalize="index"
) * 100

print("\n" + "=" * 60)
print("ÉVOLUTION DES SENTIMENTS")
print("=" * 60)

print(sentiment_annee.round(2))

# ==========================================================
# GRAPHIQUE SENTIMENT PAR ANNÉE
# ==========================================================

plt.figure(figsize=(10, 6))

sentiment_annee[
    ["negatif", "neutre", "positif"]
].plot(
    kind="line",
    marker="o"
)

plt.title("Évolution du sentiment des clients")
plt.xlabel("Année")
plt.ylabel("Pourcentage (%)")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    f"{DOSSIER_RESULTATS}/05_evolution_sentiments.png"
)

plt.close()

# ==========================================================
# 11. LONGUEUR DES AVIS
# ==========================================================
df["LongueurAvis"] = df["TextClean"].str.len()


print("\n" + "=" * 60)
print("LONGUEUR DES AVIS")
print("=" * 60)

print(
    df["LongueurAvis"].describe()
)

# ==========================================================
# 12. RÉSUMÉ FINAL
# ==========================================================

print("\n" + "=" * 60)
print("RÉSUMÉ DE L'ANALYSE")
print("=" * 60)

print("Nombre d'avis :", len(df))
print("Nombre de produits :", nombre_produits)
print("Nombre de clients :", nombre_clients)
print("Note moyenne :", round(note_moyenne, 2), "/ 5")

print("\nSentiments :")

for sentiment, nombre in repartition_sentiments.items():

    pourcentage = nombre / len(df) * 100

    print(
        f"{sentiment} : {nombre} "
        f"({pourcentage:.2f} %)"
    )

print("\nLes graphiques sont enregistrés dans :")
print(DOSSIER_RESULTATS)