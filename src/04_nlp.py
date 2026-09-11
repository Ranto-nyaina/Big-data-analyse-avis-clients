import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ============================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================

INPUT_FILE = "data/processed/reviews_clean.csv"
OUTPUT_DIR = "results/figures"

print("=" * 60)
print("ANALYSE NLP DES AVIS CLIENTS")
print("=" * 60)

print("\n1. Chargement des données...")

df = pd.read_csv(INPUT_FILE)

print(f"Nombre d'avis : {len(df)}")

# Supprimer les textes vides
df = df.dropna(subset=["TextClean"])
df = df[df["TextClean"].str.strip() != ""]

print(f"Nombre d'avis utilisables : {len(df)}")


# ============================================================
# 2. LISTE DES MOTS À IGNORER
# ============================================================

stopwords = {
    "the", "and", "a", "to", "of", "is", "in", "it", "for",
    "this", "that", "on", "with", "was", "as", "but", "are",
    "be", "have", "had", "not", "my", "i", "we", "you",
    "they", "he", "she", "from", "or", "at", "so", "very",
    "our", "your", "just", "all", "can", "would", "will",
    "one", "about", "more", "there", "what", "when", "which",
    "some", "than", "also", "out", "get", "has", "been",
    "were", "if", "their", "them", "me", "do", "did", "no",
    "too", "only", "up", "an", "by", "because", "really",
    "food", "product", "amazon", "review"
}


# ============================================================
# 3. FONCTION DE TOKENISATION
# ============================================================

def extraire_mots(textes):
    compteur = Counter()

    for texte in textes:
        mots = re.findall(r"\b[a-z]{3,}\b", str(texte).lower())

        for mot in mots:
            if mot not in stopwords:
                compteur[mot] += 1

    return compteur


# ============================================================
# 4. MOTS LES PLUS FRÉQUENTS
# ============================================================

print("\n" + "=" * 60)
print("MOTS LES PLUS FRÉQUENTS")
print("=" * 60)

compteur_global = extraire_mots(df["TextClean"])

top_mots = compteur_global.most_common(20)

print("\nTop 20 des mots :")

for mot, nombre in top_mots:
    print(f"{mot:20} {nombre}")


# ============================================================
# 5. GRAPHIQUE DES MOTS LES PLUS FRÉQUENTS
# ============================================================

mots = [x[0] for x in top_mots]
frequences = [x[1] for x in top_mots]

plt.figure(figsize=(12, 7))
plt.barh(mots[::-1], frequences[::-1])
plt.xlabel("Nombre d'occurrences")
plt.ylabel("Mots")
plt.title("Top 20 des mots les plus fréquents")
plt.tight_layout()

plt.savefig(f"{OUTPUT_DIR}/06_top_mots.png", dpi=150)
plt.close()

print("\nGraphique enregistré : results/figures/06_top_mots.png")


# ============================================================
# 6. WORDCLOUD GLOBAL
# ============================================================

print("\nGénération du WordCloud global...")

wordcloud = WordCloud(
    width=1600,
    height=800,
    background_color="white",
    max_words=100
).generate_from_frequencies(compteur_global)

plt.figure(figsize=(16, 8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("WordCloud des avis clients")

plt.savefig(
    f"{OUTPUT_DIR}/07_wordcloud_global.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("WordCloud global enregistré.")


# ============================================================
# 7. ANALYSE DES AVIS POSITIFS
# ============================================================

print("\n" + "=" * 60)
print("ANALYSE DES AVIS POSITIFS")
print("=" * 60)

df_positif = df[df["Sentiment"] == "positif"]

print(f"Nombre d'avis positifs : {len(df_positif)}")

compteur_positif = extraire_mots(df_positif["TextClean"])

top_positifs = compteur_positif.most_common(20)

print("\nTop 20 des mots dans les avis positifs :")

for mot, nombre in top_positifs:
    print(f"{mot:20} {nombre}")


# WordCloud positif

wordcloud = WordCloud(
    width=1600,
    height=800,
    background_color="white",
    max_words=100
).generate_from_frequencies(compteur_positif)

plt.figure(figsize=(16, 8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("WordCloud des avis positifs")

plt.savefig(
    f"{OUTPUT_DIR}/08_wordcloud_positif.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("WordCloud positif enregistré.")


# ============================================================
# 8. ANALYSE DES AVIS NÉGATIFS
# ============================================================

print("\n" + "=" * 60)
print("ANALYSE DES AVIS NÉGATIFS")
print("=" * 60)

df_negatif = df[df["Sentiment"] == "negatif"]

print(f"Nombre d'avis négatifs : {len(df_negatif)}")

compteur_negatif = extraire_mots(df_negatif["TextClean"])

top_negatifs = compteur_negatif.most_common(20)

print("\nTop 20 des mots dans les avis négatifs :")

for mot, nombre in top_negatifs:
    print(f"{mot:20} {nombre}")


# WordCloud négatif

wordcloud = WordCloud(
    width=1600,
    height=800,
    background_color="white",
    max_words=100
).generate_from_frequencies(compteur_negatif)

plt.figure(figsize=(16, 8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("WordCloud des avis négatifs")

plt.savefig(
    f"{OUTPUT_DIR}/09_wordcloud_negatif.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("WordCloud négatif enregistré.")


# ============================================================
# 9. MOTS CARACTÉRISTIQUES DES AVIS NÉGATIFS
# ============================================================

print("\n" + "=" * 60)
print("PROBLÈMES IDENTIFIÉS DANS LES AVIS NÉGATIFS")
print("=" * 60)

# Mots positifs et négatifs
freq_positive = Counter(compteur_positif)
freq_negative = Counter(compteur_negatif)

mots_caracteristiques = []

for mot, freq in freq_negative.items():

    freq_pos = freq_positive.get(mot, 0)

    # Le mot doit apparaître suffisamment souvent
    if freq >= 20:

        ratio = freq / (freq_pos + 1)

        mots_caracteristiques.append(
            (mot, freq, freq_pos, ratio)
        )

mots_caracteristiques.sort(
    key=lambda x: x[3],
    reverse=True
)

print("\nMots particulièrement associés aux avis négatifs :")

for mot, freq_neg, freq_pos, ratio in mots_caracteristiques[:30]:

    print(
        f"{mot:20} "
        f"negatif={freq_neg:6} | "
        f"positif={freq_pos:6} | "
        f"ratio={ratio:.2f}"
    )


# ============================================================
# 10. THÈMES SIMPLES
# ============================================================

print("\n" + "=" * 60)
print("ANALYSE DES THÈMES")
print("=" * 60)

themes = {
    "Qualité": [
        "quality", "good", "bad", "taste", "flavor",
        "fresh", "ingredient"
    ],

    "Prix": [
        "price", "expensive", "cheap", "cost", "money"
    ],

    "Livraison": [
        "shipping", "delivery", "delivered", "package",
        "arrived", "arrival"
    ],

    "Emballage": [
        "package", "packaging", "box", "container"
    ],

    "Goût": [
        "taste", "flavor", "sweet", "salty", "bitter",
        "delicious"
    ],

    "Service": [
        "service", "seller", "customer", "support"
    ],

    "Quantité": [
        "size", "small", "large", "quantity", "amount"
    ]
}

resultats_themes = []

for theme, mots_theme in themes.items():

    total = 0

    for mot in mots_theme:
        total += compteur_global.get(mot, 0)

    resultats_themes.append(
        (theme, total)
    )


resultats_themes.sort(
    key=lambda x: x[1],
    reverse=True
)

print("\nFréquence des thèmes :")

for theme, nombre in resultats_themes:
    print(f"{theme:20} {nombre}")


# ============================================================
# 11. GRAPHIQUE DES THÈMES
# ============================================================

themes_noms = [x[0] for x in resultats_themes]
themes_valeurs = [x[1] for x in resultats_themes]

plt.figure(figsize=(10, 6))

plt.barh(
    themes_noms[::-1],
    themes_valeurs[::-1]
)

plt.xlabel("Occurrences")
plt.ylabel("Thème")
plt.title("Thèmes présents dans les avis clients")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/10_themes.png",
    dpi=150
)

plt.close()

print("\nGraphique enregistré : results/figures/10_themes.png")


# ============================================================
# 12. RÉSUMÉ
# ============================================================

print("\n" + "=" * 60)
print("RÉSUMÉ NLP")
print("=" * 60)

print(f"""
Nombre total d'avis analysés : {len(df)}

Avis positifs : {len(df_positif)}
Avis négatifs : {len(df_negatif)}

Nombre de mots différents :
{len(compteur_global)}

Top 5 des mots :
{compteur_global.most_common(5)}

Les résultats graphiques sont disponibles dans :
results/figures/
""")

print("=" * 60)
print("ANALYSE NLP TERMINÉE")
print("=" * 60)