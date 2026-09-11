import pandas as pd

FICHIER = "data/raw/reviews.csv"

df = pd.read_csv(FICHIER)

print("=" * 50)
print("INFORMATIONS SUR LE DATASET")
print("=" * 50)

print("Nombre de lignes :", len(df))
print("Nombre de colonnes :", len(df.columns))

print("\nColonnes :")
for colonne in df.columns:
    print("-", colonne)

print("\nPremières lignes :")
print(df.head())

print("\nValeurs manquantes :")
print(df.isnull().sum())

print("\nDoublons :", df.duplicated().sum())