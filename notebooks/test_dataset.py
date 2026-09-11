import pandas as pd

fichier = "data/raw/reviews.csv"

df = pd.read_csv(fichier)

print("Nombre de lignes :", len(df))
print("Nombre de colonnes :", len(df.columns))

print("\nColonnes :")
print(df.columns.tolist())

print("\nPremières lignes :")
print(df.head())

print("\nInformations :")
print(df.info())