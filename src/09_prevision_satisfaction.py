import os
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

INPUT_FILE = "data/processed/reviews_clean.csv"

RESULTS_DIR = "results"
FIGURES_DIR = "results/figures"
PREDICTIONS_DIR = "results/predictions"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)


print("=" * 70)
print("ÉTAPE 09 - PRÉVISION DE LA SATISFACTION FUTURE")
print("=" * 70)


# ============================================================
# 2. CHARGEMENT DES DONNÉES
# ============================================================

print("\n[1/7] Chargement des données...")

df = pd.read_csv(
    INPUT_FILE,
    usecols=["Score", "Date"]
)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

df = df.dropna(subset=["Date", "Score"])

print(f"Nombre d'avis utilisés : {len(df):,}")
print(f"Période : {df['Date'].min().date()} -> {df['Date'].max().date()}")


# ============================================================
# 3. AGRÉGATION MENSUELLE
# ============================================================

print("\n[2/7] Calcul de la satisfaction mensuelle...")

df["MoisDate"] = df["Date"].dt.to_period("M").dt.to_timestamp()

mensuel = (
    df.groupby("MoisDate")
    .agg(
        note_moyenne=("Score", "mean"),
        nombre_avis=("Score", "count")
    )
    .sort_index()
)

print("\nPremiers mois :")
print(mensuel.head())

print("\nDerniers mois :")
print(mensuel.tail())


# ============================================================
# 4. CRÉATION D'UNE SÉRIE TEMPORELLE RÉGULIÈRE
# ============================================================

print("\n[3/7] Préparation de la série temporelle...")

# Création de tous les mois entre le début et la fin
index_complet = pd.date_range(
    start=mensuel.index.min(),
    end=mensuel.index.max(),
    freq="MS"
)

mensuel = mensuel.reindex(index_complet)

mensuel.index.name = "Date"

# Certains mois peuvent ne pas avoir d'avis.
# On interpole uniquement la note moyenne pour obtenir
# une série régulière exploitable par Holt-Winters.
mensuel["note_moyenne"] = mensuel["note_moyenne"].interpolate(
    method="linear"
)

mensuel["nombre_avis"] = mensuel["nombre_avis"].fillna(0)

# Sauvegarde de la série mensuelle
mensuel.to_csv(
    os.path.join(RESULTS_DIR, "satisfaction_mensuelle.csv")
)

print(f"Nombre de mois analysés : {len(mensuel)}")

print("\nStatistiques de satisfaction :")
print(f"Note moyenne globale : {mensuel['note_moyenne'].mean():.2f}/5")
print(f"Note minimale : {mensuel['note_moyenne'].min():.2f}/5")
print(f"Note maximale : {mensuel['note_moyenne'].max():.2f}/5")


# ============================================================
# 5. SÉPARATION TRAIN / TEST
# ============================================================

print("\n[4/7] Séparation apprentissage / test...")

serie = mensuel["note_moyenne"].copy()

# On garde les 12 derniers mois pour tester le modèle
test_size = 12

if len(serie) <= test_size + 12:
    raise ValueError(
        "Pas assez de données temporelles pour effectuer "
        "un test sur 12 mois."
    )

train = serie.iloc[:-test_size]
test = serie.iloc[-test_size:]

print(f"Nombre de mois pour entraînement : {len(train)}")
print(f"Nombre de mois pour test : {len(test)}")

print(
    f"Période entraînement : "
    f"{train.index.min().strftime('%Y-%m')} -> "
    f"{train.index.max().strftime('%Y-%m')}"
)

print(
    f"Période test : "
    f"{test.index.min().strftime('%Y-%m')} -> "
    f"{test.index.max().strftime('%Y-%m')}"
)


# ============================================================
# 6. MODÈLE HOLT-WINTERS
# ============================================================

print("\n[5/7] Entraînement du modèle Holt-Winters...")

# Modèle adapté à une série mensuelle :
# - tendance additive
# - saisonnalité additive
# - saisonnalité de 12 mois
# - tendance amortie pour éviter une extrapolation excessive

modele = ExponentialSmoothing(
    train,
    trend="add",
    damped_trend=True,
    seasonal="add",
    seasonal_periods=12,
    initialization_method="estimated"
)

resultat = modele.fit(
    optimized=True
)

# Prévision sur les 12 mois de test
prediction_test = resultat.forecast(test_size)

# Les notes doivent rester dans l'intervalle 1-5
prediction_test = prediction_test.clip(1, 5)


# ============================================================
# 7. ÉVALUATION
# ============================================================

print("\n[6/7] Évaluation du modèle...")

mae = mean_absolute_error(
    test,
    prediction_test
)

rmse = np.sqrt(
    mean_squared_error(
        test,
        prediction_test
    )
)

print(f"\nMAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")

print("\nComparaison réel / prévision :")

comparaison = pd.DataFrame({
    "Date": test.index,
    "Note_reelle": test.values,
    "Note_predite": prediction_test.values
})

comparaison["Erreur"] = (
    comparaison["Note_reelle"]
    - comparaison["Note_predite"]
)

comparaison["Erreur_absolue"] = comparaison["Erreur"].abs()

print(comparaison.to_string(index=False))

comparaison.to_csv(
    os.path.join(
        PREDICTIONS_DIR,
        "evaluation_prevision_satisfaction.csv"
    ),
    index=False
)


# ============================================================
# 8. RÉENTRAÎNEMENT SUR TOUTES LES DONNÉES
# ============================================================

print("\n[7/7] Prévision des 12 prochains mois...")

modele_final = ExponentialSmoothing(
    serie,
    trend="add",
    damped_trend=True,
    seasonal="add",
    seasonal_periods=12,
    initialization_method="estimated"
)

resultat_final = modele_final.fit(
    optimized=True
)

# Prévision des 12 prochains mois
horizon = 12

prevision_future = resultat_final.forecast(horizon)

# Limiter les prédictions entre 1 et 5
prevision_future = prevision_future.clip(1, 5)

# Création du tableau final
dates_future = pd.date_range(
    start=serie.index[-1] + pd.offsets.MonthBegin(1),
    periods=horizon,
    freq="MS"
)

previsions = pd.DataFrame({
    "Date": dates_future,
    "Satisfaction_predite": prevision_future.values
})

print("\nPrévision de satisfaction future :")
print(previsions.to_string(index=False))

# Sauvegarde
previsions.to_csv(
    os.path.join(
        PREDICTIONS_DIR,
        "prevision_satisfaction_12_mois.csv"
    ),
    index=False
)


# ============================================================
# 9. INTERPRÉTATION
# ============================================================

moyenne_historique = serie.mean()
moyenne_future = previsions["Satisfaction_predite"].mean()

variation = (
    (moyenne_future - moyenne_historique)
    / moyenne_historique
) * 100

print("\n" + "=" * 70)
print("INTERPRÉTATION")
print("=" * 70)

print(
    f"\nSatisfaction moyenne historique : "
    f"{moyenne_historique:.2f}/5"
)

print(
    f"Satisfaction moyenne prévue : "
    f"{moyenne_future:.2f}/5"
)

print(
    f"Variation prévue : "
    f"{variation:+.2f}%"
)

if variation > 1:
    conclusion = "La satisfaction devrait suivre une tendance à la hausse."
elif variation < -1:
    conclusion = "La satisfaction devrait suivre une tendance à la baisse."
else:
    conclusion = "La satisfaction devrait rester globalement stable."

print(f"\nConclusion : {conclusion}")


# ============================================================
# 10. GRAPHIQUE HISTORIQUE + PRÉVISION
# ============================================================

plt.figure(figsize=(14, 7))

plt.plot(
    serie.index,
    serie.values,
    label="Satisfaction réelle"
)

plt.plot(
    previsions["Date"],
    previsions["Satisfaction_predite"],
    linestyle="--",
    marker="o",
    label="Prévision Holt-Winters"
)

# Séparation visuelle entre historique et futur
plt.axvline(
    x=serie.index[-1],
    linestyle=":"
)

plt.title(
    "Prévision de la satisfaction client - Holt-Winters"
)

plt.xlabel("Date")
plt.ylabel("Note moyenne / 5")

plt.ylim(1, 5)

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "15_prevision_satisfaction.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 11. GRAPHIQUE ÉVALUATION
# ============================================================

plt.figure(figsize=(14, 7))

plt.plot(
    test.index,
    test.values,
    marker="o",
    label="Valeur réelle"
)

plt.plot(
    test.index,
    prediction_test.values,
    marker="o",
    linestyle="--",
    label="Valeur prédite"
)

plt.title(
    "Évaluation du modèle Holt-Winters"
)

plt.xlabel("Date")
plt.ylabel("Note moyenne / 5")

plt.ylim(1, 5)

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "16_evaluation_modele_temporel.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# FIN
# ============================================================

print("\n" + "=" * 70)
print("ÉTAPE 09 TERMINÉE")
print("=" * 70)

print("\nFichiers générés :")

print(
    "- results/satisfaction_mensuelle.csv"
)

print(
    "- results/predictions/evaluation_prevision_satisfaction.csv"
)

print(
    "- results/predictions/prevision_satisfaction_12_mois.csv"
)

print(
    "- results/figures/15_prevision_satisfaction.png"
)

print(
    "- results/figures/16_evaluation_modele_temporel.png"
)