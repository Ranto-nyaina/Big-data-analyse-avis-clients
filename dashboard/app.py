import os
from pathlib import Path

import pandas as pd
import numpy as np

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import joblib
import requests

from streamlit_autorefresh import st_autorefresh


# ==========================================================
# CONFIGURATION
# ==========================================================

st.set_page_config(

    page_title="Analyse des Avis Clients",

    page_icon="📊",

    layout="wide",

    initial_sidebar_state="expanded"
)


# ==========================================================
# CHEMINS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "dashboard" / "data"

FIGURES_DIR = BASE_DIR / "dashboard" / "figures"

MODELS_DIR = BASE_DIR / "models"

RESULTS_DIR = BASE_DIR / "results"


# ==========================================================
# API
# ==========================================================

API_URL = "http://127.0.0.1:8000"


# ==========================================================
# STYLE
# ==========================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .metric-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #eeeeee;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# CHARGEMENT DONNEES
# ==========================================================

@st.cache_data
def charger_csv(path):

    if not path.exists():

        return pd.DataFrame()

    try:

        return pd.read_csv(path)

    except Exception:

        try:

            return pd.read_csv(
                path,
                encoding="latin-1"
            )

        except Exception:

            return pd.DataFrame()


@st.cache_data
def charger_donnees():

    donnees = {}

    donnees["synthese"] = charger_csv(
        DATA_DIR / "synthese_dashboard.csv"
    )

    donnees["avis_annee"] = charger_csv(
        DATA_DIR / "avis_par_annee.csv"
    )

    donnees["notes_annee"] = charger_csv(
        DATA_DIR / "notes_par_annee.csv"
    )

    donnees["sentiments_annee"] = charger_csv(
        DATA_DIR / "sentiments_par_annee.csv"
    )

    donnees["notes"] = charger_csv(
        DATA_DIR / "repartition_notes.csv"
    )

    donnees["produits"] = charger_csv(
        DATA_DIR / "produits_top.csv"
    )

    donnees["avis_utiles"] = charger_csv(
        DATA_DIR / "avis_utiles.csv"
    )

    donnees["themes"] = charger_csv(
        DATA_DIR / "themes.csv"
    )

    donnees["prevision"] = charger_csv(
        DATA_DIR / "satisfaction_prevision.csv"
    )

    return donnees


donnees = charger_donnees()


# ==========================================================
# FONCTIONS UTILITAIRES
# ==========================================================

def obtenir_valeur_synthese(
    nom,
    valeur_defaut=0
):

    df = donnees["synthese"]

    if df.empty:

        return valeur_defaut

    try:

        ligne = df[
            df["Indicateur"]
            .astype(str)
            .str.lower()
            ==
            nom.lower()
        ]

        if not ligne.empty:

            return ligne.iloc[0]["Valeur"]

    except Exception:

        pass

    return valeur_defaut


def convertir_numerique(
    df,
    colonne
):

    if colonne not in df.columns:

        return df

    df = df.copy()

    df[colonne] = pd.to_numeric(
        df[colonne],
        errors="coerce"
    )

    return df


def trouver_colonne(
    df,
    mots
):

    for colonne in df.columns:

        colonne_lower = str(
            colonne
        ).lower()

        for mot in mots:

            if mot.lower() in colonne_lower:

                return colonne

    return None


def afficher_image(chemin, largeur=None):
    """
    Affiche une image si elle existe.
    Compatible avec les versions récentes de Streamlit.
    """

    if not chemin:
        return

    chemin = Path(chemin)

    if not chemin.exists():
        st.warning(f"Image introuvable : {chemin}")
        return

    try:
        if largeur is None:
            st.image(str(chemin), width="stretch")
        else:
            st.image(str(chemin), width=largeur)

    except Exception as e:
        st.warning(f"Impossible d'afficher l'image : {e}")

# ==========================================================
# PAGE 1 - VUE GENERALE
# ==========================================================

def page_vue_generale():

    st.title(
        "🏠 Vue générale"
    )

    st.write(
        "Analyse globale des avis clients."
    )

    total = obtenir_valeur_synthese(
        "Nombre total d'avis"
    )

    note_moyenne = obtenir_valeur_synthese(
        "Note moyenne"
    )

    positifs = obtenir_valeur_synthese(
        "Avis positifs"
    )

    negatifs = obtenir_valeur_synthese(
        "Avis négatifs"
    )

    neutres = obtenir_valeur_synthese(
        "Avis neutres"
    )

    pct_positifs = obtenir_valeur_synthese(
        "% avis positifs"
    )

    pct_negatifs = obtenir_valeur_synthese(
        "% avis négatifs"
    )

    pct_neutres = obtenir_valeur_synthese(
        "% avis neutres"
    )

    # ------------------------------------------------------
    # KPI
    # ------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "📝 Nombre total d'avis",
            f"{int(float(total)):,}"
        )

    with c2:

        st.metric(
            "⭐ Note moyenne",
            f"{float(note_moyenne):.2f}/5"
        )

    with c3:

        st.metric(
            "😊 Avis positifs",
            f"{int(float(positifs)):,}",
            f"{float(pct_positifs):.2f}%"
        )

    with c4:

        st.metric(
            "😡 Avis négatifs",
            f"{int(float(negatifs)):,}",
            f"{float(pct_negatifs):.2f}%"
        )

    st.divider()

    # ------------------------------------------------------
    # SENTIMENTS
    # ------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Répartition des sentiments"
        )

        sentiment_df = pd.DataFrame({

            "Sentiment": [
                "Positif",
                "Neutre",
                "Négatif"
            ],

            "Nombre": [
                float(positifs),
                float(neutres),
                float(negatifs)
            ]
        })

        fig = px.pie(

            sentiment_df,

            names="Sentiment",

            values="Nombre",

            hole=0.4,

            title="Sentiments des avis"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        st.subheader(
            "Répartition des notes"
        )

        df_notes = donnees["notes"].copy()

        if not df_notes.empty:

            col_score = trouver_colonne(
                df_notes,
                ["score", "note"]
            )

            col_count = trouver_colonne(
                df_notes,
                ["nombre", "count", "avis"]
            )

            if col_score and col_count:

                df_notes = convertir_numerique(
                    df_notes,
                    col_score
                )

                df_notes = convertir_numerique(
                    df_notes,
                    col_count
                )

                fig = px.bar(

                    df_notes,

                    x=col_score,

                    y=col_count,

                    text_auto=True,

                    title="Distribution des notes"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

    # ------------------------------------------------------
    # EVOLUTION AVIS
    # ------------------------------------------------------

    st.subheader(
        "Évolution du nombre d'avis"
    )

    df = donnees["avis_annee"].copy()

    if not df.empty:

        col_year = trouver_colonne(
            df,
            ["annee", "year"]
        )

        col_count = trouver_colonne(
            df,
            ["nombre", "count", "avis"]
        )

        if col_year and col_count:

            df = convertir_numerique(
                df,
                col_year
            )

            df = convertir_numerique(
                df,
                col_count
            )

            fig = px.line(

                df.sort_values(col_year),

                x=col_year,

                y=col_count,

                markers=True,

                title="Nombre d'avis par année"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # ------------------------------------------------------
    # RESUME
    # ------------------------------------------------------

    st.subheader(
        "📌 Résumé"
    )

    st.write(
        f"""
        Le dataset contient **{int(float(total)):,} avis clients**.

        La note moyenne globale est de
        **{float(note_moyenne):.2f}/5**.

        Les avis positifs représentent
        **{float(pct_positifs):.2f}%** des avis.

        Les avis négatifs représentent
        **{float(pct_negatifs):.2f}%**.

        Les avis neutres représentent
        **{float(pct_neutres):.2f}%**.
        """
    )


# ==========================================================
# PAGE 2 - EVOLUTION TEMPORELLE
# ==========================================================

def page_evolution():

    st.title(
        "📈 Évolution temporelle"
    )

    df_avis = donnees["avis_annee"].copy()

    df_notes = donnees["notes_annee"].copy()

    df_sentiments = donnees[
        "sentiments_annee"
    ].copy()

    if df_avis.empty:

        st.warning(
            "Données temporelles indisponibles."
        )

        return

    col_year = trouver_colonne(
        df_avis,
        ["annee", "year"]
    )

    col_count = trouver_colonne(
        df_avis,
        ["nombre", "count", "avis"]
    )

    if not col_year or not col_count:

        st.dataframe(
            df_avis,
            use_container_width=True
        )

        return

    df_avis = convertir_numerique(
        df_avis,
        col_year
    )

    df_avis = convertir_numerique(
        df_avis,
        col_count
    )

    annees = sorted(
        df_avis[col_year]
        .dropna()
        .unique()
    )

    selection = st.slider(
        "Période",
        int(min(annees)),
        int(max(annees)),
        (
            int(min(annees)),
            int(max(annees))
        )
    )

    df_filtre = df_avis[
        (df_avis[col_year] >= selection[0])
        &
        (df_avis[col_year] <= selection[1])
    ]

    # ------------------------------------------------------
    # AVIS PAR ANNEE
    # ------------------------------------------------------

    st.subheader(
        "📝 Nombre d'avis par année"
    )

    fig = px.line(

        df_filtre.sort_values(col_year),

        x=col_year,

        y=col_count,

        markers=True,

        title="Évolution du nombre d'avis"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ------------------------------------------------------
    # NOTE MOYENNE
    # ------------------------------------------------------

    if not df_notes.empty:

        col_year_note = trouver_colonne(
            df_notes,
            ["annee", "year"]
        )

        col_note = trouver_colonne(
            df_notes,
            ["note", "moyenne", "score"]
        )

        if col_year_note and col_note:

            df_notes = convertir_numerique(
                df_notes,
                col_year_note
            )

            df_notes = convertir_numerique(
                df_notes,
                col_note
            )

            df_notes_filtre = df_notes[
                (df_notes[col_year_note] >= selection[0])
                &
                (df_notes[col_year_note] <= selection[1])
            ]

            st.subheader(
                "⭐ Note moyenne par année"
            )

            fig = px.line(

                df_notes_filtre.sort_values(
                    col_year_note
                ),

                x=col_year_note,

                y=col_note,

                markers=True,

                title="Évolution de la note moyenne"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # ------------------------------------------------------
    # SENTIMENTS
    # ------------------------------------------------------

    if not df_sentiments.empty:

        st.subheader(
            "😊 Évolution des sentiments"
        )

        df_s = df_sentiments.copy()

        col_year_s = trouver_colonne(
            df_s,
            ["annee", "year"]
        )

        if col_year_s:

            df_s = convertir_numerique(
                df_s,
                col_year_s
            )

            # Cas où le fichier est déjà en format large
            col_pos = trouver_colonne(
                df_s,
                ["positif", "positive"]
            )

            col_neg = trouver_colonne(
                df_s,
                ["negatif", "negative"]
            )

            col_neu = trouver_colonne(
                df_s,
                ["neutre", "neutral"]
            )

            if col_pos and col_neg and col_neu:

                for col in [
                    col_pos,
                    col_neg,
                    col_neu
                ]:

                    df_s = convertir_numerique(
                        df_s,
                        col
                    )

                df_s_filtre = df_s[
                    (df_s[col_year_s] >= selection[0])
                    &
                    (df_s[col_year_s] <= selection[1])
                ]

                fig = go.Figure()

                fig.add_trace(
                    go.Scatter(
                        x=df_s_filtre[col_year_s],
                        y=df_s_filtre[col_pos],
                        mode="lines+markers",
                        name="Positif"
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=df_s_filtre[col_year_s],
                        y=df_s_filtre[col_neu],
                        mode="lines+markers",
                        name="Neutre"
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=df_s_filtre[col_year_s],
                        y=df_s_filtre[col_neg],
                        mode="lines+markers",
                        name="Négatif"
                    )
                )

                fig.update_layout(
                    title="Évolution des sentiments",
                    xaxis_title="Année",
                    yaxis_title="Nombre d'avis"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            else:

                st.dataframe(
                    df_s,
                    use_container_width=True
                )

    # ------------------------------------------------------
    # TABLEAU
    # ------------------------------------------------------

    st.subheader(
        "📋 Données temporelles"
    )

    st.dataframe(
        df_filtre,
        use_container_width=True,
        hide_index=True
    )


# ==========================================================
# PAGE 3 - PRODUITS
# ==========================================================

def page_produits():

    st.title(
        "📦 Produits"
    )

    df = donnees["produits"].copy()

    if df.empty:

        st.warning(
            "Données produits indisponibles."
        )

        return

    col_product = trouver_colonne(
        df,
        ["productid", "product", "produit"]
    )

    col_count = trouver_colonne(
        df,
        ["nombre", "count", "avis", "reviews"]
    )

    col_rating = trouver_colonne(
        df,
        ["moyenne", "rating", "score", "note"]
    )

    # ------------------------------------------------------
    # RECHERCHE
    # ------------------------------------------------------

    recherche = st.text_input(
        "🔎 Rechercher un produit",
        placeholder="Exemple : B007JFMH8M"
    )

    if recherche and col_product:

        df_recherche = df[
            df[col_product]
            .astype(str)
            .str.contains(
                recherche,
                case=False,
                na=False
            )
        ]

    else:

        df_recherche = df.copy()

    # ------------------------------------------------------
    # TOP N
    # ------------------------------------------------------

    nombre = st.slider(
        "Nombre de produits à afficher",
        5,
        min(50, len(df_recherche)),
        min(10, len(df_recherche))
    )

    df_top = df_recherche.head(
        nombre
    ).copy()

    # ------------------------------------------------------
    # KPI
    # ------------------------------------------------------

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Produits affichés",
            len(df_top)
        )

    with c2:

        if col_count:

            valeur = pd.to_numeric(
                df_top[col_count],
                errors="coerce"
            ).sum()

            st.metric(
                "Avis",
                f"{int(valeur):,}"
            )

    with c3:

        if col_rating:

            moyenne = pd.to_numeric(
                df_top[col_rating],
                errors="coerce"
            ).mean()

            st.metric(
                "Note moyenne",
                f"{moyenne:.2f}/5"
            )

    # ------------------------------------------------------
    # NOMBRE AVIS
    # ------------------------------------------------------

    if col_product and col_count:

        st.subheader(
            "📊 Produits avec le plus d'avis"
        )

        fig = px.bar(

            df_top,

            x=col_product,

            y=col_count,

            text_auto=True,

            title="Top produits par nombre d'avis"
        )

        fig.update_layout(
            xaxis_title="Produit",
            yaxis_title="Nombre d'avis"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ------------------------------------------------------
    # NOTE
    # ------------------------------------------------------

    if col_product and col_rating:

        st.subheader(
            "⭐ Note moyenne des produits"
        )

        fig = px.bar(

            df_top,

            x=col_product,

            y=col_rating,

            text_auto=True,

            title="Note moyenne"
        )

        fig.update_layout(
            xaxis_title="Produit",
            yaxis_title="Note moyenne"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ------------------------------------------------------
    # TABLEAU
    # ------------------------------------------------------

    st.subheader(
        "📋 Produits"
    )

    st.dataframe(
        df_top,
        use_container_width=True,
        hide_index=True
    )


# ==========================================================
# PAGE 4 - THEMES
# ==========================================================

def page_themes():

    st.title(
        "💬 Analyse des thèmes"
    )

    st.write(
        "Analyse des principaux thèmes présents dans les avis clients."
    )

    df = donnees["themes"].copy()

    if not df.empty:

        col_theme = trouver_colonne(
            df,
            ["theme", "thème"]
        )

        col_count = trouver_colonne(
            df,
            ["nombre", "count", "avis"]
        )

        if col_theme and col_count:

            df[col_count] = pd.to_numeric(
                df[col_count],
                errors="coerce"
            )

            df = df.sort_values(
                col_count,
                ascending=False
            )

            fig = px.bar(

                df,

                x=col_theme,

                y=col_count,

                text_auto=True,

                title="Principaux thèmes"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.subheader(
                "📋 Tableau des thèmes"
            )

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.dataframe(
                df,
                use_container_width=True
            )

    # ------------------------------------------------------
    # WORDCLOUDS
    # ------------------------------------------------------

    st.subheader(
        "☁️ WordCloud"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.write(
            "WordCloud global"
        )

        afficher_image(
            str(
                FIGURES_DIR /
                "07_wordcloud_global.png"
            )
        )

    with col2:

        st.write(
            "WordCloud positif"
        )

        afficher_image(
            str(
                FIGURES_DIR /
                "08_wordcloud_positif.png"
            )
        )

    col3, col4 = st.columns(2)

    with col3:

        st.write(
            "WordCloud négatif"
        )

        afficher_image(
            str(
                FIGURES_DIR /
                "09_wordcloud_negatif.png"
            )
        )

    with col4:

        st.write(
            "Analyse des thèmes"
        )

        afficher_image(
            str(
                FIGURES_DIR /
                "10_themes.png"
            )
        )


# ==========================================================
# PAGE 5 - IA & PREVISION
# ==========================================================

def page_ia_prevision():

    st.title(
        "🤖 IA & Prévision"
    )

    st.write(
        "Prédiction du sentiment, prédiction de la note "
        "et prévision de la satisfaction."
    )

    # ------------------------------------------------------
    # MODELES
    # ------------------------------------------------------

    sentiment_model_file = (
        MODELS_DIR /
        "sentiment_model.pkl"
    )

    sentiment_vectorizer_file = (
        MODELS_DIR /
        "tfidf_vectorizer.pkl"
    )

    rating_model_file = (
        MODELS_DIR /
        "rating_model.pkl"
    )

    rating_vectorizer_file = (
        MODELS_DIR /
        "rating_tfidf_vectorizer.pkl"
    )

    modeles_ok = all([

        sentiment_model_file.exists(),

        sentiment_vectorizer_file.exists(),

        rating_model_file.exists(),

        rating_vectorizer_file.exists()
    ])

    if modeles_ok:

        st.success(
            "✅ Les modèles IA sont disponibles."
        )

    else:

        st.error(
            "❌ Un ou plusieurs modèles sont introuvables."
        )

    # ------------------------------------------------------
    # PREDICTION AVIS
    # ------------------------------------------------------

    st.subheader(
        "🧠 Analyse d'un nouvel avis"
    )

    texte = st.text_area(
        "Entrez un avis client",
        placeholder=(
            "Exemple : "
            "This coffee tastes excellent and "
            "I really love it."
        ),
        height=150
    )

    if st.button(
        "🔮 Analyser l'avis",
        type="primary"
    ):

        if not texte.strip():

            st.warning(
                "Veuillez saisir un avis."
            )

        elif not modeles_ok:

            st.error(
                "Les modèles ne sont pas disponibles."
            )

        else:

            try:

                sentiment_model = joblib.load(
                    sentiment_model_file
                )

                sentiment_vectorizer = joblib.load(
                    sentiment_vectorizer_file
                )

                rating_model = joblib.load(
                    rating_model_file
                )

                rating_vectorizer = joblib.load(
                    rating_vectorizer_file
                )

                # ------------------------------------------
                # SENTIMENT
                # ------------------------------------------

                vector_sentiment = (
                    sentiment_vectorizer.transform(
                        [texte]
                    )
                )

                sentiment = (
                    sentiment_model.predict(
                        vector_sentiment
                    )[0]
                )

                probabilities_sentiment = (
                    sentiment_model.predict_proba(
                        vector_sentiment
                    )[0]
                )

                confidence_sentiment = (
                    np.max(
                        probabilities_sentiment
                    )
                )

                # ------------------------------------------
                # NOTE
                # ------------------------------------------

                vector_rating = (
                    rating_vectorizer.transform(
                        [texte]
                    )
                )

                rating = (
                    rating_model.predict(
                        vector_rating
                    )[0]
                )

                probabilities_rating = (
                    rating_model.predict_proba(
                        vector_rating
                    )[0]
                )

                confidence_rating = (
                    np.max(
                        probabilities_rating
                    )
                )

                rating = max(
                    1,
                    min(
                        5,
                        int(round(float(rating)))
                    )
                )

                # ------------------------------------------
                # RESULTATS
                # ------------------------------------------

                c1, c2, c3, c4 = st.columns(4)

                with c1:

                    st.metric(
                        "Sentiment",
                        str(sentiment)
                    )

                with c2:

                    st.metric(
                        "Confiance sentiment",
                        f"{confidence_sentiment * 100:.2f}%"
                    )

                with c3:

                    st.metric(
                        "Note prédite",
                        f"{rating}/5"
                    )

                with c4:

                    st.metric(
                        "Confiance note",
                        f"{confidence_rating * 100:.2f}%"
                    )

            except Exception as e:

                st.error(
                    f"Erreur pendant la prédiction : {e}"
                )

    st.divider()

    # ------------------------------------------------------
    # PREVISION
    # ------------------------------------------------------

    st.subheader(
        "📈 Prévision de la satisfaction"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Moyenne historique",
            "4.31/5"
        )

    with col2:

        st.metric(
            "Prévision future",
            "4.09/5"
        )

    with col3:

        st.metric(
            "Variation",
            "-4.91%"
        )

    with col4:

        st.metric(
            "MAE",
            "0.1359"
        )

    # ------------------------------------------------------
    # GRAPHIQUE
    # ------------------------------------------------------

    df_prevision = donnees[
        "prevision"
    ].copy()

    if not df_prevision.empty:

        st.subheader(
            "🔮 Prévision des 12 prochains mois"
        )

        col_date = trouver_colonne(
            df_prevision,
            ["date", "mois", "month"]
        )

        col_value = trouver_colonne(
            df_prevision,
            ["prevision", "prediction", "satisfaction"]
        )

        if col_date and col_value:

            df_prevision[col_date] = pd.to_datetime(
                df_prevision[col_date],
                errors="coerce"
            )

            df_prevision[col_value] = pd.to_numeric(
                df_prevision[col_value],
                errors="coerce"
            )

            fig = px.line(

                df_prevision,

                x=col_date,

                y=col_value,

                markers=True,

                title=(
                    "Prévision de la satisfaction "
                    "pour les 12 prochains mois"
                )
            )

            fig.update_yaxes(
                range=[0, 5]
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.dataframe(
                df_prevision,
                use_container_width=True,
                hide_index=True
            )

    # ------------------------------------------------------
    # INTERPRETATION
    # ------------------------------------------------------

    st.subheader(
        "📌 Interprétation"
    )

    st.info(
        """
        Le modèle Holt-Winters prévoit une satisfaction
        moyenne d'environ 4,09/5 pour les 12 prochains mois,
        contre une moyenne historique de 4,31/5.

        Cela représente une diminution estimée d'environ
        4,91 %.

        Cette prévision représente une tendance basée sur
        les données historiques et ne constitue pas une
        certitude.
        """
    )


# ==========================================================
# PAGE 6 - AVIS EN TEMPS REEL
# ==========================================================

def page_avis_temps_reel():

    st.title(
        "📡 Avis en temps réel"
    )

    st.write(
        """
        Cette page permet de recevoir les nouveaux avis,
        de les analyser automatiquement avec les modèles
        de Machine Learning et de les enregistrer dans
        une base SQLite.
        """
    )

    # ------------------------------------------------------
    # ACTUALISATION AUTOMATIQUE
    # ------------------------------------------------------

    st_autorefresh(

        interval=5000,

        key="realtime_refresh"
    )

    # ------------------------------------------------------
    # ETAT API
    # ------------------------------------------------------

    try:

        response_health = requests.get(

            f"{API_URL}/api/health",

            timeout=3
        )

        if response_health.status_code == 200:

            health = response_health.json()

            if health.get(
                "models_loaded",
                False
            ):

                st.success(
                    "🟢 API FastAPI connectée — modèles IA chargés"
                )

            else:

                st.warning(
                    "🟡 API connectée mais modèles IA non chargés"
                )

        else:

            st.error(
                "🔴 API FastAPI indisponible"
            )

            return

    except requests.exceptions.RequestException:

        st.error(
            """
            🔴 Impossible de contacter FastAPI.

            Lancez l'API dans un autre terminal :

            `python -m uvicorn api.main:app --reload --port 8000`
            """
        )

        return

    # ------------------------------------------------------
    # STATISTIQUES
    # ------------------------------------------------------

    try:

        response_stats = requests.get(

            f"{API_URL}/api/reviews/stats",

            timeout=3
        )

        stats = response_stats.json()

    except Exception as e:

        st.error(
            f"Erreur récupération statistiques : {e}"
        )

        return

    # ------------------------------------------------------
    # KPI
    # ------------------------------------------------------

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:

        st.metric(
            "📝 Avis reçus",
            stats.get(
                "total",
                0
            )
        )

    with c2:

        st.metric(
            "😊 Positifs",
            stats.get(
                "positif",
                0
            ),
            f"{stats.get('pourcentage_positif', 0):.2f}%"
        )

    with c3:

        st.metric(
            "😐 Neutres",
            stats.get(
                "neutre",
                0
            ),
            f"{stats.get('pourcentage_neutre', 0):.2f}%"
        )

    with c4:

        st.metric(
            "😡 Négatifs",
            stats.get(
                "negatif",
                0
            ),
            f"{stats.get('pourcentage_negatif', 0):.2f}%"
        )

    with c5:

        st.metric(
            "⭐ Note moyenne",
            f"{stats.get('note_moyenne', 0):.2f}/5"
        )

    st.divider()

    # ------------------------------------------------------
    # NOUVEL AVIS
    # ------------------------------------------------------

    st.subheader(
        "➕ Ajouter un nouvel avis"
    )

    with st.form(
        "form_nouvel_avis",
        clear_on_submit=True
    ):

        c1, c2 = st.columns(2)

        with c1:

            product_id = st.text_input(
                "Product ID",
                value="B007JFMH8M"
            )

        with c2:

            user_name = st.text_input(
                "Utilisateur",
                value="Utilisateur"
            )

        texte = st.text_area(

            "Texte de l'avis",

            placeholder=(
                "Écrivez ici votre avis client..."
            ),

            height=120
        )

        score = st.selectbox(

            "Note donnée par le client",

            [
                "Non renseignée",
                1,
                2,
                3,
                4,
                5
            ]
        )

        envoyer = st.form_submit_button(
            "🚀 Analyser et enregistrer",
            type="primary"
        )

    # ------------------------------------------------------
    # ENVOI VERS FASTAPI
    # ------------------------------------------------------

    if envoyer:

        if not texte.strip():

            st.warning(
                "Veuillez saisir un avis."
            )

        else:

            if score == "Non renseignée":

                score_value = None

            else:

                score_value = int(score)

            payload = {

                "product_id":
                    product_id,

                "user_name":
                    user_name,

                "text":
                    texte,

                "score":
                    score_value
            }

            try:

                response = requests.post(

                    f"{API_URL}/api/reviews",

                    json=payload,

                    timeout=15
                )

                if response.status_code == 200:

                    result = response.json()

                    st.success(
                        "✅ Avis analysé et enregistré !"
                    )

                    st.subheader(
                        "🤖 Résultat de l'intelligence artificielle"
                    )

                    c1, c2, c3, c4 = st.columns(4)

                    with c1:

                        sentiment = result[
                            "sentiment"
                        ]

                        if sentiment == "positif":

                            st.success(
                                f"😊 {sentiment}"
                            )

                        elif sentiment == "negatif":

                            st.error(
                                f"😡 {sentiment}"
                            )

                        else:

                            st.warning(
                                f"😐 {sentiment}"
                            )

                    with c2:

                        st.metric(
                            "Confiance sentiment",
                            f"{result['sentiment_confidence']}%"
                        )

                    with c3:

                        st.metric(
                            "Note prédite",
                            f"{result['predicted_score']}/5"
                        )

                    with c4:

                        st.metric(
                            "Confiance note",
                            f"{result['rating_confidence']}%"
                        )

                else:

                    try:

                        error = response.json()

                    except Exception:

                        error = response.text

                    st.error(
                        f"Erreur API : {error}"
                    )

            except requests.exceptions.RequestException as e:

                st.error(
                    f"Erreur de connexion à FastAPI : {e}"
                )

    st.divider()

    # ------------------------------------------------------
    # DERNIERS AVIS
    # ------------------------------------------------------

    st.subheader(
        "🕐 Derniers avis reçus"
    )

    try:

        response_reviews = requests.get(

            f"{API_URL}/api/reviews?limit=20",

            timeout=5
        )

        if response_reviews.status_code == 200:

            data = response_reviews.json()

            reviews = data.get(
                "reviews",
                []
            )

            if reviews:

                df = pd.DataFrame(
                    reviews
                )

                # ------------------------------------------
                # RENOMMAGE
                # ------------------------------------------

                rename_columns = {

                    "id":
                        "ID",

                    "product_id":
                        "Produit",

                    "user_name":
                        "Utilisateur",

                    "text":
                        "Avis",

                    "score":
                        "Note",

                    "sentiment":
                        "Sentiment",

                    "sentiment_confidence":
                        "Confiance sentiment",

                    "predicted_score":
                        "Note prédite",

                    "rating_confidence":
                        "Confiance note",

                    "created_at":
                        "Date"
                }

                df = df.rename(
                    columns=rename_columns
                )

                # ------------------------------------------
                # CONFIANCES
                # ------------------------------------------

                if "Confiance sentiment" in df.columns:

                    df[
                        "Confiance sentiment"
                    ] = pd.to_numeric(
                        df[
                            "Confiance sentiment"
                        ],
                        errors="coerce"
                    ) * 100

                    df[
                        "Confiance sentiment"
                    ] = df[
                        "Confiance sentiment"
                    ].map(
                        lambda x:
                        f"{x:.1f}%"
                        if pd.notna(x)
                        else "-"
                    )

                if "Confiance note" in df.columns:

                    df[
                        "Confiance note"
                    ] = pd.to_numeric(
                        df[
                            "Confiance note"
                        ],
                        errors="coerce"
                    ) * 100

                    df[
                        "Confiance note"
                    ] = df[
                        "Confiance note"
                    ].map(
                        lambda x:
                        f"{x:.1f}%"
                        if pd.notna(x)
                        else "-"
                    )

                # ------------------------------------------
                # DATE
                # ------------------------------------------

                if "Date" in df.columns:

                    df["Date"] = pd.to_datetime(
                        df["Date"],
                        errors="coerce"
                    )

                    df["Date"] = df[
                        "Date"
                    ].dt.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )

                # ------------------------------------------
                # AFFICHAGE
                # ------------------------------------------

                st.dataframe(

                    df,

                    use_container_width=True,

                    hide_index=True
                )

            else:

                st.info(
                    "Aucun avis reçu pour le moment."
                )

        else:

            st.error(
                "Impossible de récupérer les avis."
            )

    except requests.exceptions.RequestException:

        st.error(
            "Erreur de connexion à l'API."
        )

# ==========================================================
# PAGE 7 - ANALYSEEN TEMPS REEL
# ==========================================================

def page_analyse_temps_reel():

    st.title("📊 Analyse en temps réel")

    st.caption(
        "Analyse dynamique des avis enregistrés dans SQLite"
    )

    # ==========================================================
    # ACTUALISATION AUTOMATIQUE
    # ==========================================================

    st_autorefresh(
        interval=5000,
        key="analyse_realtime_refresh"
    )

    # ==========================================================
    # RÉCUPÉRATION DES STATISTIQUES
    # ==========================================================

    try:

        response_stats = requests.get(
            f"{API_URL}/api/reviews/stats",
            timeout=5
        )

        if response_stats.status_code != 200:

            st.error(
                "Impossible de récupérer les statistiques."
            )

            return

        stats = response_stats.json()

    except requests.exceptions.RequestException as e:

        st.error(
            f"❌ API FastAPI inaccessible : {e}"
        )

        return

    # ==========================================================
    # KPI
    # ==========================================================

    total = int(
        stats.get("total", 0) or 0
    )

    positif = int(
        stats.get("positif", 0) or 0
    )

    neutre = int(
        stats.get("neutre", 0) or 0
    )

    negatif = int(
        stats.get("negatif", 0) or 0
    )

    moyenne = float(
        stats.get("note_moyenne", 0) or 0
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:

        st.metric(
            "📝 Total avis",
            total
        )

    with col2:

        st.metric(
            "😊 Positifs",
            positif,
            f"{stats.get('pourcentage_positif', 0):.2f}%"
        )

    with col3:

        st.metric(
            "😐 Neutres",
            neutre,
            f"{stats.get('pourcentage_neutre', 0):.2f}%"
        )

    with col4:

        st.metric(
            "😡 Négatifs",
            negatif,
            f"{stats.get('pourcentage_negatif', 0):.2f}%"
        )

    with col5:

        st.metric(
            "⭐ Note moyenne",
            f"{moyenne:.2f}/5"
        )

    st.divider()

    # ==========================================================
    # RÉPARTITION DES SENTIMENTS
    # ==========================================================

    st.subheader(
        "📊 Répartition des sentiments"
    )

    if total > 0:

        taux_positif = (
            positif / total
        ) * 100

        taux_neutre = (
            neutre / total
        ) * 100

        taux_negatif = (
            negatif / total
        ) * 100

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "😊 Positif",
                f"{taux_positif:.2f}%"
            )

        with col2:

            st.metric(
                "😐 Neutre",
                f"{taux_neutre:.2f}%"
            )

        with col3:

            st.metric(
                "😡 Négatif",
                f"{taux_negatif:.2f}%"
            )

        df_sentiments = pd.DataFrame({

            "Sentiment": [
                "Positif",
                "Neutre",
                "Négatif"
            ],

            "Nombre": [
                positif,
                neutre,
                negatif
            ]
        })

        fig = px.pie(

            df_sentiments,

            names="Sentiment",

            values="Nombre",

            hole=0.4,

            title="Répartition des avis"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.info(
            "Aucun avis enregistré pour le moment."
        )

    st.divider()

    # ==========================================================
    # RÉCUPÉRATION DES AVIS
    # ==========================================================

    st.subheader(
        "🕐 Derniers avis reçus"
    )

    try:

        response_reviews = requests.get(

            f"{API_URL}/api/reviews?limit=100",

            timeout=5
        )

        if response_reviews.status_code != 200:

            st.error(
                "Impossible de récupérer les avis."
            )

            return

        data_reviews = response_reviews.json()

        # L'API retourne :
        # {
        #     "count": ...,
        #     "reviews": [...]
        # }

        reviews = data_reviews.get(
            "reviews",
            []
        )

    except requests.exceptions.RequestException as e:

        st.error(
            f"Erreur de connexion avec FastAPI : {e}"
        )

        return

    if not reviews:

        st.info(
            "Aucun avis disponible."
        )

        return

    df = pd.DataFrame(
        reviews
    )

    # ==========================================================
    # DATE
    # ==========================================================

    if "created_at" in df.columns:

        df["created_at"] = pd.to_datetime(
            df["created_at"],
            errors="coerce"
        )

    # ==========================================================
    # ÉVOLUTION DES AVIS
    # ==========================================================

    st.subheader(
        "📈 Évolution des avis reçus"
    )

    if "created_at" in df.columns:

        df_date = df.dropna(
            subset=["created_at"]
        ).copy()

        if not df_date.empty:

            df_date["Date"] = (
                df_date["created_at"]
                .dt.floor("min")
            )

            evolution = (
                df_date
                .groupby("Date")
                .size()
                .reset_index(
                    name="Nombre"
                )
            )

            fig = px.line(

                evolution,

                x="Date",

                y="Nombre",

                markers=True,

                title="Nombre d'avis reçus"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # ==========================================================
    # SENTIMENT DANS LE TEMPS
    # ==========================================================

    st.subheader(
        "😊 Évolution des sentiments"
    )

    if (
        "created_at" in df.columns
        and "sentiment" in df.columns
    ):

        df_sent = df.dropna(
            subset=[
                "created_at",
                "sentiment"
            ]
        ).copy()

        if not df_sent.empty:

            df_sent["Date"] = (
                df_sent["created_at"]
                .dt.floor("min")
            )

            evolution_sent = (
                df_sent
                .groupby(
                    ["Date", "sentiment"]
                )
                .size()
                .reset_index(
                    name="Nombre"
                )
            )

            fig = px.line(

                evolution_sent,

                x="Date",

                y="Nombre",

                color="sentiment",

                markers=True,

                title="Évolution des sentiments"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # ==========================================================
    # PRODUITS
    # ==========================================================

    st.subheader(
        "📦 Produits les plus commentés"
    )

    if "product_id" in df.columns:

        produits = (
            df["product_id"]
            .fillna("Produit inconnu")
            .value_counts()
            .reset_index()
        )

        produits.columns = [
            "Produit",
            "Nombre avis"
        ]

        produits = produits.head(10)

        fig = px.bar(

            produits,

            x="Produit",

            y="Nombre avis",

            text_auto=True,

            title="Top 10 des produits"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ==========================================================
    # NOTES PRÉDITES
    # ==========================================================

    if "predicted_score" in df.columns:

        st.subheader(
            "⭐ Distribution des notes prédites"
        )

        notes = pd.to_numeric(

            df["predicted_score"],

            errors="coerce"

        ).dropna()

        if not notes.empty:

            distribution = (
                notes
                .value_counts()
                .sort_index()
                .reset_index()
            )

            distribution.columns = [
                "Note",
                "Nombre"
            ]

            distribution["Note"] = (
                distribution["Note"]
                .astype(int)
                .astype(str)
            )

            fig = px.bar(

                distribution,

                x="Note",

                y="Nombre",

                text_auto=True,

                title="Distribution des notes prédites"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # ==========================================================
    # TABLEAU
    # ==========================================================

    st.subheader(
        "📋 Historique des avis"
    )

    colonnes = [

        "id",

        "product_id",

        "user_name",

        "text",

        "score",

        "sentiment",

        "sentiment_confidence",

        "predicted_score",

        "rating_confidence",

        "created_at"
    ]

    colonnes_existantes = [

        col

        for col in colonnes

        if col in df.columns
    ]

    df_affichage = df[
        colonnes_existantes
    ].copy()

    st.dataframe(

        df_affichage,

        use_container_width=True,

        hide_index=True
    )
    
# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title(
    "📊 Big Data Reviews"
)

st.sidebar.write(
    "Analyse intelligente des avis clients"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Vue générale",
        "📈 Évolution temporelle",
        "📦 Produits",
        "💬 Thèmes",
        "🤖 IA & Prévision",
        "📡 Avis en temps réel",
        "📊 Analyse en temps réel"
    ]
)


# ==========================================================
# NAVIGATION
# ==========================================================

if page == "🏠 Vue générale":

    page_vue_generale()


elif page == "📈 Évolution temporelle":

    page_evolution()


elif page == "📦 Produits":

    page_produits()


elif page == "💬 Thèmes":

    page_themes()


elif page == "🤖 IA & Prévision":

    page_ia_prevision()


elif page == "📡 Avis en temps réel":

    page_avis_temps_reel()

elif page == "📊 Analyse en temps réel":
    page_analyse_temps_reel()
