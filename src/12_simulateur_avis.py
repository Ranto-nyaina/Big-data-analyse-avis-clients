import random
import time
import requests
from collections import Counter


# ==========================================================
# CONFIGURATION
# ==========================================================

API_URL = "http://127.0.0.1:8000"

NOMBRE_AVIS = 30
INTERVALLE = 3


# ==========================================================
# PRODUITS
# ==========================================================

PRODUITS = [
    "B007JFMH8M",
    "B002QWHJOU",
    "B002QWP89S",
    "B0026RQTGE",
    "B002QWP8H0",
    "B003B3OOPA",
    "B001EO5Q64",
    "B000VK8AVK",
    "B000G6RYNE",
    "B000F4U2XK",
]


# ==========================================================
# UTILISATEURS
# ==========================================================

UTILISATEURS = [
    "Utilisateur_001",
    "Utilisateur_002",
    "Utilisateur_003",
    "Utilisateur_004",
    "Utilisateur_005",
    "Utilisateur_006",
    "Utilisateur_007",
    "Utilisateur_008",
    "Utilisateur_009",
    "Utilisateur_010",
]


# ==========================================================
# AVIS POSITIFS
# ==========================================================

AVIS_POSITIFS = [
    "This product is excellent and I really love it.",
    "Great product with a very good taste.",
    "I love this product and I will buy it again.",
    "The quality is excellent and the flavor is great.",
    "Very good product, highly recommended.",
    "This is one of the best products I have tried.",
    "Excellent quality and great flavor.",
    "I am very satisfied with this purchase.",
    "The taste is amazing and the product is fresh.",
    "Really good product and very good value.",
    "I enjoyed this product very much.",
    "Great taste and excellent quality.",
    "This product exceeded my expectations.",
    "I will definitely purchase this again.",
    "Very tasty and high quality product.",
    "I really like this product.",
    "The flavor is wonderful and the quality is excellent.",
    "One of my favorite products.",
    "Very happy with my purchase.",
    "Excellent product, good taste and good quality.",
]


# ==========================================================
# AVIS NEUTRES
# ==========================================================

AVIS_NEUTRES = [
    "The product is okay but nothing special.",
    "It is an average product with an acceptable taste.",
    "The quality is fine but I expected a little more.",
    "This product is acceptable but not exceptional.",
    "The taste is okay and the quality is average.",
    "It is a decent product but there are better options.",
    "The product works fine but I am not impressed.",
    "Average quality and average flavor.",
    "The product is neither bad nor excellent.",
    "It is okay for the price.",
    "The taste is acceptable but could be better.",
    "I have mixed feelings about this product.",
    "The product is fine but I probably will not buy it again.",
    "Nothing particularly good or bad about this product.",
    "The quality is reasonable for the price.",
    "It is an average product with an ordinary taste.",
    "The product is acceptable but not one of my favorites.",
    "The flavor is okay but not impressive.",
    "It is a decent product, but nothing special.",
    "The product meets my expectations but does not exceed them.",
]


# ==========================================================
# AVIS NEGATIFS
# ==========================================================

AVIS_NEGATIFS = [
    "This product is terrible and I am very disappointed.",
    "The taste is bad and the quality is poor.",
    "I really dislike this product.",
    "Very poor quality and unpleasant taste.",
    "This is one of the worst products I have tried.",
    "I am very disappointed with this purchase.",
    "The product tastes terrible.",
    "The quality is very bad.",
    "I would not recommend this product.",
    "The flavor is unpleasant and the product is poor.",
    "This product was a waste of money.",
    "Very bad taste and poor quality.",
    "I expected much better from this product.",
    "I will not buy this product again.",
    "The product is disappointing and not worth the price.",
    "I really did not like the taste.",
    "Poor product with a very unpleasant flavor.",
    "This product did not meet my expectations.",
    "The quality is disappointing.",
    "I regret buying this product.",
]


# ==========================================================
# COMPTEURS
# ==========================================================

avis_generes = Counter()

predictions_ia = Counter()


# ==========================================================
# GENERER UN AVIS
# ==========================================================

def generer_avis():

    categorie = random.choices(
        ["positif", "neutre", "negatif"],
        weights=[40, 30, 30],
        k=1
    )[0]

    if categorie == "positif":

        texte = random.choice(AVIS_POSITIFS)
        score = random.choice([4, 5])

    elif categorie == "neutre":

        texte = random.choice(AVIS_NEUTRES)
        score = 3

    else:

        texte = random.choice(AVIS_NEGATIFS)
        score = random.choice([1, 2])

    produit = random.choice(PRODUITS)
    utilisateur = random.choice(UTILISATEURS)

    avis_generes[categorie] += 1

    return {
        "product_id": produit,
        "user_name": utilisateur,
        "text": texte,
        "score": score,
        "categorie_attendue": categorie,
    }


# ==========================================================
# ENVOYER L'AVIS À L'API
# ==========================================================

def envoyer_avis(avis):

    url = f"{API_URL}/api/reviews"

    payload = {
        "product_id": avis["product_id"],
        "user_name": avis["user_name"],
        "text": avis["text"],
        "score": avis["score"],
    }

    try:

        response = requests.post(
            url,
            json=payload,
            timeout=10
        )

        if response.status_code == 200:

            return response.json()

        else:

            print(
                f"❌ Erreur API : "
                f"{response.status_code} - {response.text}"
            )

            return None

    except requests.exceptions.RequestException as e:

        print(f"❌ Impossible de contacter l'API : {e}")

        return None


# ==========================================================
# AFFICHER LE RESULTAT
# ==========================================================

def afficher_resultat(avis, resultat):

    if resultat is None:
        return

    sentiment = resultat.get(
        "sentiment",
        "inconnu"
    )

    sentiment_confidence = resultat.get(
        "sentiment_confidence",
        0
    )

    predicted_score = resultat.get(
        "predicted_score",
        0
    )

    rating_confidence = resultat.get(
        "rating_confidence",
        0
    )

    predictions_ia[sentiment] += 1

    print()
    print("=" * 70)

    print(
        f"👤 Utilisateur : {avis['user_name']}"
    )

    print(
        f"📦 Produit     : {avis['product_id']}"
    )

    print(
        f"📝 Avis        : {avis['text']}"
    )

    print(
        f"⭐ Note envoyée : {avis['score']}/5"
    )

    print(
        f"🎯 Catégorie attendue : {avis['categorie_attendue']}"
    )

    print()

    print(
        f"🤖 Sentiment IA : {sentiment}"
    )

    print(
        f"📊 Confiance sentiment : "
        f"{float(sentiment_confidence):.2f}%"
    )

    print(
        f"⭐ Note prédite par IA : "
        f"{predicted_score}/5"
    )

    print(
        f"📊 Confiance note : "
        f"{float(rating_confidence):.2f}%"
    )

    print("=" * 70)


# ==========================================================
# VERIFIER L'API
# ==========================================================

def verifier_api():

    try:

        response = requests.get(
            f"{API_URL}/api/health",
            timeout=5
        )

        if response.status_code == 200:

            data = response.json()

            print()
            print("✅ API accessible")
            print(
                f"🤖 Modèles chargés : "
                f"{data.get('models_loaded')}"
            )

            return True

        print(
            f"❌ API inaccessible : "
            f"{response.status_code}"
        )

        return False

    except requests.exceptions.RequestException as e:

        print("❌ Impossible de contacter l'API.")
        print(f"Erreur : {e}")

        return False


# ==========================================================
# AFFICHER LE RESUME
# ==========================================================

def afficher_resume():

    print()
    print()
    print("=" * 70)
    print("📊 RÉSUMÉ DE LA SIMULATION")
    print("=" * 70)

    print()
    print("📝 Avis générés :")

    print(
        f"   🟢 Positifs : "
        f"{avis_generes['positif']}"
    )

    print(
        f"   🟡 Neutres  : "
        f"{avis_generes['neutre']}"
    )

    print(
        f"   🔴 Négatifs : "
        f"{avis_generes['negatif']}"
    )

    print()

    print("🤖 Prédictions du modèle IA :")

    print(
        f"   🟢 Positifs : "
        f"{predictions_ia['positif']}"
    )

    print(
        f"   🟡 Neutres  : "
        f"{predictions_ia['neutre']}"
    )

    print(
        f"   🔴 Négatifs : "
        f"{predictions_ia['negatif']}"
    )

    print()

    print(
        "💡 Les catégories 'Avis générés' correspondent "
        "à la catégorie utilisée pour construire le texte."
    )

    print(
        "💡 Les catégories 'Prédictions IA' correspondent "
        "au résultat réel du modèle."
    )

    print("=" * 70)


# ==========================================================
# LANCER LE SIMULATEUR
# ==========================================================

def lancer_simulateur():

    print()
    print("=" * 70)
    print("🚀 SIMULATEUR D'AVIS CLIENTS")
    print("=" * 70)

    print()
    print(f"Nombre d'avis : {NOMBRE_AVIS}")
    print(f"Intervalle    : {INTERVALLE} secondes")
    print(f"API           : {API_URL}")

    print()

    if not verifier_api():

        print()
        print(
            "❌ Arrêt du simulateur : "
            "l'API n'est pas disponible."
        )

        return

    print()
    print("▶️ Début de la simulation...")
    print()

    try:

        for i in range(1, NOMBRE_AVIS + 1):

            print()
            print(
                f"🔄 Avis {i}/{NOMBRE_AVIS}"
            )

            avis = generer_avis()

            resultat = envoyer_avis(avis)

            afficher_resultat(
                avis,
                resultat
            )

            if i < NOMBRE_AVIS:

                print(
                    f"⏳ Attente de "
                    f"{INTERVALLE} secondes..."
                )

                time.sleep(INTERVALLE)

    except KeyboardInterrupt:

        print()
        print()
        print("🛑 Simulation interrompue par l'utilisateur.")

    finally:

        afficher_resume()


# ==========================================================
# PROGRAMME PRINCIPAL
# ==========================================================

if __name__ == "__main__":

    lancer_simulateur()