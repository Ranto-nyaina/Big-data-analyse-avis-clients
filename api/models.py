from typing import Optional

from pydantic import BaseModel, Field


# ==========================================================
# MODELE POUR CREER UN AVIS
# ==========================================================

class ReviewCreate(BaseModel):

    product_id: Optional[str] = "Produit inconnu"

    user_name: Optional[str] = "Utilisateur"

    text: str = Field(
        ...,
        min_length=3,
        description="Texte de l'avis"
    )

    score: Optional[int] = Field(
        default=None,
        ge=1,
        le=5
    )