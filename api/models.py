from typing import Optional

from pydantic import BaseModel, Field


# ==========================================================
# MODELE POUR CREER UN AVIS
# ==========================================================

class ReviewCreate(BaseModel):

    product_id: Optional[str] = Field(
        default="Produit inconnu",
        max_length=255
    )

    user_name: Optional[str] = Field(
        default="Utilisateur",
        max_length=255
    )

    text: str = Field(
        ...,
        min_length=3,
        max_length=5000,
        description="Texte de l'avis"
    )

    score: Optional[int] = Field(
        default=None,
        ge=1,
        le=5
    )