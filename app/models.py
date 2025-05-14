from pydantic import BaseModel, Field
from typing import List


class Match(BaseModel):
    """
    A similar wealthy profile matched based on embedding similarity.
    """

    name: str = Field(..., description="Name of the matched individual")
    net_worth: str = Field(
        ..., description="Net worth of the matched individual (e.g., '100B')"
    )
    similarity_score: float = Field(
        ..., ge=0.0, le=1.0, description="Cosine similarity score (0.0 to 1.0)"
    )


class PredictionResponse(BaseModel):
    """
    The response returned by the wealth estimator API.
    """

    estimated_net_worth_CAD: float = Field(
        ..., ge=0, description="Estimated net worth in Canadian Dollars"
    )
    top_similar_profiles: List[Match] = Field(
        ..., description="Top N most similar wealthy profiles"
    )
