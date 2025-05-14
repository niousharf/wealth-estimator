# app/models.py
from pydantic import BaseModel
from typing import List

class Match(BaseModel):
    name: str
    net_worth: str
    similarity_score: float

class PredictionResponse(BaseModel):
    estimated_net_worth_CAD: float
    top_similar_profiles: List[Match]
