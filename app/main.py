from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from app.models import PredictionResponse
from fastapi.templating import Jinja2Templates

from app.embeddings import extract_embedding
from app.similarity import find_top_matches
import numpy as np

app = FastAPI()

# For rendering HTML
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("upload_form.html", {"request": request})

@app.post("/predict-form", response_class=PredictionResponse)
async def predict_form(request: Request, file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        return PredictionResponse(
        estimated_net_worth_CAD=estimated_net_worth,
        top_similar_profiles=top_matches
        )

    try:
        image_bytes = await file.read()
        embedding = extract_embedding(image_bytes)
        estimated_net_worth = round(float(np.linalg.norm(embedding)) * 100_000, 2)
        top_matches = find_top_matches(embedding, top_k=3)

        return templates.TemplateResponse("upload_form.html", {
            "request": request,
            "net_worth": estimated_net_worth,
            "matches": top_matches
        })

    except Exception as e:
        return templates.TemplateResponse("upload_form.html", {
            "request": request,
            "error": str(e)
        })
