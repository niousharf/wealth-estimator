from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.embeddings import extract_embedding
from app.similarity import find_top_matches
import numpy as np

app = FastAPI()

templates = Jinja2Templates(directory="app/templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "upload_form.html")


@app.post("/predict-form", response_class=HTMLResponse)
async def predict_form(request: Request, file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        return templates.TemplateResponse(request, "upload_form.html", {
            "error": "Invalid file type. Please upload an image."
        })

    try:
        image_bytes = await file.read()
        embedding = extract_embedding(image_bytes)

        estimated_net_worth = round(float(np.linalg.norm(embedding)) * 100_000, 2)
        top_matches = find_top_matches(embedding, top_k=3)

        return templates.TemplateResponse(request, "upload_form.html", {
            "net_worth": estimated_net_worth,
            "matches": top_matches
        })

    except Exception as e:
        return templates.TemplateResponse(request, "upload_form.html", {
            "error": f"Error during processing: {str(e)}"
        })
