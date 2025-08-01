from fastapi import APIRouter
from pydantic   import BaseModel
import os
import requests
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("A variável de ambiente HF_TOKEN não está definida. Certifique-se de criá-la no seu arquivo .env")

URL = "https://router.huggingface.co/hf-inference/models/nlptown/bert-base-multilingual-uncased-sentiment"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

router = APIRouter()

class ModerationRquest(BaseModel):
    text: str

@router.post("/api/moderations")
async def moderate_text(request: ModerationRquest):
    payload = {
        "inputs": request.text
    }
    response = requests.post(URL, headers=headers, json=payload)
    response.raise_for_status()

    return response.json()