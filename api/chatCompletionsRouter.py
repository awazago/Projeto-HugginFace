from fastapi import APIRouter, APIRouter, HTTPException, status
from pydantic import BaseModel
import os
#import requests
import httpx
from dotenv import load_dotenv

#carregar variavel ambiente
load_dotenv()
#carreguando o token HuggingFace
HF_TOKEN = os.getenv("HF_TOKEN")

# Verificar se o token foi carregado
if not HF_TOKEN:
    # É bom levantar um erro aqui para que o servidor não inicie sem o token
    raise ValueError("A variável de ambiente HF_TOKEN não está definida. Certifique-se de criá-la no seu arquivo .env")

URL = "https://router.huggingface.co/v1/chat/completions"

headers = {
  "Authorization": "Bearer " + HF_TOKEN,
  "Content-Type": "application/json"
}

router = APIRouter()

class ChatRequest (BaseModel):
    prompt: str
    max_tokens: int = 100 # Mudança de max_length para max_tokens
    temperature: float = 0.7 # Adicionado para controle

@router.post("/api/chat_completion", status_code=status.HTTP_200_OK)
async def chat_completion(request: ChatRequest):
    payload = {
        "messages": [
        {
            "role": "user",
            "content": request.prompt
        }
    ],
    "model": "meta-llama/Llama-3.2-3B-Instruct:novita",
    "max_tokens": request.max_tokens,
    "temperature": request.temperature
    }

    #inicio no caso de usar httpx
    # Você precisa criar uma instância de httpx.AsyncClient para fazer a requisição.
    #Ao contrario do requests o httpx funciona melhor com requsições concorrentes
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(URL, headers=headers, json=payload, timeout=30.0)
            response.raise_for_status() # Levanta um erro HTTP para status 4xx/5xx

            # Processa a resposta do Hugging Face Router
            hf_response_json = response.json()
            if hf_response_json and hf_response_json.get("choices"):
                # Retorna apenas o conteúdo da primeira mensagem do assistente
                return {"response": hf_response_json["choices"][0]["message"]["content"]}
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Resposta inesperada do Hugging Face Router: 'choices' ou 'message' não encontrado."
                )

        except httpx.RequestError as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Erro de conexão com o Hugging Face Router: {e}"
            )
        except httpx.HTTPStatusError as e:
            error_detail = e.response.json().get("detail", str(e)) if e.response.content else str(e)
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Erro da API do Hugging Face Router ({e.response.status_code}): {error_detail}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Erro interno no servidor: {str(e)}"
            )
    #fim httpx

    #Apenas no caso de usar requests
    #response = requests.post(URL, headers=headers, json=payload)
    #response.raise_for_status()
    #return response.json()

