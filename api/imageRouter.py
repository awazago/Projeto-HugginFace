from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import requests
import base64
from dotenv import load_dotenv
from uuid import uuid4

load_dotenv()
STABILITY_KEY = os.getenv("STABILITY_KEY", "").strip()

if not STABILITY_KEY:
    raise ValueError("A variável de ambiente STABILITY_KEY não está definida. Certifique-se de criá-la no seu arquivo .env")
else:
    print(f"Chave de API encontrada. Começa com: '{STABILITY_KEY[:5]}...'")

URL = "https://api.stability.ai/v2beta/stable-image/generate/ultra"
#URL = "https://api.stability.ai/v2beta/stable-image/generate/core"

router = APIRouter()

class ImageRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    aspect_ratio: str = "1:1"
    seed: int = 0
    output_format: str = "png"

@router.post("/api/image")
async def generate_image(request: ImageRequest):
    headers = {
      "Authorization": f"Bearer {STABILITY_KEY}",
      "Accept": "image/*"
    }
    
    payload = {
        "prompt": request.prompt,
        "negative_prompt": request.negative_prompt,
        "aspect_ratio": request.aspect_ratio,
        "seed": str(request.seed), # A API espera o seed como string
        "output_format": request.output_format
    }
    try:
        response = requests.post(
          URL, 
          headers=headers, 
          files={"none": ''},
          data=payload, 
          timeout=180
        )
        
        response.raise_for_status()

        # 1. Obter metadados dos cabeçalhos da resposta
        finish_reason = response.headers.get("finish-reason")
        seed = response.headers.get("seed")
        #base64_image = base64.b64encode(response.content).decode('utf-8')

        # 2. Verificar se a imagem foi filtrada por conteúdo
        if finish_reason == 'CONTENT_FILTERED':
            raise HTTPException(
                status_code=400,
                detail="A geração falhou porque a imagem foi classificada como NSFW."
            )

        # 3. Definir o diretório de saída e criá-lo se não existir
        output_dir = "imagens_geradas"
        os.makedirs(output_dir, exist_ok=True)

        # 4. Criar um nome de ficheiro único e o caminho completo
        filename = f"gerado_{seed}.{request.output_format}"
        file_path = os.path.join(output_dir, filename)

        # 5. Salvar o conteúdo da imagem no ficheiro
        with open(file_path, "wb") as f:
            f.write(response.content)

        print(f"Imagem salva em: {file_path}")

        # 6. Retornar uma mensagem de sucesso com o caminho do ficheiro
        return {
            "message": "Imagem gerada e salva com sucesso!",
            "file_path": file_path
        }

        #return {
        #    "message": "Imagem gerada com sucesso pelo modelo Ultra",
        #    "image_data": f"data:image/webp;base64,{base64_image}"
        #}
        
    except requests.exceptions.HTTPError as e:
        print(f"Resposta de erro da API: {e.response.text}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Erro da API da Stability AI: {e.response.text}"
        )
    except requests.exceptions.HTTPError as e:
        raise HTTPException(status_code=503, detail=f"Serviço indisponível: {e}")
    

