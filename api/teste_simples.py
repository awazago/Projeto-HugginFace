import os
import requests
from dotenv import load_dotenv

print("--- Iniciando teste simples ---")

# Carrega as variáveis do arquivo .env
load_dotenv()
STABILITY_KEY = os.getenv("STABILITY_KEY", "").strip()

if not STABILITY_KEY:
    print("ERRO CRÍTICO: Não foi possível encontrar a IMG_KEY no seu arquivo .env")
else:
    print(f"Chave de API encontrada. Começa com: '{STABILITY_KEY[:5]}...'")

    # Configuração da requisição
    URL = "https://api.stability.ai/v2beta/stable-image/generate/ultra"
    headers = {
      "Authorization": f"Bearer {STABILITY_KEY}",
      "Accept": "image/*"
    }
    payload = {
        "prompt": "A beautiful cat",
        "aspect_ratio": "1:1",
    }

    print("Enviando requisição para a Stability AI...")

    try:
        # Faz a chamada de API
        response = requests.post(
            URL,
            headers=headers,
            files={"none": ''},
            data=payload,
            timeout=60
        )
        response.raise_for_status()

        print("\n--- SUCESSO! ---")
        print("A requisição foi aceite e a imagem foi gerada.")
        print(f"Status da resposta: {response.status_code}")

    except requests.exceptions.HTTPError as e:
        print("\n--- FALHA NA REQUISIÇÃO ---")
        print(f"A API retornou um erro: {e.response.status_code}")
        print(f"Detalhes do erro: {e.response.text}")
    except Exception as e:
        print(f"\n--- Ocorreu um erro inesperado ---")
        print(e)