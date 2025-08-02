# --- Parte 1: Importações de Bibliotecas ---
# Estas são as ferramentas que o nosso código precisa para funcionar.

# APIRouter permite-nos organizar este endpoint num ficheiro separado.
from fastapi import APIRouter, HTTPException
# Pydantic é usado para validar os dados que a nossa API recebe.
from pydantic import BaseModel
# 'os' é usado para aceder a variáveis de ambiente do sistema.
import os
# 'requests' é a biblioteca que usamos para fazer pedidos HTTP para a API do Hugging Face.
import requests
# 'dotenv' é usado para carregar variáveis de ambiente a partir de um ficheiro .env.
from dotenv import load_dotenv

# --- Parte 2: Configuração Inicial ---
# Este bloco é executado apenas uma vez, quando a aplicação arranca.

# Carrega as variáveis definidas no seu ficheiro .env para o ambiente.
load_dotenv()

# Obtém o token de autorização do Hugging Face a partir das variáveis de ambiente.
HF_TOKEN = os.getenv("HF_TOKEN")

# Verificação de segurança: se o token não for encontrado, a aplicação para com um erro.
# Isto evita que a aplicação arranque sem a autenticação necessária.
if not HF_TOKEN:
    raise ValueError("A variável de ambiente HF_TOKEN não está definida. Certifique-se de criá-la no seu arquivo .env")

# O URL do endpoint de inferência para o modelo de análise de sentimento que queremos usar.
# Este modelo em particular classifica o texto em 1 a 5 estrelas.
URL = "https://api-inference.huggingface.co/models/nlptown/bert-base-multilingual-uncased-sentiment" # URL corrigido para o endpoint de inferência padrão
# O cabeçalho de autorização que será enviado com cada pedido para a API do Hugging Face.
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# --- Parte 3: Definição da Rota da API ---

# Cria uma instância do APIRouter para definir os endpoints deste ficheiro.
router = APIRouter()

# Define a estrutura dos dados que esperamos receber no corpo do pedido.
# A nossa API espera um JSON com um campo "text".
# Nota: O nome da classe foi corrigido de "ModerationRquest" para "ModerationRequest".
class ModerationRequest(BaseModel):
    text: str

# Decorador que define este endpoint. Ele responde a pedidos HTTP POST para /api/moderations.
@router.post("/api/moderations")
async def moderate_text(request: ModerationRequest):
    # --- Início da Lógica do Endpoint (executada a cada pedido) ---

    # Cria o payload (corpo da requisição) no formato que a API do Hugging Face espera.
    # O modelo espera um campo "inputs" com o texto a ser analisado.
    payload = {
        "inputs": request.text
    }

    try:
        # Envia o pedido POST para a API do Hugging Face, incluindo os cabeçalhos e o payload.
        response = requests.post(URL, headers=headers, json=payload)
        
        # Verifica se a resposta da API foi um erro (ex: 4xx ou 5xx).
        # Se for um erro, esta linha irá levantar uma exceção que será capturada pelo FastAPI.
        response.raise_for_status()

        # Se o pedido foi bem-sucedido, retorna a resposta JSON da API do Hugging Face
        # diretamente para o cliente que chamou a nossa API.
        # A resposta será algo como: [[{'label': '5 stars', 'score': 0.8}, ...]]
        return response.json()
        
    except requests.exceptions.RequestException as e:
        # Se houver um erro de rede ou um erro da API do Hugging Face,
        # capturamos a exceção e retornamos um erro 500 mais informativo.
        raise HTTPException(status_code=500, detail=f"Erro ao comunicar com a API de moderação: {str(e)}")

