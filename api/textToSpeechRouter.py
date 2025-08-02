# --- Parte 1: Importações de Bibliotecas ---
# Estas são as ferramentas que o nosso código precisa para funcionar.

# Ferramentas do FastAPI para criar a API e lidar com erros.
from fastapi import APIRouter, HTTPException
# Pydantic é usado para validar os dados que a nossa API recebe (ex: o texto).
from pydantic import BaseModel
# PyTorch é a biblioteca de machine learning usada pelos modelos de TTS.
import torch
# Soundfile é uma biblioteca para ler e escrever ficheiros de áudio. Usamo-la para salvar o .wav.
import soundfile as sf
# Componentes específicos do modelo SpeechT5 da biblioteca Transformers do Hugging Face.
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
# Função para descarregar ficheiros de forma segura do Hugging Face Hub.
from huggingface_hub import hf_hub_download
# FileResponse permite que a nossa API envie um ficheiro (o áudio .wav) como resposta.
from fastapi.responses import FileResponse
# 'os' é uma biblioteca padrão do Python para interagir com o sistema operativo (ex: criar pastas).
import os

# --- Parte 2: Configuração e Carregamento dos Modelos ---
# Este bloco é executado apenas UMA VEZ, quando a aplicação arranca.
# Carregar os modelos na memória antecipadamente torna a API muito mais rápida,
# pois não precisamos de os carregar a cada pedido.

try:
    print("A carregar modelos de TTS...")
    # 1. SpeechT5Processor: Prepara o texto para o modelo, convertendo-o em números (tokens).
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    # 2. SpeechT5ForTextToSpeech: O "cérebro" principal que transforma o texto processado num espetrograma (uma representação visual do som).
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    # 3. SpeechT5HifiGan (Vocoder): Um modelo especializado que pega no espetrograma e o converte num ficheiro de áudio de alta qualidade (waveform).
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

    # Define o nome do ficheiro local que contém as diferentes "vozes".
    speaker_embeddings_local_path = "speaker_embeddings.pt"
    
    # Verifica se já temos o ficheiro de vozes localmente.
    if os.path.exists(speaker_embeddings_local_path):
        print(f"A carregar embeddings de locutor a partir do ficheiro local '{speaker_embeddings_local_path}'...")
        # Se o ficheiro existe, carrega-o diretamente para a memória.
        speaker_embeddings = torch.load(speaker_embeddings_local_path)
        print("Embeddings de locutor carregados com sucesso a partir do ficheiro local.")
    else:
        # Se o ficheiro não existe, tenta descarregá-lo do Hugging Face.
        print("Ficheiro local não encontrado. A tentar descarregar embeddings de locutor...")
        embeddings_downloaded_path = hf_hub_download(
            repo_id="Matthijs/cmu-arctic-xvectors",
            filename=speaker_embeddings_local_path,
            repo_type="dataset"
        )
        # Carrega o ficheiro que acabou de ser descarregado.
        speaker_embeddings = torch.load(embeddings_downloaded_path)
        print("Embeddings de locutor descarregados e carregados com sucesso.")
    
    # Conta quantas vozes diferentes estão disponíveis no ficheiro.
    num_speakers = len(speaker_embeddings)
    print(f"Número total de locutores disponíveis: {num_speakers}")

except Exception as e:
    # Se qualquer passo acima falhar (ex: falta de internet), a aplicação não "quebra".
    # Em vez disso, informamos o erro e definimos as variáveis dos modelos como None.
    print(f"Erro ao carregar os modelos de TTS: {e}")
    processor = None
    model = None
    vocoder = None
    speaker_embeddings = None
    num_speakers = 0

# --- Parte 3: Definição da Rota da API ---

# APIRouter permite-nos organizar os nossos endpoints num ficheiro separado (este)
# e depois importá-lo no nosso ficheiro principal (main.py).
router = APIRouter()

# Usamos o Pydantic para definir a estrutura dos dados que esperamos receber no pedido.
# A nossa API espera um JSON com um campo "text" (obrigatório) e um "speaker_id" (opcional).
class TTSRequest(BaseModel):
    text: str
    speaker_id: int = 0 # Define 0 como o ID de locutor padrão se nenhum for fornecido.

# Este é o "decorador" que transforma a nossa função Python num endpoint de API.
# Ele diz ao FastAPI: "Qualquer pedido HTTP POST para o endereço /api/tts deve executar esta função".
@router.post("/api/tts")
async def generate_speech(request: TTSRequest):
    # --- Início da Lógica do Endpoint (executada a cada pedido) ---

    # Primeira verificação: os modelos foram carregados corretamente no arranque?
    if processor is None or model is None or vocoder is None or speaker_embeddings is None:
        # Se não, retorna um erro 503 (Serviço Indisponível).
        raise HTTPException(
            status_code=503,
            detail="Os modelos de TTS não estão disponíveis ou não foram carregados corretamente."
        )

    # Segunda verificação: o ID de locutor pedido pelo utilizador é válido?
    if not (0 <= request.speaker_id < num_speakers):
        # Se não, retorna um erro 400 (Pedido Inválido) com uma mensagem útil.
        raise HTTPException(
            status_code=400,
            detail=f"ID de locutor inválido: {request.speaker_id}. Por favor, escolha um ID entre 0 e {num_speakers - 1}."
        )

    # Bloco principal para gerar o áudio.
    try:
        # 1. Processar o texto de entrada usando o processador que carregámos.
        inputs = processor(text=request.text, return_tensors="pt")

        # 2. Selecionar a voz desejada do ficheiro de embeddings.
        #    `speaker_embeddings` é uma lista de tensores, e nós escolhemos um com base no ID.
        selected_speaker_embedding = speaker_embeddings[request.speaker_id].unsqueeze(0)

        # 3. Gerar o espetrograma do áudio a partir do texto e da voz escolhida.
        spectrogram = model.generate_speech(
            inputs["input_ids"],
            speaker_embeddings=selected_speaker_embedding
        )

        # 4. Usar o vocoder para transformar o espetrograma em áudio de alta qualidade.
        #    `torch.no_grad()` é uma otimização que diz ao PyTorch para não calcular gradientes,
        #    tornando este passo mais rápido.
        with torch.no_grad():
            speech = vocoder(spectrogram)

        # 5. Criar uma pasta para guardar os ficheiros de áudio, se ainda não existir.
        output_dir = "audio_gerado"
        os.makedirs(output_dir, exist_ok=True)
        # Definir o nome e o caminho completo do ficheiro de saída.
        file_path = os.path.join(output_dir, f"speech_{request.speaker_id}.wav")

        # 6. Salvar o áudio gerado (que está em formato numpy array) num ficheiro .wav.
        #    A taxa de amostragem de 16000 é um requisito do modelo.
        sf.write(file_path, speech.numpy(), samplerate=16000)

        print(f"Áudio salvo em: {file_path}")

        # 7. Retornar o ficheiro de áudio como resposta ao pedido.
        #    O navegador irá descarregar este ficheiro ou tocá-lo diretamente.
        return FileResponse(file_path, media_type="audio/wav", filename="speech.wav")

    except Exception as e:
        # Se qualquer passo dentro do bloco 'try' falhar, capturamos o erro
        # e retornamos um erro 500 (Erro Interno do Servidor) com os detalhes.
        print(f"Ocorreu um erro durante a geração do áudio: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao gerar áudio: {str(e)}")
