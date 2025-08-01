from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import torch
import soundfile as sf
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from huggingface_hub import hf_hub_download
from fastapi.responses import FileResponse
import os

# --- Configuração Inicial do Modelo (carregado apenas uma vez) ---
try:
    print("A carregar modelos de TTS...")
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

    # Caminho para o ficheiro de embeddings local
    speaker_embeddings_local_path = "speaker_embeddings.pt"
    
    # Verifica se o ficheiro local existe. Se não, tenta descarregar.
    if os.path.exists(speaker_embeddings_local_path):
        print(f"A carregar embeddings de locutor a partir do ficheiro local '{speaker_embeddings_local_path}'...")
        speaker_embeddings = torch.load(speaker_embeddings_local_path)
        print("Embeddings de locutor carregados com sucesso a partir do ficheiro local.")
    else:
        print("Ficheiro local não encontrado. A tentar descarregar embeddings de locutor...")
        embeddings_downloaded_path = hf_hub_download(
            repo_id="Matthijs/cmu-arctic-xvectors",
            filename=speaker_embeddings_local_path,
            repo_type="dataset"
        )
        speaker_embeddings = torch.load(embeddings_downloaded_path)
        print("Embeddings de locutor descarregados e carregados com sucesso.")

except Exception as e:
    print(f"Erro ao carregar os modelos de TTS: {e}")
    # Se os modelos não carregarem, a API não pode funcionar.
    processor = None
    model = None
    vocoder = None
    speaker_embeddings = None

router = APIRouter()

class TTSRequest(BaseModel):
    text: str
    speaker_id: int = 0 # Um ID de locutor padrão como exemplo

@router.post("/api/tts")
async def generate_speech(request: TTSRequest):
    # --- CORREÇÃO APLICADA AQUI ---
    # Verificamos explicitamente se cada componente não é None,
    # em vez de usar all(), que causa o erro com o Tensor do PyTorch.
    if processor is None or model is None or vocoder is None or speaker_embeddings is None:
        raise HTTPException(
            status_code=503,
            detail="Os modelos de TTS não estão disponíveis ou não foram carregados corretamente."
        )

    try:
        # 1. Processar o texto de entrada
        inputs = processor(text=request.text, return_tensors="pt")

        # 2. Selecionar a voz a partir dos embeddings pré-carregados
        selected_speaker_embedding = speaker_embeddings[request.speaker_id].unsqueeze(0)

        # 3. Gerar o espetrograma do áudio
        spectrogram = model.generate_speech(
            inputs["input_ids"],
            speaker_embeddings=selected_speaker_embedding
        )

        # 4. Usar o vocoder para converter o espetrograma em áudio (waveform)
        with torch.no_grad():
            speech = vocoder(spectrogram)

        # 5. Definir o caminho do ficheiro de saída
        output_dir = "audio_gerado"
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"speech_{request.speaker_id}.wav")

        # 6. Salvar o áudio num ficheiro .wav
        sf.write(file_path, speech.numpy(), samplerate=16000)

        print(f"Áudio salvo em: {file_path}")

        # 7. Retornar o ficheiro de áudio diretamente na resposta
        return FileResponse(file_path, media_type="audio/wav", filename="speech.wav")

    except IndexError:
        raise HTTPException(
            status_code=400, 
            detail=f"ID de locutor inválido: {request.speaker_id}. Por favor, escolha um ID válido."
        )
    except Exception as e:
        print(f"Ocorreu um erro durante a geração do áudio: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao gerar áudio: {str(e)}")
