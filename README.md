Projeto de APIs de IA com FastAPIEste projeto consiste num back-end robusto construído com FastAPI que serve como um gateway para diversos modelos de Inteligência Artificial de última geração, oferecendo quatro endpoints principais para tarefas de IA.✨ FuncionalidadesO servidor expõe quatro APIs distintas:Text-to-Speech (TTS): Converte texto em áudio falado utilizando o modelo microsoft/speecht5_tts.Moderação de Texto: Analisa o sentimento de um texto, classificando-o de 1 a 5 estrelas, utilizando o modelo nlptown/bert-base-multilingual-uncased-sentiment.Chat Completions: Gera respostas de texto coerentes e contextuais a partir de um prompt, utilizando o modelo meta-llama/Meta-Llama-3.1-8B-Instruct.Text-to-Image: Cria imagens a partir de uma descrição em texto, utilizando o modelo stable-image/generate/ultra da Stability AI.🛠️ Tecnologias UtilizadasBack-end: Python 3.12, FastAPIServidor ASGI: UvicornModelos de IA: Hugging Face Transformers, Stability AIBibliotecas Principais: httpx, requests, torch, soundfile, python-dotenvGestão de Ambiente: venvGestão de Pacotes: pip, Homebrew (para macOS)🚀 Instalação e ConfiguraçãoSiga os passos abaixo para configurar e executar o projeto localmente.Pré-requisitosPython 3.12 (recomenda-se a instalação via Homebrew no macOS)GitPassosClone o repositório:git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio
Crie e ative o ambiente virtual:# Criar o ambiente com a versão correta do Python
python3.12 -m venv venv

# Ativar o ambiente (macOS/Linux)
source venv/bin/activate
Instale as dependências:O ficheiro requirements.txt contém todas as bibliotecas necessárias.pip install -r requirements.txt
Configure as variáveis de ambiente:Crie um ficheiro chamado .env na raiz do projeto e adicione as suas chaves de API.# Token do Hugging Face (para Chat e Moderação)
HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Chave da Stability AI (para geração de imagem)
IMG_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
Descarregue os Embeddings de Locutor (para TTS):O modelo de TTS precisa de um ficheiro com as "vozes". Execute o seguinte comando no terminal, na raiz do projeto, para o descarregar:curl -L -o speaker_embeddings.pt "https://huggingface.co/datasets/Matthijs/cmu-arctic-xvectors/resolve/main/speaker_embeddings.pt"
Nota: O ficheiro tem aproximadamente 750 MB.▶️ Como ExecutarCom o ambiente virtual ativado (venv), inicie o servidor FastAPI com o Uvicorn:python3 -m uvicorn main:app --reload
O servidor estará disponível em http://127.0.0.1:8000. A documentação interativa da API (Swagger UI) estará disponível em http://127.0.0.1:8000/docs.⚙️ Uso da API (Endpoints)1. Text-to-SpeechEndpoint: POST /api/ttsDescrição: Gera um ficheiro de áudio a partir de um texto.Corpo da Requisição (JSON):{
  "text": "Olá, mundo! Isto é um teste.",
  "speaker_id": 0
}
O speaker_id é opcional e pode ser um número inteiro para selecionar vozes diferentes.Resposta de Sucesso: Retorna o ficheiro speech.wav.2. Moderação de TextoEndpoint: POST /api/moderationsDescrição: Analisa o sentimento do texto.Corpo da Requisição (JSON):{
  "text": "Eu adoro este produto, é fantástico!"
}
Resposta de Sucesso (JSON):[[
  {"label": "1 star", "score": 0.002},
  {"label": "2 stars", "score": 0.003},
  {"label": "3 stars", "score": 0.01},
  {"label": "4 stars", "score": 0.1},
  {"label": "5 stars", "score": 0.885}
]]
3. Chat CompletionsEndpoint: POST /api/chat_completionDescrição: Gera uma resposta de chat.Corpo da Requisição (JSON):{
  "prompt": "Explique o que é a computação quântica em termos simples.",
  "max_tokens": 150,
  "temperature": 0.8
}
Resposta de Sucesso (JSON):{
  "response": "A computação quântica é um tipo de computação que usa os princípios da mecânica quântica para realizar cálculos..."
}
4. Text-to-ImageEndpoint: POST /api/imageDescrição: Gera uma imagem e salva-a na pasta imagens_geradas.Corpo da Requisição (JSON):{
  "prompt": "Um astronauta a andar a cavalo na lua, arte digital",
  "aspect_ratio": "16:9"
}
Resposta de Sucesso (JSON):{
  "message": "Imagem gerada e salva com sucesso!",
  "file_path": "imagens_geradas/gerado_123456789.webp"
}
