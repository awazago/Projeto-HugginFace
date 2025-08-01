from fastapi import FastAPI
#from fastapi.staticfiles import StaticFiles
from api.chatCompletionsRouter import router as api_chat_router
from api.imageRouter import router as api_image_router
from api.moderationsRouter import router as api_moderation_router
from api.textToSpeechRouter import router as api_tts_router

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Server is running"}

#app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(api_chat_router)
app.include_router(api_image_router)
app.include_router(api_moderation_router)
app.include_router(api_tts_router)

