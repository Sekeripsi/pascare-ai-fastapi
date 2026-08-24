from fastapi import FastAPI

from app.config import settings
from app.routes import chat

app = FastAPI(title="Postvisit Chatbot API", version="0.1.0")

app.include_router(chat.router)


@app.get("/")
def read_root():
    return {"status": "ok", "service": "postvisit-chatbot", "model": settings.groq_model}
