from fastapi import FastAPI
from endpoints import health, chat


app = FastAPI()
app.include_router(health.router)
app.include_router(chat.router)
