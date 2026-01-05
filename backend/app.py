from fastapi import FastAPI
from endpoints import health


app = FastAPI()
app.include_router(health.router)
