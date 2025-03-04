import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client

app = FastAPI()

class EmbeddingRequest(BaseModel):
    criteria: str
class Settings(BaseSettings):
    supabase_url: str
    supabase_key: str
    service_key: str

    class Config:
        env_file = ".env"

settings = Settings()

model = SentenceTransformer('all-MiniLM-L6-v2')

url: str = settings.supabase_url
key: str = settings.supabase_key
supabase: Client = create_client(url, key)

@app.post(path="/embed")
def generate_embedding(data: EmbeddingRequest):
    try:
        embedding = model.encode(data.criteria).tolist()
        return {"embedding": embedding}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))