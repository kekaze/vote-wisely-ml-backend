from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from sentence_transformers import SentenceTransformer

app = FastAPI()
class EmbeddingRequest(BaseModel):
    criteria: str

model = SentenceTransformer('all-MiniLM-L6-v2')

@app.post(path="/embed")
def generate_embedding(data: EmbeddingRequest):
    try:
        embedding = model.encode(data.criteria).tolist()
        return {"embedding": embedding}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))