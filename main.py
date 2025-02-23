from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
app = FastAPI()

model = SentenceTransformer('all-MiniLM-L6-v2')
class EmbeddingRequest(BaseModel):
    criteria: str

@app.post(path="/embed")
def generate_embedding(data: EmbeddingRequest):
    try:
        embedding = model.encode(data.criteria).tolist()
        return {"embedding": embedding}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))