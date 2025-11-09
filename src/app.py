from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model_loader import ModelLoader
import uvicorn
import time

app = FastAPI(
    title="LLaMA 3 8B API",
    description="API pour le modèle LLaMA 3 8B - Développement Équipe",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chargement du modèle
print("🚀 Démarrage de l'API LLaMA 3 8B...")
model_loader = ModelLoader()

class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

class GenerationResponse(BaseModel):
    generated_text: str
    inference_time: float
    model: str = "LLaMA 3 8B"

@app.get("/")
async def health_check():
    return {
        "status": "healthy", 
        "model": "LLaMA 3 8B",
        "message": "API de développement - Équipe LLaMA"
    }

@app.get("/health")
async def detailed_health():
    return {
        "status": "healthy",
        "model_loaded": model_loader.is_loaded,
        "device": model_loader.model.device if model_loader.is_loaded else "not_loaded"
    }

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    try:
        start_time = time.time()
        
        generated_text = model_loader.generate_text(
            prompt=request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        inference_time = time.time() - start_time
        
        return GenerationResponse(
            generated_text=generated_text,
            inference_time=inference_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
