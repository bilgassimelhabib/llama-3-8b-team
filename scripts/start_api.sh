#!/bin/bash
echo "🚀 Démarrage de l'API LLaMA 3 8B..."
cd ~/llama-project
source ~/llama-env/bin/activate
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
