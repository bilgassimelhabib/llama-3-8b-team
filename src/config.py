import os
import torch
from pathlib import Path

class Config:
    # Modèle
    MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
    
    # Device detection automatique
    @property
    def device(self):
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    # Chemins adaptatifs
    @property
    def model_cache(self):
        if os.path.exists("/content"):  # Colab
            return "/content/drive/MyDrive/llama_cache"
        elif os.path.exists("/home/ubuntu"):  # AWS
            return "/home/ubuntu/llama_cache"
        else:  # Local
            return "./models"
    
    # Paramètres d'inférence
    MAX_LENGTH = 512
    TEMPERATURE = 0.7
    TOP_P = 0.9
    TOP_K = 50

config = Config()

if __name__ == "__main__":
    print(f"Device: {config.device}")
    print(f"Model cache: {config.model_cache}")
