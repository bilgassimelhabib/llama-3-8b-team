#!/usr/bin/env python3
COLAB_QUICK_START = '''
# 🚀 DÉMARRAGE RAPIDE COLAB - Équipe LLaMA
# Copiez tout ceci dans une cellule Colab

# 1. Installation
!pip install -q torch>=2.1.0 transformers>=4.35.0 accelerate>=0.24.0 bitsandbytes>=0.41.0 huggingface-hub>=0.19.0 gradio>=4.0.0

# 2. Synchronisation code équipe
!git clone https://github.com/VOTRE_USERNAME/llama-3-8b-team.git /content/llama-project
%cd /content/llama-project

# 3. Import et test
import sys
sys.path.append('/content/llama-project/src')

from model_loader import ModelLoader
loader = ModelLoader()
model, tokenizer = loader.load_model(quantize_4bit=True)

# 4. Test rapide
response = loader.generate_text("Bonjour LLaMA! Tu fonctionnes sur Colab?")
print("🤖 Réponse:", response)

print("🎉 Environnement Colab prêt!")
'''

if __name__ == "__main__":
    print(COLAB_QUICK_START)
