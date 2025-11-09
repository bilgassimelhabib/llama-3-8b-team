#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from model_loader import ModelLoader

def main():
    print("🧪 Test du modèle LLaMA 3 8B")
    print("=" * 40)
    
    loader = ModelLoader()
    model, tokenizer = loader.load_model(quantize_4bit=True)
    
    # Test avec un prompt simple
    test_prompt = "Explique-moi l'apprentissage automatique en deux phrases:"
    
    print(f"Prompt: {test_prompt}")
    print("Génération...")
    
    response = loader.generate_text(test_prompt, max_length=150)
    print(f"Réponse: {response}")
    print("✅ Test réussi!")

if __name__ == "__main__":
    main()
