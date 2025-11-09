#!/usr/bin/env python3
import sys
sys.path.append('./src')

from config import Config

def test_environment():
    print("🧪 TEST ENVIRONNEMENT")
    print("=====================")
    
    config = Config()
    print(f"✅ Device: {config.device}")
    print(f"✅ Cache: {config.model_cache}")
    print(f"✅ Modèle: {config.MODEL_NAME}")
    
    # Test des dépendances
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch: {e}")
    
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"❌ Transformers: {e}")
    
    print("🎉 Environnement de test OK!")

if __name__ == "__main__":
    test_environment()
