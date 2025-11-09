#!/usr/bin/env python3
import sys
import torch
import transformers

print("🧪 TEST DE L'ENVIRONNEMENT DE DÉVELOPPEMENT")
print("=" * 50)

# Test Python
print(f"✅ Python: {sys.version}")

# Test PyTorch
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA disponible: {torch.cuda.is_available()}")
print(f"✅ Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

# Test Transformers
print(f"✅ Transformers: {transformers.__version__}")

# Test mémoire
if hasattr(torch.cuda, 'get_device_properties') and torch.cuda.is_available():
    gpu_props = torch.cuda.get_device_properties(0)
    print(f"✅ GPU: {gpu_props.name}")
    print(f"✅ Mémoire GPU: {gpu_props.total_memory / 1e9:.1f} GB")

print("🎉 ENVIRONNEMENT PRÊT POUR LE DÉVELOPPEMENT LLaMA !")
