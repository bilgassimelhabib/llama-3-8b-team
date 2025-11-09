# 🧪 Guide de Validation Colab - Équipe LLaMA (CORRIGÉ)

## 🚀 Validation Express (5 minutes)

**Sur Colab, collez ceci dans une cellule :**

\`\`\`python
# 🚀 VALIDATION EXPRESS CORRIGÉE - Équipe LLaMA
!pip install -q torch transformers accelerate bitsandbytes huggingface-hub
!git clone https://github.com/VOTRE_USERNAME/llama-3-8b-team.git
%cd llama-3-8b-team

import sys
sys.path.append('/content/llama-3-8b-team')
sys.path.append('/content/llama-3-8b-team/src')

from src.config import Config
from src.model_loader import ModelLoader

print("✅ Tous les imports fonctionnent!")
config = Config()
print(f"📍 Device: {config.device}")
print("🎉 Environnement Colab validé!")
\`\`\`

## 🔧 Si les imports échouent encore :

\`\`\`python
# Solution alternative
import sys
sys.path.insert(0, '/content/llama-3-8b-team/src')

try:
    from config import Config
    from model_loader import ModelLoader
except ImportError:
    print("❌ Problème d'import - Vérifiez la structure GitHub")
\`\`\`

**Bon courage à toute l'équipe!** 🦙🚀
