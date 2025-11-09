# 🧪 Guide de Validation Colab - Équipe LLaMA

## 🚀 Validation Express (5 minutes)

**Sur Colab, collez ceci dans une cellule :**

\`\`\`python
# 🚀 VALIDATION EXPRESS - Équipe LLaMA
!pip install -q torch transformers accelerate bitsandbytes huggingface-hub
!git clone https://github.com/VOTRE_USERNAME/llama-3-8b-team.git
%cd llama-3-8b-team

import sys
sys.path.append('/content/llama-3-8b-team/src')

from model_loader import ModelLoader
loader = ModelLoader()
model, tokenizer = loader.load_model(quantize_4bit=True)

response = loader.generate_text("Bonjour! Confirme que tu fonctionnes:")
print("🤖", response)
\`\`\`

## 📊 Validation Complète

1. **Téléchargez le notebook** depuis AWS :
   \`\`\`bash
   # Sur votre machine locale
   scp dev1@IP_AWS:~/llama-project/notebooks/llama3_validation_colab.ipynb .
   \`\`\`

2. **Upload sur Colab** : 
   - Allez sur [colab.research.google.com](https://colab.research.google.com)
   - Fichier → Upload notebook
   - Sélectionnez le fichier téléchargé

3. **Suivez les étapes** dans le notebook

## 📈 Métriques de Succès

| Métrique | Minimum | Idéal |
|----------|---------|--------|
| Tokens/sec | 2 | 5+ |
| Mémoire GPU | < 10GB | < 8GB |

## 🐛 Dépannage

**Problème** : Out of Memory
**Solution** : Utilisez \`quantize_4bit=True\`

**Problème** : GitHub inaccessible  
**Solution** : Utilisez l'option Google Drive dans le notebook

## 📝 Rapport de Validation

Chaque membre doit remplir :
\`\`\`markdown
## Rapport - [Votre Nom]
- **Date** : [date]
- **GPU Colab** : [T4/P100/Other]
- **Performance** : [X] tokens/sec
- **Statut** : ✅ VALIDÉ / ❌ ÉCHEC
\`\`\`

**Bon courage à toute l'équipe!** 🦙🚀
