#!/bin/bash
echo "🔄 SYNCHRONISATION DEPUIS GITHUB"
echo "================================"

GITHUB_REPO="https://github.com/VOTRE_USERNAME/llama-3-8b-team.git"

cd /content

if [ -d "llama-project" ]; then
    echo "📁 Mise à jour du dépôt existant..."
    cd llama-project
    git pull origin main
else
    echo "📁 Clone du dépôt GitHub..."
    git clone $GITHUB_REPO llama-project
    cd llama-project
fi

echo "✅ Synchronisation GitHub terminée"
echo "📊 Dernier commit: $(git log -1 --oneline)"
