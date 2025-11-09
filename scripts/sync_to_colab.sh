#!/bin/bash
echo "🔄 SYNCHRONISATION VERS COLAB"
echo "============================="

cd ~/llama-project

if [ -d "/content/drive/MyDrive" ]; then
    echo "📁 Copie du projet vers Google Drive..."
    cp -r /content/llama-project /content/drive/MyDrive/llama-project-sync
    date > /content/drive/MyDrive/llama-project-sync/last_sync.txt
    echo "✅ Synchronisation terminée: $(date)"
else
    echo "❌ Google Drive non monté - Ce script doit tourner sur Colab"
fi
