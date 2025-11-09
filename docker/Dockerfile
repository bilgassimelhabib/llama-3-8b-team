FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copier les requirements d'abord pour meilleur caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY src/ ./src/
COPY scripts/ ./scripts/

# Variables d'environnement
ENV PYTHONPATH=/app/src
ENV MODEL_CACHE=/app/models
ENV HF_HOME=/app/models

# Port pour l'API
EXPOSE 8000

CMD ["python", "src/app.py"]
