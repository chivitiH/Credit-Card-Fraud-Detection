# Credit Card Fraud Detection - Production Docker Image
FROM python:3.11-slim

# Métadonnées
LABEL maintainer="votre.email@domain.com"
LABEL description="Credit Card Fraud Detection API - Production Ready"
LABEL version="1.0.0"

# Variables d'environnement
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MODEL_PATH=/app/models
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

# Mise à jour système et installation dépendances système
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Répertoire de travail
WORKDIR /app

# Installation dépendances Python (optimisation cache Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copie du code source
COPY src/ ./src/
COPY models/ ./models/
COPY data/processed/ ./data/processed/

# Création utilisateur non-root pour sécurité
RUN useradd --create-home --shell /bin/bash --uid 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Health check pour monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${API_PORT}/health || exit 1

# Exposition du port
EXPOSE ${API_PORT}

# Commande de démarrage
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]