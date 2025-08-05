"""
Tests API - Credit Card Fraud Detection
VERSION FINALE avec TestClient pour des tests rapides et fiables en CI/CD.
"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import sys

# Ajout du chemin racine du projet pour les imports
sys.path.append(str(Path(__file__).parent.parent))

# On importe notre application FastAPI directement depuis le code source
from src.api import app 

# La fixture "magique" qui crée un client de test pour notre application.
# Pytest va l'injecter automatiquement dans chaque fonction de test qui en a besoin.
@pytest.fixture(scope="module")
def client():
    # Pas besoin de démarrer un serveur ! TestClient s'en occupe en mémoire.
    with TestClient(app) as test_client:
        yield test_client

# --- Suite de Tests pour l'API ---

# On n'a plus besoin d'une classe ici, des fonctions de test simples suffisent.
# Les données de test peuvent être définies au niveau du module.
test_transaction = {
    "Amount": 100.0, "Time": 3600.0, "V1": 0.0, "V2": 0.0, "V3": 0.0, "V4": 1.0, "V5": 0.0,
    "V6": 0.0, "V7": 0.0, "V8": 0.0, "V9": 0.0, "V10": -1.0, "V11": 0.0, "V12": 2.0, "V13": 0.0,
    "V14": -3.0, "V15": 0.0, "V16": 0.0, "V17": -5.0, "V18": 0.0, "V19": 0.0, "V20": 0.0,
    "V21": 0.0, "V22": 0.0, "V23": 0.0, "V24": 0.0, "V25": 0.0, "V26": 0.0, "V27": 0.0, "V28": 0.0
}

fraud_transaction = {
    "Amount": 5000.0, "Time": 7200.0, "V1": 0.0, "V2": 0.0, "V3": 0.0, "V4": 5.0, "V5": 0.0,
    "V6": 0.0, "V7": 0.0, "V8": 0.0, "V9": 0.0, "V10": 3.0, "V11": 0.0, "V12": -4.0, "V13": 0.0,
    "V14": -10.0, "V15": 0.0, "V16": 0.0, "V17": -8.0, "V18": 0.0, "V19": 0.0, "V20": 0.0,
    "V21": 0.0, "V22": 0.0, "V23": 0.0, "V24": 0.0, "V25": 0.0, "V26": 0.0, "V27": 0.0, "V28": 0.0
}

def test_api_health_check(client: TestClient):
    """Test health check endpoint."""
    # Fini requests, bonjour TestClient ! La syntaxe est presque identique.
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True

def test_root_endpoint(client: TestClient):
    """Test endpoint racine."""
    response = client.get("/")
    assert response.status_code == 200
    assert "service" in response.json()

def test_model_info_endpoint(client: TestClient):
    """Test informations modèle."""
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_type"] == "LightGBM"
    assert "performance_metrics" in data

def test_single_prediction_normal(client: TestClient):
    """Test prédiction transaction normale."""
    response = client.post("/predict", json=test_transaction)
    assert response.status_code == 200
    data = response.json()
    assert "is_fraud" in data

def test_invalid_transaction_data(client: TestClient):
    """Test validation données invalides."""
    invalid_data = {"Amount": 100.0, "Time": 3600.0} # V1-V28 manquants
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422 # Unprocessable Entity

def test_batch_prediction(client: TestClient):
    """Test prédictions batch."""
    batch_data = {"transactions": [test_transaction, fraud_transaction]}
    response = client.post("/predict/batch", json=batch_data)
    assert response.status_code == 200
    data = response.json()
    assert data["batch_size"] == 2
    assert len(data["predictions"]) == 2
