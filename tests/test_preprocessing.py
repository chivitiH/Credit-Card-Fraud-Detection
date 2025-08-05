# tests/test_preprocessing.py

import pytest
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys

# Ajout du chemin racine du projet pour que l'import fonctionne
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_preprocessing import FraudDataPreprocessor

@pytest.fixture
def preprocessor_instance() -> FraudDataPreprocessor:
    """Fixture pour créer une instance de notre preprocessor."""
    return FraudDataPreprocessor(
        test_size=0.2, 
        validation_size=0.2, 
        random_state=42
    )

@pytest.fixture
def synthetic_raw_data(tmp_path: Path) -> Path:
    """
    Fixture qui crée un faux fichier 'creditcard.csv' dans un répertoire temporaire.
    C'est la clé pour isoler nos tests du vrai dataset.
    """
    data_dir = tmp_path / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    file_path = data_dir / "creditcard.csv"
    
    # Création de données synthétiques mais réalistes
    n_samples = 200
    data = pd.DataFrame({
        'Time': np.random.uniform(0, 172800, n_samples),
        **{f'V{i}': np.random.randn(n_samples) for i in range(1, 29)},
        'Amount': np.random.lognormal(mean=2.5, sigma=1.5, size=n_samples),
        'Class': np.random.choice([0, 1], n_samples, p=[0.98, 0.02])
    })
    
    # Ajout de quelques doublons pour tester le nettoyage
    data = pd.concat([data, data.head(5)], ignore_index=True)
    data.to_csv(file_path, index=False)
    
    return file_path

def test_full_preprocessing_pipeline(preprocessor_instance: FraudDataPreprocessor, synthetic_raw_data: Path, tmp_path: Path):
    """
    Test d'intégration complet pour le pipeline de preprocessing.
    On teste le "contrat public" de la classe : la méthode run_full_pipeline.
    """
    # Arrange: Configurer les chemins du preprocessor pour utiliser notre répertoire temporaire
    preprocessor = preprocessor_instance
    base_dir = tmp_path
    preprocessor.base_dir = base_dir
    preprocessor.raw_data_path = synthetic_raw_data
    preprocessor.processed_dir = base_dir / "data" / "processed"
    preprocessor.models_dir = base_dir / "models"
    
    # === LA CORRECTION ===
    # On s'assure que les répertoires de destination existent bien dans notre dossier temporaire
    # avant de lancer le pipeline qui va essayer d'y écrire.
    preprocessor.processed_dir.mkdir(parents=True, exist_ok=True)
    preprocessor.models_dir.mkdir(parents=True, exist_ok=True)
    # =====================

    # Act: Lancer le pipeline complet
    success = preprocessor.run_full_pipeline()

    # Assert: Vérifier que tout s'est bien passé
    assert success is True, "Le pipeline de preprocessing a échoué"
    
    # Vérifier que les dossiers ont été créés
    assert preprocessor.processed_dir.exists()
    assert preprocessor.models_dir.exists()
    
    # Vérifier que tous les artefacts attendus ont été créés
    expected_files = [
        "X_train.pkl", "y_train.pkl",
        "X_val.pkl", "y_val.pkl",
        "X_test.pkl", "y_test.pkl"
    ]
    for f in expected_files:
        assert (preprocessor.processed_dir / f).exists(), f"Fichier manquant: {f}"
        
    assert (preprocessor.models_dir / "preprocessor.pkl").exists(), "Le scaler n'a pas été sauvegardé"

    # Vérifier le contenu d'un des fichiers pour s'assurer qu'il est correct
    X_train = pd.read_pickle(preprocessor.processed_dir / "X_train.pkl")
    y_test = pd.read_pickle(preprocessor.processed_dir / "y_test.pkl")
    
    assert isinstance(X_train, pd.DataFrame)
    assert not X_train.empty
    assert 'Amount_Log' in X_train.columns # Vérifie qu'une feature a bien été créée
    
    assert isinstance(y_test, pd.Series)
    assert not y_test.empty
    
    # Vérifier que le scaler a été sauvegardé et est fonctionnel
    with open(preprocessor.models_dir / "preprocessor.pkl", 'rb') as f:
        scaler = pickle.load(f)
    assert hasattr(scaler, 'transform'), "L'objet sauvegardé n'est pas un scaler valide"