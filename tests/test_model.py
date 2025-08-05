# tests/test_model.py

import pytest
import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
import json
from pathlib import Path
import sys

# Ajout du chemin racine
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.model_training import ManualSMOTETrainer, ManualSMOTE
from src.model_evaluation import FraudModelEvaluator, convert_numpy_types

# --- Tests pour ManualSMOTE (classe utilitaire) ---
class TestManualSMOTE:
    """Tests pour la classe ManualSMOTE, qui est purement algorithmique."""
    
    @pytest.fixture
    def smote_data(self):
        """Données déséquilibrées pour tester SMOTE."""
        X = pd.DataFrame({
            'f1': np.concatenate([np.random.normal(0, 1, 95), np.random.normal(5, 1, 5)]),
            'f2': np.concatenate([np.random.normal(0, 1, 95), np.random.normal(5, 1, 5)])
        })
        y = pd.Series([0]*95 + [1]*5)
        return X, y

    def test_smote_resampling(self, smote_data):
        """Vérifie que SMOTE génère de nouveaux échantillons pour la classe minoritaire."""
        X, y = smote_data
        sampling_ratio = 0.5
        smote = ManualSMOTE(k_neighbors=3, sampling_ratio=sampling_ratio, random_state=42)
        
        X_res, y_res = smote.fit_resample(X, y)
        
        assert len(X_res) > len(X), "SMOTE n'a pas ajouté d'échantillons."
        assert y_res.sum() > y.sum(), "SMOTE n'a pas ajouté d'échantillons minoritaires."
        
        n_majority = y.value_counts()[0]
        n_minority_final = int(n_majority * sampling_ratio)
        expected_total = n_majority + n_minority_final
        expected_ratio = n_minority_final / expected_total
        
        final_ratio = y_res.mean()
        
        assert abs(final_ratio - expected_ratio) < 0.01, f"Le ratio final ({final_ratio:.3f}) ne correspond pas au ratio attendu ({expected_ratio:.3f})"

# --- Fixtures pour les tests d'intégration ---

@pytest.fixture
def processed_data_for_training(tmp_path: Path) -> Path:
    """Fixture qui crée de faux fichiers de données pré-traitées (sortie du preprocessing)."""
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    for split in ["train", "val"]:
        n_samples = 100 if split == "train" else 40
        y_fraud_count = 10 
        X = pd.DataFrame(np.random.rand(n_samples, 10), columns=[f'V{i}' for i in range(10)])
        y = pd.Series([0]*(n_samples - y_fraud_count) + [1]*y_fraud_count)
        X.to_pickle(processed_dir / f"X_{split}.pkl")
        y.to_pickle(processed_dir / f"y_{split}.pkl")
        
    return tmp_path

@pytest.fixture
def trained_model_artifacts_for_evaluation(tmp_path: Path) -> Path:
    """Fixture qui crée de faux artefacts de training avec les NOMS EXACTS attendus par l'évaluateur."""
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_filename = "fraud_detector.pkl"
    history_filename = "training_history.json"

    model = lgb.LGBMClassifier(n_estimators=5, random_state=42)
    X_dummy, y_dummy = np.random.rand(20, 10), np.random.choice([0,1], 20)
    model.fit(X_dummy, y_dummy)
    with open(models_dir / model_filename, "wb") as f:
        pickle.dump(model, f)
        
    results = {
        'honest_validation': {'f1_score': 0.85, 'roc_auc': 0.95}
    }
    with open(models_dir / history_filename, "w") as f:
        json.dump(results, f)

    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    X_test = pd.DataFrame(np.random.rand(50, 10), columns=[f'V{i}' for i in range(10)])
    y_test = pd.Series(np.random.choice([0, 1], 50, p=[0.9, 0.1]))
    X_test.to_pickle(processed_dir / "X_test.pkl")
    y_test.to_pickle(processed_dir / "y_test.pkl")
    
    return tmp_path

# --- Tests pour ManualSMOTETrainer ---

def test_smote_trainer_pipeline(processed_data_for_training: Path):
    """Test d'intégration léger pour le pipeline de training."""
    trainer = ManualSMOTETrainer(
        n_trials=1,
        cv_folds=2,
        random_state=42,
        smote_ratio=0.5,
        smote_k_neighbors=3
    )
    
    base_dir = processed_data_for_training
    trainer.base_dir = base_dir
    trainer.processed_dir = base_dir / "data" / "processed"
    trainer.models_dir = base_dir / "models"
    
    trainer.models_dir.mkdir(parents=True, exist_ok=True)

    success = trainer.run_smote_pipeline()
    
    assert success is True, f"Le pipeline de training a échoué. Vérifiez les logs."
    
    model_path = trainer.models_dir / "fraud_detector_manual_smote.pkl"
    results_path = trainer.models_dir / "manual_smote_results.json"
    
    assert model_path.exists()
    assert results_path.exists()
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    assert 'best_f1_cv' in results['performance']
    assert results['training_history']['validation']['f1_score'] >= 0

# --- Tests pour FraudModelEvaluator ---

def test_evaluator_pipeline(trained_model_artifacts_for_evaluation: Path):
    """Test d'intégration léger pour le pipeline d'évaluation."""
    # Arrange
    evaluator = FraudModelEvaluator()
    
    base_dir = trained_model_artifacts_for_evaluation
    evaluator.base_dir = base_dir
    evaluator.processed_dir = base_dir / "data" / "processed"
    evaluator.models_dir = base_dir / "models"
    evaluator.assets_dir = base_dir / "assets"
    
    # === LA DERNIÈRE CORRECTION EST ICI ===
    # On crée le dossier assets avant de lancer le pipeline.
    evaluator.assets_dir.mkdir(parents=True, exist_ok=True)
    # ======================================

    # Act: Lancer le pipeline complet d'évaluation
    success = evaluator.run_final_evaluation()
    
    # Assert
    assert success is True, "Le pipeline d'évaluation a échoué."

    assert (evaluator.assets_dir / "final_evaluation_report.md").exists()
    assert (evaluator.models_dir / "final_test_metrics.json").exists()
    
    with open(evaluator.models_dir / "final_test_metrics.json", 'r') as f:
        metrics = json.load(f)
        
    assert "RAPPORT FINAL" in (evaluator.assets_dir / "final_evaluation_report.md").read_text(encoding='utf-8')
    assert 'f1_score' in metrics['classification_metrics']
    assert metrics['business_impact']['total_cost_eur'] >= 0