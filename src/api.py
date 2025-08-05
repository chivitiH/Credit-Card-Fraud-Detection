"""
FastAPI Service - Credit Card Fraud Detection
API haute performance pour serving modèle ML en production
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, validator
import pandas as pd
import numpy as np
import pickle
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration chemins
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

class TransactionData(BaseModel):
    """Modèle de données pour transaction"""
    
    # Features PCA V1-V28 (obligatoires)
    V1: float = Field(..., description="PCA feature V1")
    V2: float = Field(..., description="PCA feature V2")
    V3: float = Field(..., description="PCA feature V3")
    V4: float = Field(..., description="PCA feature V4")
    V5: float = Field(..., description="PCA feature V5")
    V6: float = Field(..., description="PCA feature V6")
    V7: float = Field(..., description="PCA feature V7")
    V8: float = Field(..., description="PCA feature V8")
    V9: float = Field(..., description="PCA feature V9")
    V10: float = Field(..., description="PCA feature V10")
    V11: float = Field(..., description="PCA feature V11")
    V12: float = Field(..., description="PCA feature V12")
    V13: float = Field(..., description="PCA feature V13")
    V14: float = Field(..., description="PCA feature V14 (important)")
    V15: float = Field(..., description="PCA feature V15")
    V16: float = Field(..., description="PCA feature V16")
    V17: float = Field(..., description="PCA feature V17 (important)")
    V18: float = Field(..., description="PCA feature V18")
    V19: float = Field(..., description="PCA feature V19")
    V20: float = Field(..., description="PCA feature V20")
    V21: float = Field(..., description="PCA feature V21")
    V22: float = Field(..., description="PCA feature V22")
    V23: float = Field(..., description="PCA feature V23")
    V24: float = Field(..., description="PCA feature V24")
    V25: float = Field(..., description="PCA feature V25")
    V26: float = Field(..., description="PCA feature V26")
    V27: float = Field(..., description="PCA feature V27")
    V28: float = Field(..., description="PCA feature V28")
    
    # Features originales
    Amount: float = Field(..., ge=0, description="Montant transaction (≥0)")
    Time: float = Field(..., ge=0, description="Temps depuis première transaction (secondes)")
    
    @validator('Amount')
    def validate_amount(cls, v):
        if v < 0:
            raise ValueError('Amount must be non-negative')
        if v > 25000:  # Montant suspicieusement élevé
            logger.warning(f"Very high amount detected: {v}")
        return v
    
    @validator('Time')  
    def validate_time(cls, v):
        if v < 0:
            raise ValueError('Time must be non-negative')
        if v > 172800:  # Plus de 48h
            logger.warning(f"Very high time value: {v}")
        return v

class BatchTransactionData(BaseModel):
    """Modèle pour prédictions batch"""
    transactions: List[TransactionData] = Field(..., min_items=1, max_items=1000)

class PredictionResponse(BaseModel):
    """Réponse prédiction unique"""
    is_fraud: bool = Field(..., description="Transaction frauduleuse (True/False)")
    fraud_probability: float = Field(..., ge=0, le=1, description="Probabilité fraude [0-1]")
    risk_score: int = Field(..., ge=0, le=100, description="Score risque [0-100]")
    confidence: str = Field(..., description="Niveau confiance (LOW/MEDIUM/HIGH)")
    processing_time_ms: float = Field(..., description="Temps traitement (millisecondes)")
    model_version: str = Field(..., description="Version modèle utilisé")

class BatchPredictionResponse(BaseModel):
    """Réponse prédictions batch"""
    predictions: List[PredictionResponse]
    batch_size: int
    total_fraud_detected: int
    average_risk_score: float
    processing_time_ms: float

class ModelInfo(BaseModel):
    """Informations modèle"""
    model_type: str
    model_version: str
    training_date: str
    features_count: int
    performance_metrics: Dict[str, float]
    optimal_threshold: float

class FraudDetectionAPI:
    """API Fraud Detection avec modèle chargé"""
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.model_info = {}
        self.optimal_threshold = 0.5
        self.model_version = "1.0.0"
        
    def load_model_artifacts(self):
        """Chargement modèle et artefacts"""
        try:
            # Chargement modèle principal
            model_path = MODELS_DIR / "fraud_detector.pkl"
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info("✅ Modèle chargé avec succès")
            
            # Chargement preprocessor
            try:
                preprocessor_path = MODELS_DIR / "preprocessor.pkl"
                with open(preprocessor_path, 'rb') as f:
                    self.preprocessor = pickle.load(f)
                logger.info("✅ Preprocessor chargé")
            except FileNotFoundError:
                logger.warning("⚠️ Preprocessor non trouvé, scaling désactivé")
            
            # Chargement métriques
            try:
                # === CORRECTION ICI ===
                # On s'assure que l'API charge le fichier de métriques que le pipeline d'évaluation produit réellement.
                metrics_path = MODELS_DIR / "final_test_metrics.json"
                with open(metrics_path, 'r') as f:
                    metrics_data = json.load(f)
                    # On va chercher les métriques dans la bonne structure du JSON
                    cls_metrics = metrics_data.get('classification_metrics', {})
                    prob_metrics = metrics_data.get('probabilistic_metrics', {})
                    self.optimal_threshold = metrics_data.get('optimal_threshold', 0.5)
                    
                    self.model_info = {
                        'model_type': 'LightGBM',
                        'model_version': self.model_version,
                        'training_date': '2024-01-15', # Tu peux rendre ça dynamique plus tard
                        'features_count': len(self.model.feature_name_) if hasattr(self.model, 'feature_name_') else 0,
                        'performance_metrics': {
                            'f1_score': cls_metrics.get('f1_score', 0.0),
                            'precision': cls_metrics.get('precision', 0.0),
                            'recall': cls_metrics.get('recall', 0.0),
                            'roc_auc': prob_metrics.get('roc_auc', 0.0)
                        },
                        'optimal_threshold': self.optimal_threshold
                    }
                logger.info("✅ Métriques chargées")
            except FileNotFoundError:
                logger.warning("⚠️ Métriques non trouvées, valeurs par défaut utilisées")
                self.model_info = {
                        'model_type': 'LightGBM',
                        'model_version': self.model_version,
                        'training_date': 'N/A',
                        'features_count': 0,
                        'performance_metrics': {},
                        'optimal_threshold': self.optimal_threshold
                    }
                
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement modèle: {e}")
            return False
    
    def preprocess_transaction(self, transaction_data: TransactionData) -> pd.DataFrame:
        """Preprocessing transaction pour prédiction"""
        
        data_dict = transaction_data.dict()
        df = pd.DataFrame([data_dict])
        
        try:
            df['Hour'] = (df['Time'] / 3600) % 24
            df['Day'] = (df['Time'] / 86400) % 7
            df['Is_Weekend'] = df['Day'].isin([5, 6]).astype(int)
            df['Is_Night'] = ((df['Hour'] >= 22) | (df['Hour'] <= 6)).astype(int)
            
            df['Amount_Log'] = np.log1p(df['Amount'])
            df['Amount_Sqrt'] = np.sqrt(df['Amount'])
            df['Is_High_Amount'] = (df['Amount'] > 640.90).astype(int)
            df['Is_Very_High_Amount'] = (df['Amount'] > 2125.87).astype(int)
            df['Is_Low_Amount'] = (df['Amount'] == 0).astype(int)
            
            df['V14_V4_ratio'] = df['V14'] / (df['V4'] + 1e-8)
            df['V12_V10_ratio'] = df['V12'] / (df['V10'] + 1e-8)
            df['V17_V14_ratio'] = df['V17'] / (df['V14'] + 1e-8)
            
            important_features = ['V14', 'V4', 'V12', 'V10', 'V17', 'V16', 'V3', 'V11']
            df['Important_Features_Sum'] = df[important_features].sum(axis=1)
            df['Important_Features_Mean'] = df[important_features].mean(axis=1)
            df['Important_Features_Std'] = df[important_features].std(axis=1)
            
            pca_features = [f'V{i}' for i in range(1, 29)]
            df['PCA_Sum'] = df[pca_features].sum(axis=1)
            df['PCA_Mean'] = df[pca_features].mean(axis=1)
            df['PCA_Max'] = df[pca_features].max(axis=1)
            df['PCA_Min'] = df[pca_features].min(axis=1)
            
            if self.preprocessor is not None:
                features_to_scale = ['Amount', 'Time', 'Amount_Log', 'Amount_Sqrt',
                                   'Important_Features_Sum', 'Important_Features_Mean',
                                   'PCA_Sum', 'PCA_Mean', 'PCA_Max', 'PCA_Min']
                
                available_features = [f for f in features_to_scale if f in df.columns]
                if available_features:
                    df[available_features] = self.preprocessor.transform(df[available_features])
            
            # S'assurer que les colonnes sont dans le bon ordre pour le modèle
            if hasattr(self.model, 'feature_name_'):
                model_features = self.model.feature_name_
                df = df[model_features]

            return df
            
        except Exception as e:
            logger.error(f"Erreur preprocessing: {e}")
            return pd.DataFrame([transaction_data.dict()])
    
    def predict_single(self, transaction_data: TransactionData) -> PredictionResponse:
        """Prédiction transaction unique"""
        start_time = time.time()
        
        try:
            df_processed = self.preprocess_transaction(transaction_data)
            
            fraud_probability = float(self.model.predict_proba(df_processed)[0, 1])
            
            is_fraud = fraud_probability >= self.optimal_threshold
            
            risk_score = int(fraud_probability * 100)
            
            # Logique de confiance corrigée
            if (is_fraud and fraud_probability > 0.75) or (not is_fraud and fraud_probability < 0.25):
                confidence = "HIGH"
            else:
                confidence = "MEDIUM" if 0.4 < fraud_probability < 0.6 else "LOW"

            processing_time = (time.time() - start_time) * 1000
            
            return PredictionResponse(
                is_fraud=is_fraud,
                fraud_probability=fraud_probability,
                risk_score=risk_score,
                confidence=confidence,
                processing_time_ms=processing_time,
                model_version=self.model_version
            )
            
        except Exception as e:
            logger.error(f"Erreur prédiction: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur prédiction: {str(e)}")
    
    def predict_batch(self, batch_data: BatchTransactionData) -> BatchPredictionResponse:
        """Prédictions batch"""
        start_time = time.time()
        
        try:
            predictions = []
            fraud_count = 0
            total_risk = 0
            
            for transaction in batch_data.transactions:
                pred = self.predict_single(transaction)
                predictions.append(pred)
                
                if pred.is_fraud:
                    fraud_count += 1
                total_risk += pred.risk_score
            
            processing_time = (time.time() - start_time) * 1000
            avg_risk = total_risk / len(predictions) if predictions else 0
            
            return BatchPredictionResponse(
                predictions=predictions,
                batch_size=len(predictions),
                total_fraud_detected=fraud_count,
                average_risk_score=avg_risk,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Erreur batch: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur batch: {str(e)}")

# Initialisation
fraud_api = FraudDetectionAPI()
app = FastAPI(title="Credit Card Fraud Detection API", version="1.0.0")

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Démarrage API Fraud Detection...")
    success = fraud_api.load_model_artifacts()
    if not success:
        logger.error("❌ Impossible de charger le modèle. L'API démarrera avec des fonctionnalités limitées.")
        # Ne pas lever d'erreur permet à /health de répondre, ce qui est mieux pour les orchestrateurs
    else:
        logger.info("✅ API prête pour prédictions")

# Endpoints
@app.get("/", summary="Root endpoint")
async def root():
    return {"service": "Credit Card Fraud Detection API", "status": "operational"}

@app.get("/health", summary="Health check")
async def health_check():
    model_loaded = fraud_api.model is not None
    return {"status": "healthy" if model_loaded else "unhealthy", "model_loaded": model_loaded}

@app.get("/model/info", response_model=ModelInfo, summary="Informations modèle")
async def get_model_info():
    if not fraud_api.model_info:
        raise HTTPException(status_code=503, detail="Informations du modèle non disponibles. Vérifiez les logs de démarrage.")
    return ModelInfo(**fraud_api.model_info)

@app.post("/predict", response_model=PredictionResponse, summary="Prédiction unique")
async def predict_transaction(transaction: TransactionData):
    if not fraud_api.model:
        raise HTTPException(status_code=503, detail="Modèle non disponible pour la prédiction.")
    return fraud_api.predict_single(transaction)

@app.post("/predict/batch", response_model=BatchPredictionResponse, summary="Prédictions batch")
async def predict_batch(batch: BatchTransactionData):
    if not fraud_api.model:
        raise HTTPException(status_code=503, detail="Modèle non disponible pour la prédiction.")
    return fraud_api.predict_batch(batch)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)