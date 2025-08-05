"""
Model Training avec SMOTE Manuel - Credit Card Fraud Detection
SMOTE implémenté manuellement pour éviter conflits imblearn
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from optuna.integration import LightGBMPruningCallback
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from sklearn.neighbors import NearestNeighbors
from sklearnex import patch_sklearn
import pickle
import json
import time
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
patch_sklearn()


class ManualSMOTE:
    """
    Implémentation SMOTE simple sans imblearn
    Génère échantillons synthétiques en interpolant entre voisins
    """
    
    def __init__(self, k_neighbors=5, sampling_ratio=0.1, random_state=42):
        """
        k_neighbors: Nombre de voisins pour interpolation
        sampling_ratio: Ratio final désiré pour classe minoritaire
        """
        self.k_neighbors = k_neighbors
        self.sampling_ratio = sampling_ratio
        self.random_state = random_state
        np.random.seed(random_state)
        
    def fit_resample(self, X, y):
        """Génération échantillons SMOTE"""
        
        # Conversion en numpy si nécessaire
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            feature_names = X.columns
        else:
            X_array = X
            feature_names = None
            
        y_array = y.values if isinstance(y, pd.Series) else y
        
        # Identification classes
        minority_class = 1  # Fraudes
        majority_class = 0  # Normal
        
        # Indices des classes
        minority_indices = np.where(y_array == minority_class)[0]
        majority_indices = np.where(y_array == majority_class)[0]
        
        # Échantillons par classe
        X_minority = X_array[minority_indices]
        X_majority = X_array[majority_indices]
        
        # Calcul nombre d'échantillons à générer
        n_majority = len(majority_indices)
        n_minority = len(minority_indices)
        
        # Nombre final désiré pour minoritaire
        target_minority = int(n_majority * self.sampling_ratio)
        n_synthetic = target_minority - n_minority
        
        if n_synthetic <= 0:
            print(f"⚠️ Pas besoin de SMOTE: ratio déjà {n_minority/n_majority:.3f}")
            return X, y
            
        print(f"🧬 SMOTE: Génération de {n_synthetic:,} échantillons synthétiques...")
        
        # Recherche des k plus proches voisins
        nbrs = NearestNeighbors(n_neighbors=min(self.k_neighbors + 1, len(X_minority)))
        nbrs.fit(X_minority)
        
        # Génération échantillons synthétiques
        synthetic_samples = []
        
        for _ in range(n_synthetic):
            # Échantillon aléatoire de la classe minoritaire
            random_idx = np.random.randint(0, len(X_minority))
            sample = X_minority[random_idx]
            
            # Trouver voisins (excluant l'échantillon lui-même)
            distances, indices = nbrs.kneighbors([sample])
            neighbor_idx = np.random.choice(indices[0][1:])  # Exclut index 0 (lui-même)
            neighbor = X_minority[neighbor_idx]
            
            # Interpolation linéaire
            alpha = np.random.random()  # Facteur d'interpolation [0,1]
            synthetic_sample = sample + alpha * (neighbor - sample)
            synthetic_samples.append(synthetic_sample)
        
        # Combinaison données originales + synthétiques
        X_synthetic = np.array(synthetic_samples)
        y_synthetic = np.ones(n_synthetic, dtype=int)  # Tous fraudes
        
        # Concaténation
        X_resampled = np.vstack([X_array, X_synthetic])
        y_resampled = np.hstack([y_array, y_synthetic])
        
        # Conversion en DataFrame si nécessaire
        if feature_names is not None:
            X_resampled = pd.DataFrame(X_resampled, columns=feature_names)
            y_resampled = pd.Series(y_resampled)
        
        print(f"✅ SMOTE terminé:")
        print(f"   • Échantillons originaux: {len(X_array):,}")
        print(f"   • Échantillons synthétiques: {n_synthetic:,}")
        print(f"   • Total final: {len(X_resampled):,}")
        print(f"   • Nouveau ratio fraude: {y_resampled.mean()*100:.1f}%")
        
        return X_resampled, y_resampled


class ManualSMOTETrainer:
    """Trainer avec SMOTE manuel"""
    
    def __init__(self, n_trials=70, cv_folds=3, random_state=42, 
                 smote_ratio=0.1, smote_k_neighbors=5):
        """
        smote_ratio: Ratio cible pour classe minoritaire (0.1 = 10%)
        smote_k_neighbors: Voisins pour génération SMOTE
        """
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.smote_ratio = smote_ratio
        self.smote_k_neighbors = smote_k_neighbors
        
        # Chemins
        self.base_dir = Path(__file__).resolve().parent.parent
        self.processed_dir = self.base_dir / "data" / "processed"
        self.models_dir = self.base_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Résultats
        self.best_model = None
        self.best_params = None
        self.best_score = 0.0
        self.training_history = {}
        self.smote_sampler = ManualSMOTE(
            k_neighbors=smote_k_neighbors,
            sampling_ratio=smote_ratio,
            random_state=random_state
        )
    
    def load_training_data(self):
        """Chargement données"""
        print("📊 Chargement données train et validation...")
        
        try:
            self.X_train = pd.read_pickle(self.processed_dir / "X_train.pkl")
            self.y_train = pd.read_pickle(self.processed_dir / "y_train.pkl")
            self.X_val = pd.read_pickle(self.processed_dir / "X_val.pkl")
            self.y_val = pd.read_pickle(self.processed_dir / "y_val.pkl")
            
            print("✅ Données chargées:")
            print(f"   • Train samples: {len(self.X_train):,}")
            print(f"   • Validation samples: {len(self.X_val):,}")
            print(f"   • Features: {self.X_train.shape[1]}")
            print(f"   • Train fraud rate: {self.y_train.mean()*100:.2f}%")
            
            # Stats déséquilibre
            fraud_count = self.y_train.sum()
            normal_count = (self.y_train == 0).sum()
            ratio = normal_count / fraud_count
            print(f"   • Ratio déséquilibre: 1:{ratio:.0f}")
            
            return True
            
        except FileNotFoundError as e:
            print(f"❌ Données non trouvées: {e}")
            return False
    
    def create_lgb_feval(self):
        """Fonction d'évaluation F1-Score"""
        def f1_eval(y_true, y_pred):
            y_pred_binary = (y_pred > 0.5).astype(int)
            f1 = f1_score(y_true, y_pred_binary)
            return 'f1', f1, True
        return f1_eval
    
    def smote_objective_function(self, trial):
        """Fonction objectif avec SMOTE manuel"""
        
        # Hyperparamètres (pas de scale_pos_weight avec SMOTE)
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'random_state': self.random_state,
            'n_jobs': -1,
            
            # Space de recherche élargi
            'n_estimators': trial.suggest_int('n_estimators', 200, 3000, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.15, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 400),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            
            # Régularisation
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 20.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 20.0, log=True),
            
            # Sampling
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 150),
        }
        
        # Cross-validation avec SMOTE
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        f1_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.X_train, self.y_train)):
            try:
                X_fold_train = self.X_train.iloc[train_idx]
                X_fold_val = self.X_train.iloc[val_idx]
                y_fold_train = self.y_train.iloc[train_idx]
                y_fold_val = self.y_train.iloc[val_idx]
                
                # Vérification samples minimum
                if y_fold_train.sum() < 5:
                    continue
                
                # Application SMOTE sur fold
                X_fold_train_smote, y_fold_train_smote = self.smote_sampler.fit_resample(
                    X_fold_train, y_fold_train
                )
                
                # Modèle
                pruning_callback = LightGBMPruningCallback(trial, 'f1')
                model = lgb.LGBMClassifier(**params)
                
                # Training sur données SMOTE
                model.fit(
                    X_fold_train_smote, y_fold_train_smote,
                    eval_set=[(X_fold_val, y_fold_val)],  # Validation sur originales
                    eval_metric=self.create_lgb_feval(),
                    callbacks=[
                        lgb.early_stopping(100, verbose=False),
                        pruning_callback
                    ]
                )
                
                # Évaluation sur validation originale
                y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
                y_pred_binary = (y_pred_proba > 0.5).astype(int)
                f1 = f1_score(y_fold_val, y_pred_binary)
                f1_scores.append(f1)
                
            except Exception as e:
                print(f"   ⚠️ Erreur fold {fold}: {str(e)[:50]}")
                continue
        
        return np.mean(f1_scores) if f1_scores else 0.0
    
    def optimize_hyperparameters(self):
        """Optimisation avec SMOTE manuel"""
        print(f"🎯 Optimisation avec SMOTE MANUEL...")
        print(f"   • Trials: {self.n_trials}")
        print(f"   • CV Folds: {self.cv_folds}")
        print(f"   • SMOTE ratio cible: {self.smote_ratio*100:.1f}%")
        print(f"   • SMOTE k-neighbors: {self.smote_k_neighbors}")
        
        start_time = time.time()
        
        # Étude Optuna
        study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
        )
        
        # Optimisation
        study.optimize(
            self.smote_objective_function,
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        
        optimization_time = time.time() - start_time
        
        # Résultats
        self.best_score = study.best_value
        self.best_params = study.best_params.copy()
        
        # Ajout paramètres fixes
        self.best_params.update({
            'objective': 'binary',
            'metric': 'binary_logloss',
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbosity': -1
        })
        
        print(f"✅ Optimisation SMOTE terminée:")
        print(f"   • Meilleur F1-Score CV: {self.best_score:.4f}")
        print(f"   • Temps: {optimization_time:.1f}s")
        
        # Historique
        self.training_history['smote_optimization'] = {
            'best_f1_score_cv': self.best_score,
            'optimization_time': optimization_time,
            'smote_config': {
                'ratio': self.smote_ratio,
                'k_neighbors': self.smote_k_neighbors
            },
            'best_params': self.best_params
        }
    
    def train_final_model(self):
        """Training final avec SMOTE"""
        print("🚀 Training modèle final avec SMOTE...")
        
        start_time = time.time()
        
        # Application SMOTE sur données train complètes
        X_train_smote, y_train_smote = self.smote_sampler.fit_resample(
            self.X_train, self.y_train
        )
        
        # Modèle final
        self.best_model = lgb.LGBMClassifier(**self.best_params)
        self.best_model.fit(X_train_smote, y_train_smote)
        
        training_time = time.time() - start_time
        
        print(f"✅ Modèle final SMOTE entraîné:")
        print(f"   • Temps: {training_time:.1f}s")
        print(f"   • Samples après SMOTE: {len(X_train_smote):,}")
        
        self.training_history['final_training'] = {
            'training_time': training_time,
            'smote_samples': len(X_train_smote),
            'original_samples': len(self.X_train)
        }
    
    def evaluate_on_validation_set(self):
        """Évaluation sur validation originale"""
        print("📊 Évaluation sur validation (données originales)...")
        
        # Prédictions sur validation ORIGINALE
        y_pred_proba = self.best_model.predict_proba(self.X_val)[:, 1]
        y_pred_binary = (y_pred_proba > 0.5).astype(int)
        
        # Métriques
        metrics = {
            'f1_score': f1_score(self.y_val, y_pred_binary),
            'roc_auc': roc_auc_score(self.y_val, y_pred_proba),
            'precision': precision_score(self.y_val, y_pred_binary),
            'recall': recall_score(self.y_val, y_pred_binary)
        }
        
        print("📈 Métriques validation (SMOTE → originales):")
        for metric, value in metrics.items():
            print(f"   • {metric.upper()}: {value:.4f}")
        
        self.training_history['validation'] = metrics
        return metrics
    
    def save_model_artifacts(self):
        """Sauvegarde avec infos SMOTE"""
        print("💾 Sauvegarde artefacts SMOTE...")
        
        # Modèle
        with open(self.models_dir / "fraud_detector_manual_smote.pkl", 'wb') as f:
            pickle.dump(self.best_model, f)
        
        # Configuration complète
        artifacts = {
            'smote_manual_config': {
                'ratio': self.smote_ratio,
                'k_neighbors': self.smote_k_neighbors,
                'implementation': 'manual_no_imblearn'
            },
            'performance': {
                'best_f1_cv': float(self.best_score)
            },
            'best_params': {k: v.item() if hasattr(v, 'item') else v 
                          for k, v in self.best_params.items()},
            'training_history': self.training_history
        }
        
        with open(self.models_dir / "manual_smote_results.json", 'w') as f:
            json.dump(artifacts, f, indent=2)
        
        print("✅ Artefacts SMOTE sauvegardés")
    
    def display_training_summary(self):
        """Résumé avec comparaison"""
        print("\n" + "=" * 70)
        print("🎉 TRAINING AVEC SMOTE MANUEL TERMINÉ")
        print("=" * 70)
        
        if 'smote_optimization' in self.training_history:
            opt = self.training_history['smote_optimization']
            print("🧬 OPTIMISATION AVEC SMOTE MANUEL:")
            print(f"   • F1-Score CV: {opt['best_f1_score_cv']:.4f}")
            print(f"   • Ratio SMOTE: {self.smote_ratio*100:.1f}%")
        
        if 'validation' in self.training_history:
            val = self.training_history['validation']
            print(f"\n📊 VALIDATION FINALE:")
            print(f"   • F1-Score: {val['f1_score']:.4f}")
            print(f"   • Precision: {val['precision']:.4f}")
            print(f"   • Recall: {val['recall']:.4f}")
            print(f"   • ROC-AUC: {val['roc_auc']:.4f}")
        
        # Comparaison vs sans SMOTE
        baseline_no_smote = 0.8060  # Ton résultat précédent
        if 'validation' in self.training_history:
            current = self.training_history['validation']['f1_score']
            improvement = ((current - baseline_no_smote) / baseline_no_smote) * 100
            print(f"\n📈 COMPARAISON:")
            print(f"   • Sans SMOTE: {baseline_no_smote:.4f}")
            print(f"   • Avec SMOTE Manuel: {current:.4f}")
            print(f"   • Gain: {improvement:+.1f}%")
        
        print("=" * 70)
    
    def run_smote_pipeline(self):
        """Pipeline complet SMOTE manuel"""
        print("🚀 DÉMARRAGE PIPELINE SMOTE MANUEL")
        print("=" * 60)
        print("🧬 SMOTE implémenté sans imblearn (évite conflits)")
        print("=" * 60)
        
        try:
            if not self.load_training_data():
                return False
            
            self.optimize_hyperparameters()
            self.train_final_model()
            self.evaluate_on_validation_set()
            self.save_model_artifacts()
            self.display_training_summary()
            
            return True
            
        except Exception as e:
            print(f"\n❌ ERREUR SMOTE MANUEL: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Exécution SMOTE manuel"""
    print("🎯 Credit Card Fraud Detection - SMOTE MANUEL")
    print("🧬 Équilibrage données SANS imblearn")
    
    # Configuration SMOTE manuel
    trainer = ManualSMOTETrainer(
        n_trials=40,
        cv_folds=3,
        random_state=42,
        smote_ratio=0.1,        # 10% de fraudes après SMOTE
        smote_k_neighbors=3     # 3 voisins pour interpolation
    )
    
    success = trainer.run_smote_pipeline()
    
    if success:
        print("\n✅ SMOTE MANUEL RÉUSSI!")
        print("🧬 Données équilibrées sans conflit de versions")
    else:
        print("\n❌ ÉCHEC SMOTE MANUEL!")

if __name__ == "__main__":
    main()