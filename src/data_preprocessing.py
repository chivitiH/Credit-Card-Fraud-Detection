"""
Data Preprocessing Pipeline - Credit Card Fraud Detection
Pipeline CORRIGÉ pour éviter le data leakage
RÈGLE D'OR: Test set complètement isolé dès le début
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearnex import patch_sklearn  # Intel optimization
import pickle
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration Intel optimization
patch_sklearn()

class FraudDataPreprocessor:
    """Préprocesseur données fraud detection - VERSION CORRIGÉE"""
    
    def __init__(self, test_size=0.2, validation_size=0.2, random_state=42):
        """
        NOUVEAU: Triple split pour éviter data leakage
        - Train: 60% (training + hyperparameter optimization)
        - Validation: 20% (validation finale pour tuning)
        - Test: 20% (évaluation finale, JAMAIS touché pendant training)
        """
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        
        # Configuration chemins
        self.base_dir = Path(__file__).resolve().parent.parent
        self.data_dir = self.base_dir / "data"
        self.raw_data_path = self.data_dir / "raw" / "creditcard.csv"
        self.processed_dir = self.data_dir / "processed"
        self.models_dir = self.base_dir / "models"
        
        # Création dossiers
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
    
    def load_raw_data(self) -> pd.DataFrame:
        """Chargement données brutes avec vérifications"""
        print("📊 Chargement dataset Credit Card Fraud...")
        
        if not self.raw_data_path.exists():
            raise FileNotFoundError(
                f"Dataset non trouvé: {self.raw_data_path}\n"
                "Téléchargez creditcard.csv depuis Kaggle: "
                "https://www.kaggle.com/mlg-ulb/creditcardfraud"
            )
        
        df = pd.read_csv(self.raw_data_path)
        
        # Validation structure dataset
        expected_columns = ['Time', 'Amount', 'Class'] + [f'V{i}' for i in range(1, 29)]
        missing_cols = set(expected_columns) - set(df.columns)
        
        if missing_cols:
            raise ValueError(f"Colonnes manquantes: {missing_cols}")
        
        print(f"✅ Dataset chargé: {df.shape}")
        print(f"   • Transactions totales: {len(df):,}")
        print(f"   • Fraudes: {df['Class'].sum():,} ({df['Class'].mean()*100:.2f}%)")
        print(f"   • Features: {len(df.columns)}")
        
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Nettoyage données"""
        print("🧹 Nettoyage des données...")
        
        initial_shape = df.shape
        
        # Suppression doublons
        df_clean = df.drop_duplicates()
        duplicates_removed = initial_shape[0] - df_clean.shape[0]
        
        # Vérification valeurs manquantes
        missing_values = df_clean.isnull().sum().sum()
        
        # Vérification valeurs aberrantes (Amount négatif)
        negative_amounts = (df_clean['Amount'] < 0).sum()
        if negative_amounts > 0:
            print(f"⚠️ {negative_amounts} montants négatifs trouvés, suppression...")
            df_clean = df_clean[df_clean['Amount'] >= 0]
        
        print(f"✅ Nettoyage terminé:")
        print(f"   • Doublons supprimés: {duplicates_removed}")
        print(f"   • Valeurs manquantes: {missing_values}")
        print(f"   • Shape finale: {df_clean.shape}")
        
        return df_clean
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineering features avancé"""
        print("⚙️ Feature engineering...")
        
        df_eng = df.copy()
        
        # Features temporelles
        df_eng['Hour'] = (df_eng['Time'] / 3600) % 24
        df_eng['Day'] = (df_eng['Time'] / 86400) % 7
        df_eng['Is_Weekend'] = df_eng['Day'].isin([5, 6]).astype(int)
        df_eng['Is_Night'] = ((df_eng['Hour'] >= 22) | (df_eng['Hour'] <= 6)).astype(int)
        
        # Features montant
        df_eng['Amount_Log'] = np.log1p(df_eng['Amount'])
        df_eng['Amount_Sqrt'] = np.sqrt(df_eng['Amount'])
        
        # Seuils business (basés sur analyse exploratoire)
        df_eng['Is_High_Amount'] = (df_eng['Amount'] > 640.90).astype(int)  # 95e percentile
        df_eng['Is_Very_High_Amount'] = (df_eng['Amount'] > 2125.87).astype(int)  # 99e percentile
        df_eng['Is_Low_Amount'] = (df_eng['Amount'] == 0).astype(int)
        
        # Features interactions PCA (basées sur importance)
        # V14, V4, V12, V10, V17 sont les plus importantes
        df_eng['V14_V4_ratio'] = df_eng['V14'] / (df_eng['V4'] + 1e-8)
        df_eng['V12_V10_ratio'] = df_eng['V12'] / (df_eng['V10'] + 1e-8)
        df_eng['V17_V14_ratio'] = df_eng['V17'] / (df_eng['V14'] + 1e-8)
        
        # Agrégations features importantes
        important_features = ['V14', 'V4', 'V12', 'V10', 'V17', 'V16', 'V3', 'V11']
        df_eng['Important_Features_Sum'] = df_eng[important_features].sum(axis=1)
        df_eng['Important_Features_Mean'] = df_eng[important_features].mean(axis=1)
        df_eng['Important_Features_Std'] = df_eng[important_features].std(axis=1)
        
        # Features statistiques
        pca_features = [f'V{i}' for i in range(1, 29)]
        df_eng['PCA_Sum'] = df_eng[pca_features].sum(axis=1)
        df_eng['PCA_Mean'] = df_eng[pca_features].mean(axis=1)
        df_eng['PCA_Max'] = df_eng[pca_features].max(axis=1)
        df_eng['PCA_Min'] = df_eng[pca_features].min(axis=1)
        
        new_features = len(df_eng.columns) - len(df.columns)
        print(f"✅ Features créées: {new_features}")
        print(f"   • Features temporelles: Hour, Day, Is_Weekend, Is_Night")
        print(f"   • Features montant: Amount_Log, Amount_Sqrt, seuils")
        print(f"   • Features interactions: ratios PCA importantes")
        print(f"   • Features agrégations: sum, mean, std")
        
        return df_eng
    
    def triple_split_data(self, df: pd.DataFrame) -> tuple:
        """
        🎯 NOUVEAU: Triple split stratifié pour éviter data leakage
        
        1. Split initial: 80% (train+val) / 20% (test) 
        2. Split train+val: 75% train / 25% validation
        
        Résultat final:
        - Train: 60% des données totales
        - Validation: 20% des données totales  
        - Test: 20% des données totales (JAMAIS touché)
        """
        print("✂️ Triple split stratifié ANTI-LEAKAGE...")
        
        # Séparation features et target
        y = df['Class'].copy()
        X = df.drop(columns=['Class'])
        
        # 1er split: Séparer le test set (20%) du reste (80%)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        # 2ème split: Du reste (80%), faire train (75%) et validation (25%)
        # 75% de 80% = 60% du total pour train
        # 25% de 80% = 20% du total pour validation
        validation_size_adjusted = self.validation_size / (1 - self.test_size)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=validation_size_adjusted,
            random_state=self.random_state,
            stratify=y_temp
        )
        
        print(f"✅ Triple split terminé:")
        print(f"   • Train: {X_train.shape}, Fraude: {y_train.mean()*100:.2f}%")
        print(f"   • Validation: {X_val.shape}, Fraude: {y_val.mean()*100:.2f}%")
        print(f"   • Test: {X_test.shape}, Fraude: {y_test.mean()*100:.2f}%")
        print(f"   🔒 Test set ISOLÉ - jamais utilisé pendant training")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def fit_scaler_on_train_only(self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
        """
        🎯 CRITIQUE: Normalisation sur train uniquement pour éviter data leakage
        """
        print("📏 Normalisation ANTI-LEAKAGE (fit sur train uniquement)...")
        
        # Features à normaliser
        features_to_scale = ['Amount', 'Time', 'Amount_Log', 'Amount_Sqrt',
                           'Important_Features_Sum', 'Important_Features_Mean',
                           'PCA_Sum', 'PCA_Mean', 'PCA_Max', 'PCA_Min']
        
        # Vérification présence features
        available_features = [f for f in features_to_scale if f in X_train.columns]
        
        if available_features:
            # 🎯 CRUCIAL: Fit UNIQUEMENT sur train
            self.scaler.fit(X_train[available_features])
            
            # Transform sur tous les sets
            X_train_scaled = X_train.copy()
            X_val_scaled = X_val.copy()
            X_test_scaled = X_test.copy()
            
            X_train_scaled[available_features] = self.scaler.transform(X_train[available_features])
            X_val_scaled[available_features] = self.scaler.transform(X_val[available_features])
            X_test_scaled[available_features] = self.scaler.transform(X_test[available_features])
            
            print(f"✅ Features normalisées: {len(available_features)}")
            print(f"   🎯 Scaler fit UNIQUEMENT sur train (anti-leakage)")
            
            return X_train_scaled, X_val_scaled, X_test_scaled
        else:
            print("⚠️ Aucune feature à normaliser trouvée")
            return X_train, X_val, X_test
    
    def save_processed_data(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Sauvegarde données preprocessées - VERSION TRIPLE SPLIT"""
        print("💾 Sauvegarde données preprocessées...")
        
        # Sauvegarde datasets
        X_train.to_pickle(self.processed_dir / "X_train.pkl")
        X_val.to_pickle(self.processed_dir / "X_val.pkl")
        X_test.to_pickle(self.processed_dir / "X_test.pkl")
        y_train.to_pickle(self.processed_dir / "y_train.pkl")
        y_val.to_pickle(self.processed_dir / "y_val.pkl")
        y_test.to_pickle(self.processed_dir / "y_test.pkl")
        
        # Sauvegarde scaler
        with open(self.models_dir / "preprocessor.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"✅ Données sauvegardées dans {self.processed_dir}")
        print(f"   • Train/Val pour training: X_train.pkl, X_val.pkl")
        print(f"   • Test ISOLÉ pour évaluation finale: X_test.pkl")
        print(f"✅ Preprocessor sauvegardé dans {self.models_dir}")
    
    def get_preprocessing_summary(self, X_train, X_val, X_test, y_train, y_val, y_test) -> dict:
        """Résumé preprocessing pour logging"""
        total_samples = len(X_train) + len(X_val) + len(X_test)
        
        return {
            'total_samples': total_samples,
            'train_samples': len(X_train),
            'validation_samples': len(X_val),
            'test_samples': len(X_test),
            'features_count': X_train.shape[1],
            'train_fraud_rate': y_train.mean(),
            'validation_fraud_rate': y_val.mean(),
            'test_fraud_rate': y_test.mean(),
            'split_strategy': 'triple_split_anti_leakage',
            'preprocessing_steps': [
                'data_cleaning',
                'feature_engineering', 
                'triple_split_stratified',
                'scaling_on_train_only'
            ]
        }
    
    def run_full_pipeline(self):
        """Pipeline preprocessing complet - VERSION CORRIGÉE"""
        print("🚀 DÉMARRAGE PIPELINE PREPROCESSING ANTI-LEAKAGE")
        print("=" * 70)
        
        try:
            # 1. Chargement
            df = self.load_raw_data()
            
            # 2. Nettoyage
            df_clean = self.clean_data(df)
            
            # 3. Feature Engineering
            df_engineered = self.feature_engineering(df_clean)
            
            # 4. 🎯 NOUVEAU: Triple split AVANT normalisation
            X_train, X_val, X_test, y_train, y_val, y_test = self.triple_split_data(df_engineered)
            
            # 5. 🎯 CRITIQUE: Normalisation sur train uniquement
            X_train_scaled, X_val_scaled, X_test_scaled = self.fit_scaler_on_train_only(
                X_train, X_val, X_test
            )
            
            # 6. Sauvegarde
            self.save_processed_data(
                X_train_scaled, X_val_scaled, X_test_scaled, 
                y_train, y_val, y_test
            )
            
            # 7. Résumé
            summary = self.get_preprocessing_summary(
                X_train_scaled, X_val_scaled, X_test_scaled,
                y_train, y_val, y_test
            )
            
            print("\n🎉 PREPROCESSING ANTI-LEAKAGE TERMINÉ")
            print("=" * 70)
            print(f"📊 Train: {summary['train_samples']:,} ({summary['train_fraud_rate']*100:.2f}% fraud)")
            print(f"📊 Validation: {summary['validation_samples']:,} ({summary['validation_fraud_rate']*100:.2f}% fraud)")
            print(f"📊 Test: {summary['test_samples']:,} ({summary['test_fraud_rate']*100:.2f}% fraud)")
            print(f"🔢 Features finales: {summary['features_count']}")
            print(f"🔒 Test set COMPLÈTEMENT ISOLÉ")
            print(f"✅ Données prêtes pour training sans data leakage")
            
            return True
            
        except Exception as e:
            print(f"\n❌ ERREUR PREPROCESSING: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Exécution pipeline preprocessing"""
    print("🎯 Credit Card Fraud Detection - Preprocessing Anti-Leakage")
    
    preprocessor = FraudDataPreprocessor(
        test_size=0.2,      # 20% pour test (isolé)
        validation_size=0.2, # 20% pour validation
        random_state=42     # 60% pour train
    )
    
    success = preprocessor.run_full_pipeline()
    
    if success:
        print("\n✅ PREPROCESSING RÉUSSI!")
        print("🎯 Prochaine étape: python src/model_training.py")
    else:
        print("\n❌ PREPROCESSING ÉCHOUÉ!")
        exit(1)

if __name__ == "__main__":
    main()