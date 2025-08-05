"""
Fix Preprocessor - Credit Card Fraud Detection
Script utilitaire pour corriger/recréer le preprocessor si nécessaire
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import os

def create_preprocessor():
    """Création/Recréation du preprocessor StandardScaler"""
    
    print("🔧 Création du preprocessor StandardScaler...")
    
    # Chemins
    base_dir = Path(__file__).parent
    data_path = base_dir / "data" / "raw" / "creditcard.csv"
    models_dir = base_dir / "models"
    
    # Création dossier models si nécessaire
    os.makedirs(models_dir, exist_ok=True)
    
    # Vérification dataset
    if not data_path.exists():
        print(f"❌ Dataset non trouvé: {data_path}")
        print("📥 Téléchargez creditcard.csv depuis Kaggle et placez-le dans data/raw/")
        return False
    
    # Chargement données
    print("📊 Chargement dataset...")
    df = pd.read_csv(data_path)
    print(f"✅ Dataset chargé: {df.shape}")
    
    # Création et entraînement du scaler sur Amount et Time
    print("⚙️ Entraînement StandardScaler...")
    scaler = StandardScaler()
    
    # Fit sur les données Amount et Time combinées
    amount_time_data = np.column_stack([df['Amount'].values, df['Time'].values])
    scaler.fit(amount_time_data)
    
    # Test transformation
    scaled_data = scaler.transform(amount_time_data)
    print(f"✅ Transformation testée: {scaled_data.shape}")
    
    # Sauvegarde preprocessor
    preprocessor_path = models_dir / "preprocessor.pkl"
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"💾 Preprocessor sauvegardé: {preprocessor_path}")
    
    # Vérification fonctionnement
    print("🧪 Test de fonctionnement...")
    with open(preprocessor_path, 'rb') as f:
        loaded_scaler = pickle.load(f)
    
    # Test sur échantillon
    test_data = np.array([[100.0, 3600.0]])  # 100$ à 1h
    scaled_test = loaded_scaler.transform(test_data)
    print(f"✅ Test réussi: [100.0, 3600.0] -> {scaled_test[0]}")
    
    # Informations techniques
    print("\n📋 Informations Preprocessor:")
    print(f"   • Type: {type(loaded_scaler).__name__}")
    print(f"   • Features: ['Amount', 'Time']")
    print(f"   • Mean: {loaded_scaler.mean_}")
    print(f"   • Scale: {loaded_scaler.scale_}")
    
    return True

def verify_preprocessor():
    """Vérification preprocessor existant"""
    
    print("🔍 Vérification preprocessor existant...")
    
    base_dir = Path(__file__).parent
    preprocessor_path = base_dir / "models" / "preprocessor.pkl"
    
    if not preprocessor_path.exists():
        print("❌ Preprocessor non trouvé")
        return False
    
    try:
        with open(preprocessor_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Test basique
        test_data = np.array([[0.0, 0.0], [100.0, 3600.0]])
        result = scaler.transform(test_data)
        
        print("✅ Preprocessor fonctionne correctement")
        print(f"   • Type: {type(scaler).__name__}")
        print(f"   • Shape entrée: {test_data.shape}")
        print(f"   • Shape sortie: {result.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur preprocessor: {e}")
        return False

def main():
    """Fonction principale"""
    print("=" * 60)
    print("🔧 FIX PREPROCESSOR - CREDIT CARD FRAUD DETECTION")
    print("=" * 60)
    
    # Vérification d'abord
    if verify_preprocessor():
        print("\n✅ Preprocessor déjà fonctionnel!")
        print("💡 Utilisez --force pour recréer")
        return
    
    # Création si nécessaire
    print("\n🔄 Création nouveau preprocessor...")
    success = create_preprocessor()
    
    if success:
        print("\n🎉 Preprocessor créé avec succès!")
        print("✅ Vous pouvez maintenant lancer le training")
    else:
        print("\n❌ Échec création preprocessor")
        print("🔍 Vérifiez que creditcard.csv est dans data/raw/")

if __name__ == "__main__":
    import sys
    
    # Support argument --force
    if "--force" in sys.argv:
        print("🔄 Mode force: Recréation du preprocessor...")
        create_preprocessor()
    else:
        main()