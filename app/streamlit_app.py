"""
Streamlit Demo App - Credit Card Fraud Detection
Interface interactive pour démonstration du modèle ML
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import time
from pathlib import Path
import sys

# Ajout du dossier parent au path pour imports
# Note: Cette approche peut parfois être fragile. 
# Pour un projet plus grand, envisagez de l'installer en tant que package.
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Il est préférable d'importer après avoir modifié le path
from app.utils import (
    load_sample_transactions, get_dataset_stats, check_api_connection,
    predict_single_transaction, predict_batch_transactions,
    create_risk_gauge, display_alert_box, format_currency,
    format_percentage, format_duration
)

# Configuration page
st.set_page_config(
    page_title="Fraud Detection Demo",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}

.metric-card {
    background: white;
    padding: 1rem;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    border-left: 4px solid #667eea;
}

.fraud-alert {
    background: linear-gradient(90deg, #ff6b6b 0%, #feca57 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    font-size: 1.2rem;
    font-weight: bold;
}

.safe-alert {
    background: linear-gradient(90deg, #26de81 0%, #20bf6b 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    font-size: 1.2rem;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

class FraudDetectionApp:
    """Application Streamlit pour Fraud Detection"""
    
    def __init__(self):
        self.api_base = "http://localhost:8000"
        self.sample_data = None
        self.dataset_stats = None
        
    def initialize_app(self):
        """Initialisation application"""
        # Header principal
        st.markdown("""
        <div class="main-header">
            <h1>🛡️ Credit Card Fraud Detection</h1>
            <p>Système de détection ML haute performance • ROI 15:1 • F1-Score 86%</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Vérification API
        self.check_api_status()
        
        # Le reste de tes méthodes pour construire l'UI sera appelé ici
        # Par exemple: self.display_prediction_interface()
        # self.display_dashboard()
    
    def check_api_status(self):
        """Vérification statut API"""
        with st.sidebar:
            st.subheader("🔧 Status Système")
            
            if check_api_connection(self.api_base):
                st.success("🟢 API Opérationnelle")
                
                # Informations modèle
                try:
                    response = requests.get(f"{self.api_base}/model/info")
                    if response.status_code == 200:
                        model_info = response.json()
                        st.info(f"📊 Modèle: {model_info['model_type']}")
                        st.info(f"🎯 F1-Score: {model_info['performance_metrics']['f1_score']:.3f}")
                except Exception as e:
                    st.warning(f"Impossible de charger les infos du modèle : {e}")
            else:
                st.error("🔴 API Non Connectée")
                st.warning("Veuillez démarrer le serveur API dans un terminal avec : uvicorn src.api:app --reload")

    # Tu ajouteras ici le reste de tes méthodes de la classe (load_data, etc.)

# === LA CORRECTION EST ICI ===
# Ce bloc de code est le point d'entrée qui lance l'application.
if __name__ == "__main__":
    app = FraudDetectionApp()
    app.initialize_app()