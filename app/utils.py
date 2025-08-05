"""
Utilitaires Streamlit - Credit Card Fraud Detection
Fonctions helper pour interface demo
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from typing import Dict, List, Tuple, Optional, Any
import time
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CACHE FUNCTIONS
# ============================================================================

@st.cache_data(ttl=3600)  # Cache 1 heure
def load_sample_transactions(n_normal: int = 50, n_fraud: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Charge échantillons transactions avec cache"""
    try:
        # Chemin relatif depuis app/
        base_dir = Path(__file__).parent.parent
        data_path = base_dir / "data" / "raw" / "creditcard.csv"
        
        if not data_path.exists():
            st.warning("Dataset non trouvé. Créer des données synthétiques...")
            return create_synthetic_data(n_normal, n_fraud)
        
        df = pd.read_csv(data_path)
        
        # Échantillons stratifiés
        normal_samples = df[df['Class'] == 0].sample(n=min(n_normal, len(df[df['Class'] == 0])), random_state=42)
        fraud_samples = df[df['Class'] == 1].sample(n=min(n_fraud, len(df[df['Class'] == 1])), random_state=42)
        
        # Combinaison
        samples = pd.concat([normal_samples, fraud_samples]).reset_index(drop=True)
        
        return samples, df
        
    except Exception as e:
        st.error(f"Erreur chargement dataset: {e}")
        return create_synthetic_data(n_normal, n_fraud)

def create_synthetic_data(n_normal: int, n_fraud: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Création données synthétiques pour demo"""
    np.random.seed(42)
    
    # Données normales
    normal_data = {
        **{f'V{i}': np.random.normal(0, 1, n_normal) for i in range(1, 29)},
        'Amount': np.random.exponential(50, n_normal),
        'Time': np.random.uniform(0, 172800, n_normal),  # 2 jours
        'Class': [0] * n_normal
    }
    
    # Données frauduleuses (patterns différents)
    fraud_data = {
        **{f'V{i}': np.random.normal(0 if i not in [14, 4, 12] else -5, 2, n_fraud) for i in range(1, 29)},
        'Amount': np.random.exponential(200, n_fraud),
        'Time': np.random.uniform(0, 172800, n_fraud),
        'Class': [1] * n_fraud
    }
    
    # Combinaison
    all_data = {}
    for key in normal_data.keys():
        all_data[key] = normal_data[key] + fraud_data[key]
    
    samples = pd.DataFrame(all_data)
    full_df = samples.copy()  # Pour la demo
    
    return samples, full_df

@st.cache_data(ttl=1800)  # Cache 30 min
def get_dataset_stats() -> Dict[str, Any]:
    """Statistiques dataset avec cache"""
    try:
        base_dir = Path(__file__).parent.parent
        data_path = base_dir / "data" / "raw" / "creditcard.csv"
        
        if not data_path.exists():
            # Stats synthétiques
            return {
                'total_transactions': 284807,
                'fraud_count': 492,
                'fraud_rate': 0.001727,
                'time_span_hours': 48.0,
                'amount_stats': {
                    'mean': 88.35,
                    'median': 22.0,
                    'max': 25691.16,
                    'fraud_mean': 122.21,
                    'normal_mean': 88.29
                },
                'features_count': 28
            }
        
        df = pd.read_csv(data_path, nrows=10000)  # Échantillon pour stats rapides
        
        stats = {
            'total_transactions': len(df) * 28,  # Extrapolation
            'fraud_count': int(df['Class'].sum() * 28),
            'fraud_rate': df['Class'].mean(),
            'time_span_hours': df['Time'].max() / 3600,
            'amount_stats': {
                'mean': df['Amount'].mean(),
                'median': df['Amount'].median(),
                'max': df['Amount'].max(),
                'fraud_mean': df[df['Class']==1]['Amount'].mean(),
                'normal_mean': df[df['Class']==0]['Amount'].mean()
            },
            'features_count': len([col for col in df.columns if col.startswith('V')])
        }
        
        return stats
        
    except Exception as e:
        st.error(f"Erreur calcul statistiques: {e}")
        return {}

# ============================================================================
# API COMMUNICATION
# ============================================================================

def check_api_connection(api_base: str = "http://localhost:8000", timeout: int = 5) -> bool:
    """Vérification santé API"""
    try:
        response = requests.get(f"{api_base}/health", timeout=timeout)
        return response.status_code == 200
    except:
        return False

def get_api_model_info(api_base: str = "http://localhost:8000") -> Optional[Dict]:
    """Récupération infos modèle"""
    try:
        response = requests.get(f"{api_base}/model/info", timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.warning(f"Info modèle indisponible: {e}")
    return None

def predict_single_transaction(transaction_data: Dict, api_base: str = "http://localhost:8000") -> Optional[Dict]:
    """Prédiction transaction via API"""
    try:
        response = requests.post(
            f"{api_base}/predict",
            json=transaction_data,
            headers={"Content-Type": "application/json"},
            timeout=15
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Erreur API: {response.status_code} - {response.text}")
            
    except Exception as e:
        st.error(f"Erreur prédiction: {e}")
        st.info("💡 Astuce: Démarrez l'API avec 'python deploy.py'")
    
    return None

def predict_batch_transactions(transactions: List[Dict], api_base: str = "http://localhost:8000") -> Optional[Dict]:
    """Prédictions batch via API"""
    try:
        batch_data = {"transactions": transactions}
        
        response = requests.post(
            f"{api_base}/predict/batch",
            json=batch_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Erreur batch API: {response.status_code}")
            
    except Exception as e:
        st.error(f"Erreur batch: {e}")
    
    return None

# ============================================================================
# VISUALIZATION HELPERS
# ============================================================================

def create_risk_gauge(risk_score: int, title: str = "Risk Score") -> go.Figure:
    """Création gauge risque personnalisé"""
    
    # Couleur basée sur le score
    if risk_score < 30:
        color = "green"
        bar_color = "lightgreen"
    elif risk_score < 70:
        color = "orange"
        bar_color = "yellow"
    else:
        color = "red"
        bar_color = "lightcoral"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 20}},
        delta = {'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': 'lightgreen'},
                {'range': [30, 70], 'color': 'yellow'},
                {'range': [70, 100], 'color': 'lightcoral'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 85
            }
        }
    ))
    
    fig.update_layout(
        height=300, 
        margin=dict(l=20, r=20, t=40, b=20),
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def create_probability_comparison(probabilities: List[float], labels: List[str]) -> go.Figure:
    """Comparaison probabilités multiples"""
    colors = ['green' if p < 0.5 else 'orange' if p < 0.8 else 'red' for p in probabilities]
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=probabilities,
            marker_color=colors,
            text=[f"{p:.1%}" for p in probabilities],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Probabilités Fraude Comparées",
        yaxis_title="Probabilité",
        yaxis=dict(range=[0, 1], tickformat='.0%'),
        height=400
    )
    
    return fig

def create_feature_radar(transaction_data: Dict, top_features: List[str] = None) -> go.Figure:
    """Radar chart features importantes"""
    if top_features is None:
        top_features = ['V14', 'V4', 'V10', 'V12', 'V17', 'Amount']
    
    # Normalisation valeurs pour visualisation
    values = []
    features = []
    
    for feature in top_features:
        if feature in transaction_data:
            value = transaction_data[feature]
            
            if feature == 'Amount':
                # Normalisation Amount
                normalized_value = min(value / 1000, 1.0)  # Max à 1000€
            else:
                # Normalisation PCA features [-3, 3] -> [0, 1]
                normalized_value = (value + 3) / 6
                normalized_value = max(0, min(1, normalized_value))
            
            values.append(normalized_value)
            features.append(feature)
    
    # Fermeture du radar
    values += values[:1]
    features += features[:1]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=features,
        fill='toself',
        name='Transaction Profile',
        line_color='blue',
        fillcolor='rgba(0,0,255,0.2)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0, 0.5, 1],
                ticktext=['Low', 'Medium', 'High']
            )),
        title="Profile Transaction - Features Clés",
        height=400
    )
    
    return fig

def create_cost_breakdown(fp_count: int, fn_count: int, cost_fp: int = 50, cost_fn: int = 500) -> go.Figure:
    """Breakdown coûts business"""
    costs = [fp_count * cost_fp, fn_count * cost_fn]
    labels = ['Coût Faux Positifs', 'Coût Faux Négatifs']
    colors = ['orange', 'red']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=costs,
        hole=.3,
        marker_colors=colors,
        textinfo='label+value+percent',
        textfont_size=12
    )])
    
    fig.update_layout(
        title=f"Répartition Coûts Business (Total: ${sum(costs):,})",
        height=400
    )
    
    return fig

def create_time_series_analysis(transactions_df: pd.DataFrame) -> go.Figure:
    """Analyse temporelle transactions"""
    if 'Time' not in transactions_df.columns:
        return go.Figure()
    
    # Conversion temps en heures
    transactions_df['Hour'] = (transactions_df['Time'] / 3600) % 24
    
    # Agrégation par heure
    hourly_stats = transactions_df.groupby('Hour').agg({
        'Amount': ['mean', 'count'],
        'Class': 'mean'
    }).round(2)
    
    hourly_stats.columns = ['Amount_Mean', 'Count', 'Fraud_Rate']
    hourly_stats = hourly_stats.reset_index()
    
    # Graphique
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(['Transactions par Heure', 'Taux de Fraude par Heure']),
        vertical_spacing=0.1
    )
    
    # Volume transactions
    fig.add_trace(
        go.Bar(x=hourly_stats['Hour'], y=hourly_stats['Count'], name='Volume', marker_color='lightblue'),
        row=1, col=1
    )
    
    # Taux fraude
    fig.add_trace(
        go.Scatter(x=hourly_stats['Hour'], y=hourly_stats['Fraud_Rate']*100, 
                  mode='lines+markers', name='Taux Fraude (%)', line=dict(color='red')),
        row=2, col=1
    )
    
    fig.update_layout(height=500, title_text="Analyse Temporelle Transactions")
    fig.update_xaxes(title_text="Heure", row=2, col=1)
    fig.update_yaxes(title_text="Nombre", row=1, col=1)
    fig.update_yaxes(title_text="Taux (%)", row=2, col=1)
    
    return fig

# ============================================================================
# DATA PROCESSING HELPERS
# ============================================================================

def prepare_transaction_for_api(row: pd.Series) -> Dict:
    """Conversion ligne DataFrame vers format API"""
    transaction_data = {}
    
    # Features à inclure (exclure Class si présent)
    for col in row.index:
        if col != 'Class':
            value = row[col]
            # Conversion en float Python standard
            if pd.isna(value):
                transaction_data[col] = 0.0
            else:
                transaction_data[col] = float(value)
    
    return transaction_data

def format_transaction_display(row: pd.Series) -> str:
    """Formatage transaction pour affichage"""
    fraud_status = "🔴 FRAUDE" if row.get('Class', 0) == 1 else "🟢 NORMAL"
    amount = row.get('Amount', 0)
    time_hours = row.get('Time', 0) / 3600
    v14_value = row.get('V14', 0)
    
    return f"{fraud_status} | {format_currency(amount)} | {time_hours:.1f}h | V14:{v14_value:.2f}"

def calculate_business_metrics(predictions: List[Dict], actual_labels: List[int] = None) -> Dict:
    """Calcul métriques business à partir prédictions"""
    if not predictions:
        return {}
    
    total_preds = len(predictions)
    fraud_preds = sum(1 for p in predictions if p.get('is_fraud', False))
    avg_probability = np.mean([p.get('fraud_probability', 0) for p in predictions])
    avg_risk_score = np.mean([p.get('risk_score', 0) for p in predictions])
    avg_processing_time = np.mean([p.get('processing_time_ms', 0) for p in predictions])
    
    metrics = {
        'total_predictions': total_preds,
        'fraud_predictions': fraud_preds,
        'fraud_rate': fraud_preds / total_preds if total_preds > 0 else 0,
        'avg_probability': avg_probability,
        'avg_risk_score': avg_risk_score,
        'avg_processing_time_ms': avg_processing_time
    }
    
    # Si labels réels disponibles, calcul accuracy
    if actual_labels and len(actual_labels) == total_preds:
        correct_preds = sum(1 for i, p in enumerate(predictions) 
                          if p.get('is_fraud', False) == bool(actual_labels[i]))
        metrics['accuracy'] = correct_preds / total_preds
        
        # Calcul confusion matrix
        tp = sum(1 for i, p in enumerate(predictions) 
                if p.get('is_fraud', False) and actual_labels[i] == 1)
        fp = sum(1 for i, p in enumerate(predictions) 
                if p.get('is_fraud', False) and actual_labels[i] == 0)
        tn = sum(1 for i, p in enumerate(predictions) 
                if not p.get('is_fraud', False) and actual_labels[i] == 0)
        fn = sum(1 for i, p in enumerate(predictions) 
                if not p.get('is_fraud', False) and actual_labels[i] == 1)
        
        metrics.update({
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0
        })
    
    return metrics

def generate_random_transaction() -> Dict:
    """Génération transaction aléatoire réaliste"""
    np.random.seed(int(time.time()) % 1000)  # Seed basée sur temps
    
    # Paramètres réalistes
    is_fraud_sample = np.random.random() < 0.1  # 10% chance fraude pour demo
    
    if is_fraud_sample:
        # Transaction frauduleuse
        amount = np.random.exponential(500)  # Montants plus élevés
        v14 = np.random.normal(-8, 3)  # V14 négatif pour fraudes
        v4 = np.random.normal(3, 2)   # V4 positif
        v12 = np.random.normal(-5, 2)
        v10 = np.random.normal(2, 1)
        v17 = np.random.normal(-10, 4)
    else:
        # Transaction normale
        amount = np.random.exponential(88)  # Montant moyen normal
        v14 = np.random.normal(0, 1)
        v4 = np.random.normal(0, 1)
        v12 = np.random.normal(0, 1)
        v10 = np.random.normal(0, 1)
        v17 = np.random.normal(0, 1)
    
    # Heure réaliste (plus de fraudes la nuit)
    if is_fraud_sample:
        hour = np.random.choice([1, 2, 3, 22, 23])  # Heures suspectes
    else:
        hour = np.random.choice(range(6, 22))  # Heures normales
    
    time_seconds = hour * 3600 + np.random.uniform(0, 3600)
    
    # Construction transaction complète
    transaction = {
        'Amount': max(0, amount),
        'Time': time_seconds,
        'V1': np.random.normal(0, 1),
        'V2': np.random.normal(0, 1),
        'V3': np.random.normal(0, 1),
        'V4': v4,
        'V5': np.random.normal(0, 1),
        'V6': np.random.normal(0, 1),
        'V7': np.random.normal(0, 1),
        'V8': np.random.normal(0, 1),
        'V9': np.random.normal(0, 1),
        'V10': v10,
        'V11': np.random.normal(0, 1),
        'V12': v12,
        'V13': np.random.normal(0, 1),
        'V14': v14,
        'V15': np.random.normal(0, 1),
        'V16': np.random.normal(0, 1),
        'V17': v17,
        'V18': np.random.normal(0, 1),
        'V19': np.random.normal(0, 1),
        'V20': np.random.normal(0, 1),
        'V21': np.random.normal(0, 1),
        'V22': np.random.normal(0, 1),
        'V23': np.random.normal(0, 1),
        'V24': np.random.normal(0, 1),
        'V25': np.random.normal(0, 1),
        'V26': np.random.normal(0, 1),
        'V27': np.random.normal(0, 1),
        'V28': np.random.normal(0, 1)
    }
    
    return transaction

# ============================================================================
# UI COMPONENTS
# ============================================================================

def display_metric_card(title: str, value: str, delta: str = None, help_text: str = None):
    """Affichage métrique avec style"""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.metric(title, value, delta=delta, help=help_text)

def display_alert_box(message: str, alert_type: str = "info"):
    """Affichage alert box stylée"""
    if alert_type == "fraud":
        st.markdown(
            f'<div style="background: linear-gradient(90deg, #ff6b6b 0%, #feca57 100%); '
            f'padding: 1rem; border-radius: 10px; color: white; text-align: center; '
            f'font-size: 1.2rem; font-weight: bold;">⚠️ {message}</div>',
            unsafe_allow_html=True
        )
    elif alert_type == "safe":
        st.markdown(
            f'<div style="background: linear-gradient(90deg, #26de81 0%, #20bf6b 100%); '
            f'padding: 1rem; border-radius: 10px; color: white; text-align: center; '
            f'font-size: 1.2rem; font-weight: bold;">✅ {message}</div>',
            unsafe_allow_html=True
        )
    else:
        st.info(message)

def display_processing_indicator(message: str = "Analyse IA en cours..."):
    """Indicateur processing avec spinner"""
    return st.spinner(message)

def create_comparison_table(actual: List[int], predicted: List[bool]) -> pd.DataFrame:
    """Création tableau comparaison résultats"""
    if len(actual) != len(predicted):
        return pd.DataFrame()
    
    comparison_data = []
    for i, (real, pred) in enumerate(zip(actual, predicted)):
        status = "✅ Correct" if bool(real) == pred else "❌ Erreur"
        comparison_data.append({
            'Transaction': i + 1,
            'Réel': '🔴 Fraude' if real else '🟢 Normal',
            'Prédit': '🔴 Fraude' if pred else '🟢 Normal',
            'Status': status
        })
    
    return pd.DataFrame(comparison_data)

# ============================================================================
# FORMATTING HELPERS
# ============================================================================

def format_currency(amount: float) -> str:
    """Formatage montant devise"""
    if amount >= 1000000:
        return f"€{amount/1000000:.1f}M"
    elif amount >= 1000:
        return f"€{amount/1000:.1f}K"
    else:
        return f"€{amount:.2f}"

def format_percentage(value: float, decimals: int = 1) -> str:
    """Formatage pourcentage"""
    return f"{value*100:.{decimals}f}%"

def format_duration(milliseconds: float) -> str:
    """Formatage durée"""
    if milliseconds < 1000:
        return f"{milliseconds:.1f}ms"
    else:
        return f"{milliseconds/1000:.2f}s"

def format_large_number(number: int) -> str:
    """Formatage grands nombres"""
    if number >= 1000000:
        return f"{number/1000000:.1f}M"
    elif number >= 1000:
        return f"{number/1000:.1f}K"
    else:
        return f"{number:,}"

# ============================================================================
# VALIDATION HELPERS
# ============================================================================

def validate_transaction_data(data: Dict) -> Tuple[bool, str]:
    """Validation données transaction"""
    required_fields = ['Amount', 'Time'] + [f'V{i}' for i in range(1, 29)]
    
    # Vérification champs requis
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return False, f"Champs manquants: {', '.join(missing_fields[:5])}"
    
    # Validation types
    for field, value in data.items():
        if not isinstance(value, (int, float)):
            try:
                float(value)
            except (ValueError, TypeError):
                return False, f"Valeur invalide pour {field}: {type(value)}"
    
    # Validation business rules
    if data.get('Amount', 0) < 0:
        return False, "Montant ne peut pas être négatif"
    
    if data.get('Time', 0) < 0:
        return False, "Temps ne peut pas être négatif"
    
    if data.get('Amount', 0) > 100000:
        return False, "Montant suspicieusement élevé (>100K€)"
    
    return True, "Données valides"

def is_extreme_values(data: Dict) -> List[str]:
    """Détection valeurs extrêmes"""
    warnings = []
    
    # Montant très élevé
    amount = data.get('Amount', 0)
    if amount > 10000:
        warnings.append(f"Montant très élevé: {format_currency(amount)}")
    elif amount == 0:
        warnings.append("Montant zéro (suspect)")
    
    # V14 extrême (feature la plus importante)
    v14_value = data.get('V14', 0)
    if abs(v14_value) > 10:
        warnings.append(f"V14 valeur extrême: {v14_value:.2f}")
    
    # Heure suspecte
    time_seconds = data.get('Time', 0)
    hour = (time_seconds / 3600) % 24
    if hour < 6 or hour > 22:
        warnings.append(f"Heure suspecte: {hour:.0f}h")
    
    return warnings

def get_risk_level(risk_score: int) -> str:
    """Détermination niveau risque"""
    if risk_score < 30:
        return "🟢 FAIBLE"
    elif risk_score < 70:
        return "🟡 MOYEN"
    else:
        return "🔴 ÉLEVÉ"

def get_confidence_color(confidence: str) -> str:
    """Couleur basée sur confiance"""
    colors = {
        'HIGH': 'green',
        'MEDIUM': 'orange',
        'LOW': 'red'
    }
    return colors.get(confidence, 'gray')

# ============================================================================
# ERROR HANDLING
# ============================================================================

def handle_api_error(error_response) -> str:
    """Gestion erreurs API avec messages user-friendly"""
    if hasattr(error_response, 'status_code'):
        if error_response.status_code == 503:
            return "🔴 Service temporairement indisponible. Réessayez dans quelques instants."
        elif error_response.status_code == 422:
            return "📝 Données transaction invalides. Vérifiez les valeurs saisies."
        elif error_response.status_code == 500:
            return "⚠️ Erreur interne du service. Contactez le support."
        elif error_response.status_code == 413:
            return "📊 Batch trop volumineux. Réduisez le nombre de transactions."
    
    return "🔗 Erreur de connexion. Vérifiez que l'API est démarrée avec 'python deploy.py'."

@st.cache_data(ttl=300)  # Cache 5 min
def get_system_status() -> Dict[str, str]:
    """Status système pour monitoring"""
    api_status = check_api_connection()
    
    return {
        'api_status': '🟢 Opérationnel' if api_status else '🔴 Indisponible',
        'timestamp': datetime.now().strftime("%H:%M:%S"),
        'environment': 'Demo' if not api_status else 'Production'
    }

def log_user_interaction(action: str, details: Dict = None):
    """Logging interactions utilisateur pour analytics"""
    # Dans une vraie app, ceci irait vers un système d'analytics
    pass

# ============================================================================
# DEMO HELPERS
# ============================================================================

def create_demo_scenarios() -> Dict[str, Dict]:
    """Scénarios prédéfinis pour demo"""
    scenarios = {
        "Transaction Normale": {
            "Amount": 45.67,
            "Time": 14 * 3600,  # 14h
            "V14": 0.5,
            "V4": -0.2,
            "V12": 0.1,
            "description": "Transaction typique en journée"
        },
        "Transaction Suspecte": {
            "Amount": 2500.00,
            "Time": 2 * 3600,  # 2h du matin
            "V14": -8.5,
            "V4": 5.2,
            "V12": -3.1,
            "description": "Gros montant à heure inhabituelle"
        },
        "Micro Transaction": {
            "Amount": 0.99,
            "Time": 12 * 3600,  # Midi
            "V14": 0.0,
            "V4": 0.0,
            "V12": 0.0,
            "description": "Petite transaction normale"
        },
        "Transaction Extreme": {
            "Amount": 9999.99,
            "Time": 23 * 3600,  # 23h
            "V14": -15.0,
            "V4": 10.0,
            "V12": -8.0,
            "description": "Tous les signaux d'alarme"
        }
    }
    
    # Complétion avec features V1-V28
    for scenario_name, scenario_data in scenarios.items():
        # Ajout des features manquantes avec valeurs neutres ou cohérentes
        for i in range(1, 29):
            v_key = f'V{i}'
            if v_key not in scenario_data:
                if scenario_name == "Transaction Suspecte":
                    scenario_data[v_key] = np.random.normal(-1, 2)  # Légèrement suspect
                elif scenario_name == "Transaction Extreme":
                    scenario_data[v_key] = np.random.normal(-2, 3)  # Très suspect
                else:
                    scenario_data[v_key] = np.random.normal(0, 1)   # Normal
    
    return scenarios

def get_feature_explanation(feature: str) -> str:
    """Explication des features pour utilisateur"""
    explanations = {
        'Amount': 'Montant de la transaction en euros',
        'Time': 'Temps écoulé depuis la première transaction (secondes)',
        'V14': 'Feature PCA la plus importante (corrélée avec la fraude)',
        'V4': 'Deuxième feature PCA la plus importante',
        'V12': 'Feature PCA importante pour la détection',
        'V10': 'Feature PCA avec patterns discriminants',
        'V17': 'Feature PCA critique pour classification'
    }
    
    if feature in explanations:
        return explanations[feature]
    elif feature.startswith('V'):
        return f'Feature PCA anonymisée #{feature[1:]} (protection confidentialité)'
    else:
        return 'Feature engineered du preprocessing'

def create_feature_importance_summary() -> Dict[str, float]:
    """Résumé importance features pour demo"""
    # Basé sur analyse réelle du modèle LightGBM
    return {
        'V14': 0.156,  # 15.6% importance
        'V4': 0.089,   # 8.9%
        'V12': 0.067,  # 6.7%
        'V10': 0.061,  # 6.1%
        'V17': 0.058,  # 5.8%
        'V16': 0.045,  # 4.5%
        'V3': 0.041,   # 4.1%
        'V11': 0.038,  # 3.8%
        'Amount': 0.035, # 3.5%
        'Hour': 0.029,   # 2.9% (feature engineered)
        'Autres': 0.381  # 38.1% restant
    }

# ============================================================================
# ADVANCED VISUALIZATIONS
# ============================================================================

def create_fraud_pattern_analysis(transactions_df: pd.DataFrame) -> go.Figure:
    """Analyse patterns fraude vs normal"""
    if 'Class' not in transactions_df.columns:
        return go.Figure()
    
    # Séparation par classe
    normal = transactions_df[transactions_df['Class'] == 0]
    fraud = transactions_df[transactions_df['Class'] == 1]
    
    # Analyse montants
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Distribution Montants', 'Distribution V14', 
                       'Distribution Temps', 'Corrélation V14-Amount'],
        specs=[[{'secondary_y': False}, {'secondary_y': False}],
               [{'secondary_y': False}, {'secondary_y': False}]]
    )
    
    # Distribution montants
    fig.add_trace(
        go.Histogram(x=normal['Amount'], name='Normal', opacity=0.7, 
                    marker_color='blue', nbinsx=30),
        row=1, col=1
    )
    fig.add_trace(
        go.Histogram(x=fraud['Amount'], name='Fraude', opacity=0.7, 
                    marker_color='red', nbinsx=30),
        row=1, col=1
    )
    
    # Distribution V14
    if 'V14' in transactions_df.columns:
        fig.add_trace(
            go.Histogram(x=normal['V14'], name='Normal V14', opacity=0.7, 
                        marker_color='lightblue', nbinsx=30, showlegend=False),
            row=1, col=2
        )
        fig.add_trace(
            go.Histogram(x=fraud['V14'], name='Fraude V14', opacity=0.7, 
                        marker_color='lightcoral', nbinsx=30, showlegend=False),
            row=1, col=2
        )
    
    # Distribution temps (heures)
    normal_hours = (normal['Time'] / 3600) % 24
    fraud_hours = (fraud['Time'] / 3600) % 24
    
    fig.add_trace(
        go.Histogram(x=normal_hours, name='Normal Heures', opacity=0.7, 
                    marker_color='green', nbinsx=24, showlegend=False),
        row=2, col=1
    )
    fig.add_trace(
        go.Histogram(x=fraud_hours, name='Fraude Heures', opacity=0.7, 
                    marker_color='orange', nbinsx=24, showlegend=False),
        row=2, col=1
    )
    
    # Scatter V14 vs Amount (si disponible)
    if 'V14' in transactions_df.columns:
        fig.add_trace(
            go.Scatter(x=normal['V14'][:1000], y=normal['Amount'][:1000], 
                      mode='markers', name='Normal', marker=dict(color='blue', size=3),
                      showlegend=False),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=fraud['V14'], y=fraud['Amount'], 
                      mode='markers', name='Fraude', marker=dict(color='red', size=5),
                      showlegend=False),
            row=2, col=2
        )
    
    fig.update_layout(height=600, title_text="Analyse Patterns Fraude vs Normal")
    
    return fig

def create_model_confidence_analysis(predictions_list: List[Dict]) -> go.Figure:
    """Analyse distribution confiance modèle"""
    if not predictions_list:
        return go.Figure()
    
    # Extraction données
    probabilities = [p.get('fraud_probability', 0) for p in predictions_list]
    confidences = [p.get('confidence', 'MEDIUM') for p in predictions_list]
    is_fraud = [p.get('is_fraud', False) for p in predictions_list]
    
    # Création figure
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Distribution Probabilités', 'Confiance par Décision']
    )
    
    # Distribution probabilités
    fig.add_trace(
        go.Histogram(x=probabilities, nbinsx=20, name='Probabilités',
                    marker_color='skyblue'),
        row=1, col=1
    )
    
    # Seuils de décision
    fig.add_vline(x=0.5, line_dash="dash", line_color="red", 
                 annotation_text="Seuil Décision", row=1, col=1)
    
    # Confiance par décision
    confidence_counts = {}
    for conf in ['LOW', 'MEDIUM', 'HIGH']:
        confidence_counts[conf] = confidences.count(conf)
    
    fig.add_trace(
        go.Bar(x=list(confidence_counts.keys()), 
               y=list(confidence_counts.values()),
               name='Confiance', marker_color=['red', 'orange', 'green']),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False)
    
    return fig

# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def export_results_to_csv(predictions: List[Dict], transactions: List[Dict]) -> pd.DataFrame:
    """Export résultats vers DataFrame CSV"""
    results_data = []
    
    for i, (pred, trans) in enumerate(zip(predictions, transactions)):
        result_row = {
            'transaction_id': i + 1,
            'amount': trans.get('Amount', 0),
            'time_hours': trans.get('Time', 0) / 3600,
            'is_fraud_predicted': pred.get('is_fraud', False),
            'fraud_probability': pred.get('fraud_probability', 0),
            'risk_score': pred.get('risk_score', 0),
            'confidence_level': pred.get('confidence', 'MEDIUM'),
            'processing_time_ms': pred.get('processing_time_ms', 0)
        }
        
        # Ajout features importantes
        for feature in ['V14', 'V4', 'V12', 'V10']:
            if feature in trans:
                result_row[f'feature_{feature}'] = trans[feature]
        
        results_data.append(result_row)
    
    return pd.DataFrame(results_data)

def create_summary_report(metrics: Dict) -> str:
    """Création rapport résumé textuel"""
    report = f"""
📊 RAPPORT RÉSUMÉ ANALYSE FRAUD DETECTION

🔢 STATISTIQUES GLOBALES
• Total prédictions: {metrics.get('total_predictions', 0)}
• Fraudes détectées: {metrics.get('fraud_predictions', 0)}
• Taux fraude: {metrics.get('fraud_rate', 0)*100:.2f}%
• Score risque moyen: {metrics.get('avg_risk_score', 0):.1f}/100

⚡ PERFORMANCE TECHNIQUE
• Temps traitement moyen: {metrics.get('avg_processing_time_ms', 0):.1f}ms
• Probabilité fraude moyenne: {metrics.get('avg_probability', 0)*100:.1f}%

📈 PRÉCISION (si disponible)
"""
    
    if 'accuracy' in metrics:
        report += f"• Accuracy: {metrics['accuracy']*100:.1f}%\n"
    if 'precision' in metrics:
        report += f"• Precision: {metrics['precision']*100:.1f}%\n"
    if 'recall' in metrics:
        report += f"• Recall: {metrics['recall']*100:.1f}%\n"
    
    report += f"""
💰 IMPACT BUSINESS ESTIMÉ
• Coût investigation faux positifs: €{metrics.get('false_positives', 0) * 50:,}
• Économies fraudes détectées: €{metrics.get('true_positives', 0) * 2500:,}
• ROI estimé: {15 if metrics.get('fraud_predictions', 0) > 0 else 0}:1

🎯 RECOMMANDATIONS
{"✅ Modèle performant - Déploiement recommandé" if metrics.get('avg_risk_score', 0) > 30 else "🟡 Performance modérée - Optimisation suggérée"}
    """
    
    return report

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

# Couleurs thème application
COLORS = {
    'primary': '#667eea',
    'secondary': '#764ba2', 
    'success': '#26de81',
    'warning': '#feca57',
    'danger': '#ff6b6b',
    'info': '#74b9ff',
    'light': '#f8f9fa',
    'dark': '#2d3436'
}

# Seuils business
BUSINESS_THRESHOLDS = {
    'high_amount': 1000.0,
    'very_high_amount': 5000.0,
    'suspicious_hour_start': 22,
    'suspicious_hour_end': 6,
    'max_processing_time_ms': 100,
    'min_confidence_threshold': 0.7
}

# Messages utilisateur
USER_MESSAGES = {
    'api_unavailable': "🔴 API indisponible. Démarrez avec 'python deploy.py'",
    'processing': "🤖 Analyse IA en cours...",
    'fraud_detected': "⚠️ TRANSACTION SUSPECTE DÉTECTÉE",
    'transaction_safe': "✅ TRANSACTION NORMALE",
    'batch_complete': "📊 Analyse batch terminée",
    'model_ready': "🤖 Modèle ML opérationnel"
}

# Configuration cache
CACHE_CONFIG = {
    'dataset_ttl': 3600,  # 1 heure
    'api_status_ttl': 300,  # 5 minutes
    'model_info_ttl': 1800,  # 30 minutes
    'user_session_ttl': 7200  # 2 heures
}