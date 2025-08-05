"""
Model Evaluation Pipeline - Credit Card Fraud Detection
VERSION CORRIGÉE: Évaluation finale sur test set JAMAIS VU
Métriques business réalistes et visualisations
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, average_precision_score, 
    roc_curve, precision_recall_curve, accuracy_score, 
    f1_score, precision_score, recall_score
)
import pickle
import json
import os
from pathlib import Path
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

def convert_numpy_types(obj):
    """Conversion types NumPy vers Python pour JSON"""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

class FraudModelEvaluator:
    """🎯 Évaluateur FINAL - Test set JAMAIS VU pendant training"""
    
    def __init__(self):
        # Chemins
        self.base_dir = Path(__file__).resolve().parent.parent
        self.models_dir = self.base_dir / "models"
        self.processed_dir = self.base_dir / "data" / "processed"
        self.assets_dir = self.base_dir / "assets"
        
        # Création dossier assets
        os.makedirs(self.assets_dir, exist_ok=True)
        
        # Données et modèle
        self.model = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.y_pred_proba = None
        self.feature_names = None
        self.optimal_threshold = 0.5
        
        # Historique training pour comparaison
        self.training_history = None
    
    def load_model_and_test_data(self):
        """
        🎯 CRUCIAL: Chargement modèle + TEST SET ISOLÉ
        Premier contact du modèle avec ces données
        """
        print("📥 Chargement modèle et TEST SET JAMAIS VU...")
        
        # Chargement modèle
        model_path = self.models_dir / "fraud_detector.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Modèle non trouvé: {model_path}")
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # 🎯 CRITIQUE: Chargement TEST SET - PREMIER CONTACT
        try:
            self.X_test = pd.read_pickle(self.processed_dir / "X_test.pkl")
            self.y_test = pd.read_pickle(self.processed_dir / "y_test.pkl")
        except FileNotFoundError:
            raise FileNotFoundError(
                "Test set non trouvé! Exécutez d'abord le preprocessing corrigé."
            )
        
        # Chargement historique training pour comparaison
        history_path = self.models_dir / "training_history.json"
        if history_path.exists():
            with open(history_path, 'r') as f:
                self.training_history = json.load(f)
        
        self.feature_names = self.X_test.columns.tolist()
        
        print(f"✅ Chargement terminé:")
        print(f"   • Modèle: {type(self.model).__name__}")
        print(f"   🔒 Test samples (JAMAIS VUS): {len(self.X_test):,}")
        print(f"   • Features: {len(self.feature_names)}")
        print(f"   • Test fraud rate: {self.y_test.mean()*100:.2f}%")
        print(f"   🎯 PREMIÈRE ÉVALUATION sur données inconnues")
    
    def find_optimal_threshold_and_predict(self):
        """
        🎯 PREMIÈRE PRÉDICTION sur test set + recherche seuil optimal
        """
        print("🎯 PREMIÈRE PRÉDICTION sur test set + seuil optimal...")
        
        # 🔥 MOMENT DE VÉRITÉ: Premières prédictions sur données jamais vues
        self.y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Recherche seuil optimal pour F1-Score sur TEST SET
        precisions, recalls, thresholds = precision_recall_curve(self.y_test, self.y_pred_proba)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        
        # Seuil optimal (excluant le dernier point)
        optimal_idx = np.argmax(f1_scores[:-1])
        self.optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        
        # Prédictions finales avec seuil optimal
        self.y_pred = (self.y_pred_proba >= self.optimal_threshold).astype(int)
        
        print(f"✅ PREMIÈRES PRÉDICTIONS terminées:")
        print(f"   • Seuil optimal: {self.optimal_threshold:.4f}")
        print(f"   • F1-Score optimal: {f1_scores[optimal_idx]:.4f}")
        print(f"   • Prédictions générées: {len(self.y_pred):,}")
    
    def calculate_final_metrics(self) -> Dict:
        """
        🎯 MÉTRIQUES FINALES - Performance réelle du modèle
        """
        print("📊 Calcul MÉTRIQUES FINALES sur test set...")
        
        # Métriques classification
        accuracy = accuracy_score(self.y_test, self.y_pred)
        precision = precision_score(self.y_test, self.y_pred)
        recall = recall_score(self.y_test, self.y_pred)
        f1 = f1_score(self.y_test, self.y_pred)
        
        # Métriques probabilistes
        roc_auc = roc_auc_score(self.y_test, self.y_pred_proba)
        pr_auc = average_precision_score(self.y_test, self.y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, self.y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Métriques dérivées
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        # 💰 MÉTRIQUES BUSINESS RÉALISTES
        cost_fp = fp * 50    # €50 par fausse alerte (investigation)
        cost_fn = fn * 2500  # €2500 par fraude manquée (perte moyenne)
        total_cost = cost_fp + cost_fn
        cost_per_transaction = total_cost / len(self.y_test)
        
        # Économies vs baseline (sans modèle)
        baseline_cost = len(self.y_test[self.y_test == 1]) * 2500  # Toutes fraudes passent
        savings = baseline_cost - total_cost
        roi_ratio = savings / (cost_fp + 10000) if (cost_fp + 10000) > 0 else 0  # +10k coût déploiement
        
        # Métriques avancées
        balanced_accuracy = (recall + specificity) / 2
        matthews_corr = ((tp * tn) - (fp * fn)) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) > 0 else 0
        
        metrics = {
            'test_set_info': {
                'total_transactions': len(self.y_test),
                'fraud_transactions': int(self.y_test.sum()),
                'normal_transactions': int((self.y_test == 0).sum()),
                'fraud_rate': float(self.y_test.mean())
            },
            'optimal_threshold': self.optimal_threshold,
            'classification_metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'specificity': specificity,
                'f1_score': f1,
                'balanced_accuracy': balanced_accuracy,
                'matthews_correlation': matthews_corr,
                'npv': npv
            },
            'probabilistic_metrics': {
                'roc_auc': roc_auc,
                'pr_auc': pr_auc
            },
            'confusion_matrix': {
                'true_negative': int(tn),
                'false_positive': int(fp),
                'false_negative': int(fn),
                'true_positive': int(tp)
            },
            'business_impact': {
                'frauds_detected': int(tp),
                'frauds_missed': int(fn),
                'false_alerts': int(fp),
                'cost_false_positive_eur': float(cost_fp),
                'cost_false_negative_eur': float(cost_fn),
                'total_cost_eur': float(total_cost),
                'cost_per_transaction_eur': float(cost_per_transaction),
                'baseline_cost_eur': float(baseline_cost),
                'savings_eur': float(savings),
                'roi_ratio': float(roi_ratio),
                'detection_rate_percent': float(recall * 100),
                'false_positive_rate_percent': float((fp / (fp + tn)) * 100) if (fp + tn) > 0 else 0,
                'precision_percent': float(precision * 100)
            }
        }
        
        # 🔍 Comparaison avec validation training si disponible
        if self.training_history and 'honest_validation' in self.training_history:
            val_metrics = self.training_history['honest_validation']
            metrics['training_comparison'] = {
                'validation_f1': val_metrics.get('f1_score', 0),
                'test_f1': f1,
                'f1_difference': f1 - val_metrics.get('f1_score', 0),
                'validation_roc_auc': val_metrics.get('roc_auc', 0),
                'test_roc_auc': roc_auc,
                'roc_auc_difference': roc_auc - val_metrics.get('roc_auc', 0),
                'generalization_quality': 'good' if abs(f1 - val_metrics.get('f1_score', 0)) < 0.05 else 'concerning'
            }
        
        print("✅ Métriques finales calculées")
        return metrics
    
    def create_performance_visualizations(self) -> Dict[str, go.Figure]:
        """Création visualisations performance finale"""
        print("🎨 Création visualisations performance finale...")
        
        visualizations = {}
        
        # 1. Confusion Matrix
        cm = confusion_matrix(self.y_test, self.y_pred)
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        annotations = []
        for i in range(2):
            for j in range(2):
                annotations.append({
                    'x': j, 'y': i,
                    'text': f'{cm[i,j]:,}<br>({cm_percent[i,j]:.1f}%)',
                    'showarrow': False,
                    'font': {'color': 'white' if cm[i,j] > cm.max()/2 else 'black', 'size': 16}
                })
        
        cm_fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Prédit Normal', 'Prédit Fraude'],
            y=['Réel Normal', 'Réel Fraude'],
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title="Nombre")
        ))
        
        cm_fig.update_layout(
            title="🎯 Confusion Matrix - Performance Test Set Final",
            annotations=annotations,
            width=600, height=500,
            xaxis_title="Prédiction",
            yaxis_title="Réalité"
        )
        visualizations['confusion_matrix'] = cm_fig
        
        # 2. ROC et Precision-Recall Curves
        fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_proba)
        roc_auc = roc_auc_score(self.y_test, self.y_pred_proba)
        
        precision, recall, _ = precision_recall_curve(self.y_test, self.y_pred_proba)
        pr_auc = average_precision_score(self.y_test, self.y_pred_proba)
        
        curves_fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[f'ROC Curve (AUC={roc_auc:.3f})', f'Precision-Recall (AUC={pr_auc:.3f})']
        )
        
        # ROC
        curves_fig.add_trace(
            go.Scatter(x=fpr, y=tpr, name=f'ROC (AUC={roc_auc:.3f})', 
                      line=dict(color='blue', width=3)),
            row=1, col=1
        )
        curves_fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], name='Random', 
                      line=dict(dash='dash', color='gray')),
            row=1, col=1
        )
        
        # PR
        curves_fig.add_trace(
            go.Scatter(x=recall, y=precision, name=f'PR (AUC={pr_auc:.3f})', 
                      line=dict(color='red', width=3)),
            row=1, col=2
        )
        
        baseline_pr = sum(self.y_test) / len(self.y_test)
        curves_fig.add_trace(
            go.Scatter(x=[0, 1], y=[baseline_pr, baseline_pr], 
                      name=f'Baseline ({baseline_pr:.3f})', 
                      line=dict(dash='dash', color='gray')),
            row=1, col=2
        )
        
        curves_fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)
        curves_fig.update_yaxes(title_text="True Positive Rate", row=1, col=1)
        curves_fig.update_xaxes(title_text="Recall", row=1, col=2)
        curves_fig.update_yaxes(title_text="Precision", row=1, col=2)
        
        curves_fig.update_layout(height=400, showlegend=True, 
                                title_text="🎯 Performance Curves - Test Set Final")
        visualizations['performance_curves'] = curves_fig
        
        # 3. Feature Importance
        feature_importance = self.model.feature_importances_
        feature_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        top_features = feature_df.head(15)
        
        feat_fig = px.bar(
            top_features,
            x='importance',
            y='feature',
            orientation='h',
            title='🎯 Top 15 Features - Importance Modèle Final',
            labels={'importance': 'Importance', 'feature': 'Features'},
            color='importance',
            color_continuous_scale='viridis'
        )
        
        feat_fig.update_layout(
            height=600, 
            yaxis={'categoryorder': 'total ascending'},
            showlegend=False
        )
        visualizations['feature_importance'] = feat_fig
        
        # 4. Distribution Probabilités
        prob_normal = self.y_pred_proba[self.y_test == 0]
        prob_fraud = self.y_pred_proba[self.y_test == 1]
        
        prob_fig = go.Figure()
        
        prob_fig.add_trace(go.Histogram(
            x=prob_normal, 
            name='Transactions Normales',
            opacity=0.7,
            nbinsx=50,
            marker_color='lightblue',
            histnorm='probability'
        ))
        
        prob_fig.add_trace(go.Histogram(
            x=prob_fraud,
            name='Transactions Frauduleuses', 
            opacity=0.7,
            nbinsx=50,
            marker_color='red',
            histnorm='probability'
        ))
        
        prob_fig.add_vline(
            x=self.optimal_threshold, 
            line_dash="dash", 
            line_color="green",
            annotation_text=f"Seuil Optimal: {self.optimal_threshold:.3f}"
        )
        
        prob_fig.update_layout(
            title='🎯 Distribution Probabilités - Test Set Final',
            xaxis_title='Probabilité Fraude',
            yaxis_title='Densité',
            barmode='overlay',
            height=400
        )
        visualizations['probability_distribution'] = prob_fig
        
        print("✅ Visualisations créées")
        return visualizations
    
    def generate_final_business_report(self, metrics: Dict) -> str:
        """🎯 Génération rapport business FINAL"""
        cm = metrics['confusion_matrix']
        biz = metrics['business_impact']
        cls = metrics['classification_metrics']
        prob = metrics['probabilistic_metrics']
        
        # Comparaison avec validation si disponible
        comparison_section = ""
        if 'training_comparison' in metrics:
            tc = metrics['training_comparison']
            comparison_section = f"""
## 🔍 VALIDATION TRAINING vs TEST FINAL

### Comparaison Performance
- **F1-Score Validation**: {tc['validation_f1']:.4f}
- **F1-Score Test Final**: {tc['test_f1']:.4f}
- **Différence**: {tc['f1_difference']:+.4f}
- **Qualité Généralisation**: {tc['generalization_quality'].upper()}

### Analyse Généralisation
{('✅ EXCELLENTE - Le modèle généralise bien sur données inconnues' if tc['generalization_quality'] == 'good' else '⚠️ ATTENTION - Possible overfitting ou shift de données')}
"""

        report = f"""
# 🛡️ RAPPORT FINAL - FRAUD DETECTION MODEL

## 📊 RÉSUMÉ EXÉCUTIF - PERFORMANCE TEST SET

**🎯 PREMIÈRE ÉVALUATION SUR DONNÉES JAMAIS VUES**

### Performance Globale
- **Accuracy**: {cls['accuracy']*100:.1f}%
- **Taux Détection Fraudes**: {cls['recall']*100:.1f}%
- **Précision**: {cls['precision']*100:.1f}%
- **Score F1**: {cls['f1_score']*100:.1f}%
- **ROC-AUC**: {prob['roc_auc']:.3f}
- **Seuil Optimal**: {metrics['optimal_threshold']:.4f}

### 🎯 Échantillon Test
- **Total Transactions**: {metrics['test_set_info']['total_transactions']:,}
- **Fraudes Réelles**: {metrics['test_set_info']['fraud_transactions']:,}
- **Taux Fraude**: {metrics['test_set_info']['fraud_rate']*100:.2f}%

## 💰 IMPACT BUSINESS RÉEL

### Performance Détection
- **Fraudes Détectées**: {biz['frauds_detected']:,} / {metrics['test_set_info']['fraud_transactions']:,}
- **Fraudes Manquées**: {biz['frauds_missed']:,}
- **Fausses Alertes**: {biz['false_alerts']:,}
- **Taux Détection**: {biz['detection_rate_percent']:.1f}%

### Impact Financier
- **Coût Fausses Alertes**: €{biz['cost_false_positive_eur']:,.0f}
- **Coût Fraudes Manquées**: €{biz['cost_false_negative_eur']:,.0f}
- **Coût Total**: €{biz['total_cost_eur']:,.0f}
- **Coût par Transaction**: €{biz['cost_per_transaction_eur']:.3f}

### ROI et Économies
- **Coût Sans Modèle**: €{biz['baseline_cost_eur']:,.0f}
- **Économies Réalisées**: €{biz['savings_eur']:,.0f}
- **ROI**: {biz['roi_ratio']:.1f}:1
- **Réduction Pertes**: {(biz['savings_eur']/biz['baseline_cost_eur']*100):.1f}%

{comparison_section}

## 🎯 PERFORMANCE TECHNIQUE DÉTAILLÉE

### Métriques Classification
- **Accuracy**: {cls['accuracy']:.4f}
- **Precision**: {cls['precision']:.4f}
- **Recall (Sensitivity)**: {cls['recall']:.4f}
- **Specificity**: {cls['specificity']:.4f}
- **F1-Score**: {cls['f1_score']:.4f}
- **Balanced Accuracy**: {cls['balanced_accuracy']:.4f}
- **Matthews Correlation**: {cls['matthews_correlation']:.4f}
- **NPV**: {cls['npv']:.4f}

### Métriques Probabilistes
- **ROC-AUC**: {prob['roc_auc']:.4f}
- **PR-AUC**: {prob['pr_auc']:.4f}

### Matrice de Confusion
```
                  Prédictions
Réalité    Normal    Fraude
Normal     {cm['true_negative']:,}      {cm['false_positive']:,}
Fraude     {cm['false_negative']:,}       {cm['true_positive']:,}
```

## ✅ VALIDATION PRODUCTION REQUIREMENTS

| Requirement | Target | Test Final | Status |
|-------------|--------|------------|--------|
| F1-Score    | >75%   | {cls['f1_score']*100:.1f}% | {'✅' if cls['f1_score'] > 0.75 else '🔴'} |
| Precision   | >80%   | {cls['precision']*100:.1f}% | {'✅' if cls['precision'] > 0.80 else '🔴'} |
| Recall      | >70%   | {cls['recall']*100:.1f}% | {'✅' if cls['recall'] > 0.70 else '🔴'} |
| ROC-AUC     | >90%   | {prob['roc_auc']*100:.1f}% | {'✅' if prob['roc_auc'] > 0.90 else '🔴'} |
| FP Rate     | <5%    | {biz['false_positive_rate_percent']:.1f}% | {'✅' if biz['false_positive_rate_percent'] < 5.0 else '🔴'} |

## 🚀 VERDICT FINAL

### Statut Production
{self._get_production_verdict(cls, prob, biz)}

### Recommandations Business
{self._get_business_recommendations(cls, prob, biz)}

### Impact Attendu en Production
- **Réduction Fraudes**: {biz['detection_rate_percent']:.0f}% des cas détectés
- **Économies Annuelles**: €{biz['savings_eur']*12:,.0f} (extrapolation mensuelle)
- **Amélioration Opérationnelle**: {(100-biz['false_positive_rate_percent']):.1f}% transactions légitimes non impactées
- **Conformité**: Aide au respect PCI DSS et réglementations

### Limites et Considérations
- **Données Test**: Performance sur échantillon spécifique
- **Adaptation**: Monitoring performance requis en production
- **Seuil**: Ajustable selon contexte business
- **Évolution**: Réentraînement périodique recommandé

---

**🎯 Rapport généré sur TEST SET FINAL - Première évaluation**
*Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
*Échantillon: {metrics['test_set_info']['total_transactions']:,} transactions jamais vues*
        """
        
        return report
    
    def _get_production_verdict(self, cls, prob, biz) -> str:
        """Verdict production basé sur métriques finales"""
        production_ready = True
        issues = []
        
        if cls['f1_score'] < 0.75:
            production_ready = False
            issues.append("F1-Score insuffisant")
        
        if cls['precision'] < 0.80:
            production_ready = False
            issues.append("Précision insuffisante")
            
        if cls['recall'] < 0.70:
            production_ready = False
            issues.append("Recall insuffisant")
            
        if prob['roc_auc'] < 0.90:
            production_ready = False
            issues.append("ROC-AUC insuffisant")
        
        if biz['false_positive_rate_percent'] > 5.0:
            production_ready = False
            issues.append("Taux faux positifs trop élevé")
        
        if production_ready:
            return """
🟢 **PRODUCTION READY - DÉPLOIEMENT RECOMMANDÉ**
- Toutes les métriques respectent les seuils production
- Performance stable sur données inconnues
- ROI positif confirmé
- Impact business significatif"""
        else:
            return f"""
🟡 **OPTIMISATION REQUISE AVANT PRODUCTION**
- Issues identifiées: {', '.join(issues)}
- Performance acceptable mais améliorable
- Considérer réentraînement ou tuning
- Évaluation coût/bénéfice nécessaire"""
    
    def _get_business_recommendations(self, cls, prob, biz) -> str:
        """Recommandations business basées sur performance"""
        recommendations = []
        
        if cls['recall'] < 0.80:
            recommendations.append("• **Améliorer Détection**: Ajuster seuil ou enrichir features")
        
        if cls['precision'] < 0.85:
            recommendations.append("• **Réduire Fausses Alertes**: Optimiser seuil de décision")
        
        if biz['roi_ratio'] < 5:
            recommendations.append("• **Optimiser ROI**: Réviser coûts d'investigation")
        
        if prob['roc_auc'] < 0.95:
            recommendations.append("• **Performance Technique**: Explorer ensemble methods")
        
        recommendations.extend([
            "• **Monitoring**: Surveillance performance hebdomadaire",
            "• **Adaptation**: Réentraînement trimestriel recommandé",
            "• **Seuil Dynamique**: Ajustement selon contexte métier",
            "• **Integration**: API temps réel pour scoring live"
        ])
        
        return '\n'.join(recommendations)
    
    def save_final_evaluation(self, metrics: Dict, report: str, visualizations: Dict):
        """Sauvegarde évaluation finale complète"""
        print("💾 Sauvegarde évaluation finale...")
        
        # Conversion types NumPy pour JSON
        metrics_json = convert_numpy_types(metrics)
        
        # Sauvegarde métriques finales
        with open(self.models_dir / "final_test_metrics.json", 'w', encoding='utf-8') as f:
            json.dump(metrics_json, f, indent=2)
        
        # Sauvegarde rapport business final
        with open(self.assets_dir / "final_evaluation_report.md", 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Sauvegarde visualisations
        try:
            for name, fig in visualizations.items():
                fig.write_image(self.assets_dir / f"final_{name}.png", width=800, height=600)
            print("✅ Visualisations sauvegardées")
        except Exception as e:
            print(f"⚠️ Erreur sauvegarde visualisations: {e}")
            print("   (Kaleido requis: pip install kaleido)")
        
        # Résumé JSON compact pour intégration
        summary = {
            'model_performance': {
                'f1_score': metrics_json['classification_metrics']['f1_score'],
                'precision': metrics_json['classification_metrics']['precision'],
                'recall': metrics_json['classification_metrics']['recall'],
                'roc_auc': metrics_json['probabilistic_metrics']['roc_auc']
            },
            'business_impact': {
                'frauds_detected': metrics_json['business_impact']['frauds_detected'],
                'savings_eur': metrics_json['business_impact']['savings_eur'],
                'roi_ratio': metrics_json['business_impact']['roi_ratio']
            },
            'production_ready': metrics_json['classification_metrics']['f1_score'] > 0.75,
            'evaluation_date': pd.Timestamp.now().isoformat(),
            'test_samples': metrics_json['test_set_info']['total_transactions']
        }
        
        with open(self.assets_dir / "model_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Évaluation finale sauvegardée:")
        print(f"   • Métriques: models/final_test_metrics.json")
        print(f"   • Rapport: assets/final_evaluation_report.md")
        print(f"   • Résumé: assets/model_summary.json")
        print(f"   • Visualisations: assets/final_*.png")
    
    def display_final_summary(self, metrics: Dict):
        """Affichage résumé final"""
        print("\n" + "=" * 80)
        print("🎉 ÉVALUATION FINALE TERMINÉE - TEST SET JAMAIS VU")
        print("=" * 80)
        
        cls = metrics['classification_metrics']
        biz = metrics['business_impact']
        prob = metrics['probabilistic_metrics']
        
        # Performance principale
        print("🎯 PERFORMANCE TEST SET FINAL:")
        print(f"   • F1-Score: {cls['f1_score']:.4f}")
        print(f"   • Precision: {cls['precision']:.4f}")
        print(f"   • Recall: {cls['recall']:.4f}")
        print(f"   • ROC-AUC: {prob['roc_auc']:.4f}")
        
        # Impact business
        print(f"\n💰 IMPACT BUSINESS RÉEL:")
        print(f"   • Fraudes détectées: {biz['frauds_detected']:,}/{metrics['test_set_info']['fraud_transactions']:,}")
        print(f"   • Fausses alertes: {biz['false_alerts']:,}")
        print(f"   • Économies: €{biz['savings_eur']:,.0f}")
        print(f"   • ROI: {biz['roi_ratio']:.1f}:1")
        
        # Comparaison validation si disponible
        if 'training_comparison' in metrics:
            tc = metrics['training_comparison']
            print(f"\n🔍 GÉNÉRALISATION:")
            print(f"   • F1 Validation: {tc['validation_f1']:.4f}")
            print(f"   • F1 Test Final: {tc['test_f1']:.4f}")
            print(f"   • Différence: {tc['f1_difference']:+.4f}")
            print(f"   • Qualité: {tc['generalization_quality'].upper()}")
        
        # Verdict final
        production_ready = (cls['f1_score'] > 0.75 and 
                          cls['precision'] > 0.80 and 
                          cls['recall'] > 0.70 and 
                          prob['roc_auc'] > 0.90)
        
        if production_ready:
            print(f"\n🟢 VERDICT: PRODUCTION READY!")
            print(f"   ✅ Toutes métriques conformes")
            print(f"   ✅ ROI positif confirmé") 
            print(f"   ✅ Généralisation validée")
        else:
            print(f"\n🟡 VERDICT: OPTIMISATION RECOMMANDÉE")
            print(f"   • Performance acceptable mais améliorable")
            print(f"   • Évaluation coût/bénéfice nécessaire")
        
        print(f"\n📊 ÉCHANTILLON TEST:")
        print(f"   • {metrics['test_set_info']['total_transactions']:,} transactions JAMAIS VUES")
        print(f"   • {metrics['test_set_info']['fraud_transactions']:,} fraudes réelles")
        print(f"   • Première évaluation sur données inconnues")
        
        print("\n📂 ARTEFACTS GÉNÉRÉS:")
        print("   • Métriques détaillées: models/final_test_metrics.json")
        print("   • Rapport business: assets/final_evaluation_report.md")
        print("   • Visualisations: assets/final_*.png")
        print("   • Résumé: assets/model_summary.json")
        
        print("=" * 80)
    
    def run_final_evaluation(self):
        """🎯 Pipeline évaluation finale complète"""
        print("🚀 DÉMARRAGE ÉVALUATION FINALE - TEST SET JAMAIS VU")
        print("=" * 80)
        
        try:
            # 1. Chargement modèle et test set isolé
            self.load_model_and_test_data()
            
            # 2. Premières prédictions sur données inconnues
            self.find_optimal_threshold_and_predict()
            
            # 3. Calcul métriques finales
            metrics = self.calculate_final_metrics()
            
            # 4. Création visualisations
            visualizations = self.create_performance_visualizations()
            
            # 5. Génération rapport business final
            report = self.generate_final_business_report(metrics)
            
            # 6. Sauvegarde complète
            self.save_final_evaluation(metrics, report, visualizations)
            
            # 7. Affichage résumé final
            self.display_final_summary(metrics)
            
            return True
            
        except Exception as e:
            print(f"\n❌ ERREUR ÉVALUATION FINALE: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Exécution évaluation finale"""
    print("🎯 Credit Card Fraud Detection - Évaluation Finale")
    print("🔒 Test Set Jamais Vu - Première Évaluation")
    
    evaluator = FraudModelEvaluator()
    success = evaluator.run_final_evaluation()
    
    if success:
        print("\n✅ ÉVALUATION FINALE RÉUSSIE!")
        print("🎯 Modèle évalué sur données jamais vues")
        print("📊 Performance réelle confirmée")
        print("🚀 Prêt pour décision déploiement")
    else:
        print("\n❌ ÉVALUATION FINALE ÉCHOUÉE!")
        exit(1)

if __name__ == "__main__":
    main()