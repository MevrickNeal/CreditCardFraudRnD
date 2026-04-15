import numpy as np
"""
Friday Fraud - Hybrid XGBoost Ensemble
Updated: 2026-04-15
"""
import xgboost as xgb
from sklearn.metrics import roc_auc_score

class HybridEnsemble:
    """
    Batch-trained XGBoost on VAE hybrid features (64 latent + 1 recon error).
    scale_pos_weight handles class imbalance natively (~27:1 legit:fraud).
    Online adaptation: stores recent predictions for live AUC tracking.
    """
    def __init__(self):
        self.xgb_model = None
        self.is_trained = False
        self.samples_seen = 0
        self.y_true_history = []
        self.y_pred_history = []

    def train(self, X_train_hybrid, y_train):
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        spw = neg / pos if pos > 0 else 1.0
        print(f"  XGBoost scale_pos_weight={spw:.1f} (neg={neg}, pos={pos})")
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=spw,
            tree_method='hist',
            eval_metric='auc',
            random_state=42,
            use_label_encoder=False
        )
        self.xgb_model.fit(X_train_hybrid, y_train)
        self.is_trained = True
        self.samples_seen = len(y_train)

    def predict_proba_one(self, hybrid_features):
        if not self.is_trained:
            return 0.5
        X = np.array([hybrid_features])
        try:
            prob = self.xgb_model.predict_proba(X)[0][1]
            self.y_pred_history.append(float(prob))
            return prob
        except:
            return 0.5

    def fit_one(self, hybrid_features, y_label):
        """Record label for live AUC tracking (no re-train, too expensive)."""
        self.y_true_history.append(int(y_label))

    def get_metric(self):
        n = min(len(self.y_true_history), len(self.y_pred_history), 1000)
        if n < 10:
            return 0.5
        y_t = self.y_true_history[-n:]
        y_p = self.y_pred_history[-n:]
        if len(set(y_t)) < 2:
            return 0.5
        try:
            return roc_auc_score(y_t, y_p)
        except:
            return 0.5
