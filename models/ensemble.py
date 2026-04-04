import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score

class StreamingEnsemble:
    """
    Continuous Learning Ensemble relying on SGDClassifier (Logistic Regression).
    Provides online/incremental learning capabilities since River installation failed.
    """
    def __init__(self, seed=42):
        self.model = SGDClassifier(loss='log_loss', penalty='l2', random_state=seed, warm_start=True)
        self.samples_seen = 0
        self.classes_seen = set()
        self.y_true_history = []
        self.y_pred_history = []
        
    def fit_one(self, x_array, y):
        """ Train the ensemble incrementally on a single sample (emulated with partial_fit) """
        X = np.array([x_array])
        Y = np.array([y])
        self.classes_seen.add(y)
        
        if self.samples_seen > 0 and len(self.classes_seen) > 1:
            try:
                y_pred = self.model.predict_proba(X)[0][1]
                self.y_pred_history.append(y_pred)
                self.y_true_history.append(y)
            except Exception:
                pass
        else:
            self.y_true_history.append(y)
            self.y_pred_history.append(0.5)
            
        self.model.partial_fit(X, Y, classes=np.array([0, 1]))
        self.samples_seen += 1
        
    def predict_proba_one(self, x_array):
        """ Predict fraud probability for a single transaction """
        if self.samples_seen == 0:
            return 0.0
            
        X = np.array([x_array])
        try:
            probas = self.model.predict_proba(X)
            return probas[0][1]
        except:
            return 0.0
            
    def get_metric(self):
        """ Returns the current ROC AUC based on recent history """
        if len(set(self.y_true_history)) > 1:

            y_t = self.y_true_history[-1000:]
            y_p = self.y_pred_history[-1000:]
            if len(set(y_t)) > 1:
                return roc_auc_score(y_t, y_p)
        return 0.5