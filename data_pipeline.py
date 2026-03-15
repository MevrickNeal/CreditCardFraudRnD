import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

class IEEECISDataPipeline:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self, sample_frac=0.1):
        """Loads a sample of the dataset to allow local training."""
        print("Loading IEEE-CIS dataset...")
        train_transaction = pd.read_csv(os.path.join(self.data_dir, 'train_transaction.csv'))
        train_identity = pd.read_csv(os.path.join(self.data_dir, 'train_identity.csv'))
        
        df = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
        
        if sample_frac < 1.0:
            print(f"Sampling {sample_frac*100}% of data to reduce memory load...")

            df = df.sample(frac=sample_frac, random_state=42)
        
        return df
        
    def preprocess(self, df):
        print("Preprocessing dataset...")
        
        print(f"Columns in dataset: {[c for c in df.columns if 'fraud' in c.lower()]}")
        target_col = 'isFraud' if 'isFraud' in df.columns else 'isFraud'
        
        y = df[target_col].values
        X = df.drop([target_col, 'TransactionID'], axis=1)
        
        categorical_cols = X.select_dtypes(include=['object']).columns
        numerical_cols = X.select_dtypes(exclude=['object']).columns
        
        X[numerical_cols] = X[numerical_cols].fillna(-999)
        
        for col in categorical_cols:
            X[col] = X[col].fillna('missing')
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
            
        X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
        
        print(f"Final shape of X: {X.shape}")
        print(f"Fraud prevalence: {(y.sum() / len(y)) * 100:.2f}%")
        
        return X.values, y
        
    def save_artifacts(self, output_dir):
        """Save scaler and encoders for inference."""
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(self.scaler, os.path.join(output_dir, 'scaler.pkl'))
        joblib.dump(self.label_encoders, os.path.join(output_dir, 'label_encoders.pkl'))
        print("Preprocessing artifacts saved.")

if __name__ == "__main__":
    DATA_DIR = r"C:\Users\Lian Mollick\Desktop\Personal\learning\Mahi_s paper"
    OUTPUT_DIR = "models/artifacts"
    
    pipeline = IEEECISDataPipeline(DATA_DIR)
    
    df = pipeline.load_data(sample_frac=0.1) 
    
    X, y = pipeline.preprocess(df)
    
    pipeline.save_artifacts(OUTPUT_DIR)
    
    os.makedirs('data', exist_ok=True)
    np.save('data/X_processed.npy', X)
    np.save('data/y_processed.npy', y)
    print("Processed arrays saved to data/ directory.")