import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

class IEEECISDataPipeline:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.label_encoders = {}

    def load_data(self, sample_frac=0.1):
        print("Loading IEEE-CIS dataset...")
        train_transaction = pd.read_csv(os.path.join(self.data_dir, 'train_transaction.csv'))
        train_identity = pd.read_csv(os.path.join(self.data_dir, 'train_identity.csv'))
        df = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
        if sample_frac < 1.0:
            print(f"Sampling {sample_frac*100}% of data...")
            df = df.sample(frac=sample_frac, random_state=42)
        return df

    def preprocess(self, df):
        print("Preprocessing dataset...")
        y = df['isFraud'].values
        X = df.drop(['isFraud', 'TransactionID'], axis=1)
        categorical_cols = X.select_dtypes(include=['object']).columns
        numerical_cols = X.select_dtypes(exclude=['object']).columns
        X[numerical_cols] = X[numerical_cols].fillna(-999)
        for col in categorical_cols:
            X[col] = X[col].fillna('missing')
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        print(f"Final shape: {X.shape}, Fraud rate: {(y.sum()/len(y))*100:.2f}%")
        return X.values, y

    def save_artifacts(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        # Scaler is now saved in train_engine.py to prevent double-scaling
        joblib.dump(self.label_encoders, os.path.join(output_dir, 'label_encoders.pkl'))
        print("Preprocessing artifacts saved.")

if __name__ == "__main__":
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'raw')
    OUTPUT_DIR = "models/artifacts"
    pipeline = IEEECISDataPipeline(DATA_DIR)
    df = pipeline.load_data(sample_frac=0.1)
    X, y = pipeline.preprocess(df)
    pipeline.save_artifacts(OUTPUT_DIR)
    os.makedirs('data', exist_ok=True)
    np.save('data/X_processed.npy', X)
    np.save('data/y_processed.npy', y)
    print("Processed arrays saved to data/ directory.")
