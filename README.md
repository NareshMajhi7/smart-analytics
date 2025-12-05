# smart_analytics.py
"""
Smart Analytics - Credit Card Fraud Detection (starter)
Features:
 - Load CSV dataset (user-provided) or generate synthetic data for testing
 - Preprocessing: missing values, encoding, scaling
 - Handle class imbalance with SMOTE
 - Train RandomForest with GridSearchCV
 - Evaluate using precision, recall, F1, ROC-AUC
 - Save trained model and preprocessing pipeline
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import joblib
from sklearn.impute import SimpleImputer

# -------------------------
# CONFIG
# -------------------------
DATA_PATH = "data/transactions.csv"   # <-- replace with your CSV path
TARGET_COLUMN = "is_fraud"            # <-- replace with your target column name (0/1)
SAVE_DIR = "output"
RANDOM_STATE = 42

os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------------
# Helper functions
# -------------------------
def load_data(path):
    """Load CSV dataset. Expects pandas-readable CSV."""
    df = pd.read_csv(path)
    print(f"Loaded dataset with shape: {df.shape}")
    return df

def generate_synthetic_data(n_samples=10000, fraud_ratio=0.02, random_state=RANDOM_STATE):
    """Generate a synthetic dataset for testing purposes."""
    rng = np.random.RandomState(random_state)
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    # Numeric features
    amt_legit = rng.normal(loc=50, scale=30, size=n_legit).clip(0.01)
    amt_fraud = rng.normal(loc=300, scale=200, size=n_fraud).clip(0.01)
    amount = np.concatenate([amt_legit, amt_fraud])

    # Time of day (0-23)
    time_legit = rng.randint(6, 22, size=n_legit)  # daytime
    time_fraud = rng.randint(0, 5, size=n_fraud)   # night
    time = np.concatenate([time_legit, time_fraud])

    # Merchant categories (categorical)
    merchants = ['grocery', 'electronics', 'travel', 'luxury', 'utilities']
    merchant = np.concatenate([
        rng.choice(merchants, size=n_legit, p=[0.4,0.2,0.1,0.05,0.25]),
        rng.choice(merchants, size=n_fraud, p=[0.1,0.4,0.2,0.25,0.05])
    ])

    # device type
    devices = ['mobile', 'desktop', 'tablet']
    device = rng.choice(devices, size=n_samples, p=[0.7,0.25,0.05])

    # target
    target = np.array([0]*n_legit + [1]*n_fraud)

    df = pd.DataFrame({
        'amount': amount,
        'hour': time,
        'merchant': merchant,
        'device': device,
        TARGET_COLUMN: target
    })
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    print(f"Generated synthetic dataset with shape: {df.shape} and fraud ratio: {df[TARGET_COLUMN].mean():.4f}")
    return df

def summarize_data(df, target_col=TARGET_COLUMN):
    print("\n--- Data Summary ---")
    print(df.describe(include='all').T)
    print("\nClass distribution:")
    print(df[target_col].value_counts(normalize=True))
    print("\nMissing values per column:")
    print(df.isna().sum())

# -------------------------
# Main pipeline
# -------------------------
def main(use_synthetic=False):
    # 1. Load or generate data
    if use_synthetic:
        df = generate_synthetic_data(n_samples=20000, fraud_ratio=0.02)
    else:
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Set DATA_PATH or use synthetic data.")
        df = load_data(DATA_PATH)

    summarize_data(df)

    # 2. Split features/target
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' not found in dataset columns.")
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)

    # 3. Identify numeric and categorical columns
    numeric_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object','category']).columns.tolist()
    print(f"\nNumeric columns: {numeric_cols}")
    print(f"Categorical columns: {categorical_cols}")

    # 4. Preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ], remainder='drop')

    # 5. Train-test split (stratify to keep class ratios)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )

    # 6. Apply preprocessing to training data (fit) and test data (transform)
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)
    print(f"\nPreprocessed shapes -> Train: {X_train_prep.shape}, Test: {X_test_prep.shape}")

    # 7. Handle class imbalance using SMOTE on training set only
    print("\nApplying SMOTE to training data...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train_prep, y_train)
    print(f"After SMOTE -> X: {X_res.shape}, y distribution: {np.bincount(y_res)}")

    # 8. Model training with RandomForest + simple grid search
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_leaf': [1, 2]
    }
    grid = GridSearchCV(rf, param_grid, scoring='f1', cv=3, verbose=1)
    grid.fit(X_res, y_res)
    print(f"\nBest params: {grid.best_params_}")
    best_model = grid.best_estimator_

    # 9. Evaluation on test set
    y_pred = best_model.predict(X_test_prep)
    y_proba = best_model.predict_proba(X_test_prep)[:, 1]

    print("\n--- Classification Report (Test set) ---")
    print(classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC: {roc_auc:.4f}")

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0,1], [0,1], '--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'roc_curve.png'))
    plt.close()
    print(f"ROC curve saved to {os.path.join(SAVE_DIR, 'roc_curve.png')}")

    # 10. Feature importance (only if we have numeric+onehot combined -> get names)
    try:
        # Build feature names list
        num_names = numeric_cols
        cat_names = []
        if categorical_cols:
            # get categories from fitted OneHotEncoder
            ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
            cat_enc_names = ohe.get_feature_names_out(categorical_cols)
            cat_names = list(cat_enc_names)
        feature_names = num_names + cat_names
        importances = best_model.feature_importances_
        fi_series = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(20)
        fi_series.to_csv(os.path.join(SAVE_DIR, 'feature_importance.csv'))
        print(f"Top features saved to {os.path.join(SAVE_DIR, 'feature_importance.csv')}")
    except Exception as e:
        print(f"Feature importance skipped due to: {e}")

    # 11. Save pipeline (preprocessor + model) for deployment
    pipeline_full = {
        'preprocessor': preprocessor,
        'model': best_model
    }
    joblib.dump(pipeline_full, os.path.join(SAVE_DIR, 'smart_analytics_pipeline.pkl'))
    print(f"Saved preprocessing+model pipeline to {os.path.join(SAVE_DIR, 'smart_analytics_pipeline.pkl')}")

if __name__ == "__main__":
    # To test quickly, set use_synthetic=True
    main(use_synthetic=True)
