# scripts/preprocessing.py

# ====================================================
# Preprocessing Functions
# ----------------------------------------------------
# Functions for:
# - Missing value imputation
# - Scaling numeric features
# - Encoding categorical features
# - Model-specific preprocessing pipelines
# - Feature selection based on VIF
# ====================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor

# =========================================
# Shared preprocessing steps
# =========================================
# This function fits imputation only on the training set,
# and applies the same rule to validation/test sets.
# It is important to avoid data leakage by not using validation/test data

def impute_missing_values(X_train, X_val, X_test, strategy='median'):
    """
    Impute only numeric columns using the specified strategy.
    Categorical columns will be left unchanged.

    Parameters:
    - X_train, X_val, X_test: DataFrames
    - strategy: 'median' or 'mean'

    Returns:
    - X_train_imputed, X_val_imputed, X_test_imputed
    """
    # Detect strictly numeric columns
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Copy original data
    X_train_imp = X_train.copy()
    X_val_imp = X_val.copy()
    X_test_imp = X_test.copy()

    # Apply imputer only to numeric columns
    imputer = SimpleImputer(strategy=strategy)
    X_train_imp[numeric_cols] = imputer.fit_transform(X_train[numeric_cols])
    X_val_imp[numeric_cols] = imputer.transform(X_val[numeric_cols])
    X_test_imp[numeric_cols] = imputer.transform(X_test[numeric_cols])

    return X_train_imp, X_val_imp, X_test_imp

# =========================================
# Logistic Regression specific preprocessing
# =========================================

def preprocess_for_logistic(X_train, X_val, X_test, categorical_cols):
    """
    Preprocess data for Logistic Regression:
    - Standardize numeric features
    - One-hot encode categorical features (drop first)
    """
    from sklearn.preprocessing import StandardScaler, OneHotEncoder

    numeric_cols = [col for col in X_train.columns if col not in categorical_cols]

    # Scale numeric features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train[numeric_cols]), columns=numeric_cols)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val[numeric_cols]), columns=numeric_cols)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test[numeric_cols]), columns=numeric_cols)

    # Encode categorical features
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
    X_train_encoded = pd.DataFrame(encoder.fit_transform(X_train[categorical_cols]), columns=encoder.get_feature_names_out(categorical_cols))
    X_val_encoded = pd.DataFrame(encoder.transform(X_val[categorical_cols]), columns=encoder.get_feature_names_out(categorical_cols))
    X_test_encoded = pd.DataFrame(encoder.transform(X_test[categorical_cols]), columns=encoder.get_feature_names_out(categorical_cols))

    # Concatenate
    X_train_final = pd.concat([X_train_scaled, X_train_encoded], axis=1)
    X_val_final = pd.concat([X_val_scaled, X_val_encoded], axis=1)
    X_test_final = pd.concat([X_test_scaled, X_test_encoded], axis=1)

    return X_train_final, X_val_final, X_test_final

# =========================================
# Feature Selection based on VIF
# =========================================

def select_low_vif_features(X, threshold=5.0):
    """
    Select features with VIF lower than the specified threshold.
    
    Parameters:
    - X: DataFrame of features
    - threshold: maximum VIF allowed
    
    Returns:
    - X_selected: DataFrame with selected features
    """
    X_temp = X.copy()
    dropped = True

    while dropped:
        dropped = False
        vif = pd.DataFrame()
        vif["feature"] = X_temp.columns
        vif["VIF"] = [variance_inflation_factor(X_temp.values, i) for i in range(X_temp.shape[1])]

        max_vif = vif["VIF"].max()
        if max_vif > threshold:
            feature_to_drop = vif.sort_values("VIF", ascending=False)["feature"].iloc[0]
            X_temp = X_temp.drop(columns=[feature_to_drop])
            dropped = True

    return X_temp


# =========================================
# Random Forest & XGBoost preprocessing
# =========================================

def preprocess_for_tree_models(X_train, X_val, X_test):
    """
    Preprocessing pipeline for Random Forest and XGBoost:
    - No scaling needed
    - Use imputed data directly
    
    Returns:
    - X_train, X_val, X_test (already imputed)
    """
    return X_train.copy(), X_val.copy(), X_test.copy()

# =========================================
# LightGBM preprocessing
# =========================================

def preprocess_for_lightgbm(X_train, X_val, X_test, categorical_cols):
    """
    Preprocessing pipeline for LightGBM:
    - Set categorical columns' dtype to 'category'
    
    Returns:
    - X_train, X_val, X_test with categorical columns correctly typed
    """
    X_train_lgb = X_train.copy()
    X_val_lgb = X_val.copy()
    X_test_lgb = X_test.copy()
    
    for col in categorical_cols:
        for df in [X_train_lgb, X_val_lgb, X_test_lgb]:
            df[col] = df[col].astype('category')
    
    return X_train_lgb, X_val_lgb, X_test_lgb
