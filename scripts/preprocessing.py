# ====================================================
# Preprocessing Functions
# ----------------------------------------------------
# Functions for:
# - Missing value imputation
# - Scaling numeric features
# - Encoding categorical features
# - Model-specific preprocessing pipelines
# - Feature selection based on VIF and t-SNE
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
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

    X_train_imp = X_train.copy()
    X_val_imp = X_val.copy()
    X_test_imp = X_test.copy()

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
    - Remove outliers from numeric columns (using training set quantiles)
    - No scaling needed
    - Use imputed data directly
    
    Returns:
    - X_train, X_val, X_test (already imputed and outliers handled)
    """
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Use training set quantiles to clip outliers for all sets
    def remove_outliers_consistent(df, ref_df, columns, k=1.5):
        for column in columns:
            q1 = ref_df[column].quantile(0.25)
            q3 = ref_df[column].quantile(0.75)
            iqr = q3 - q1
            df[column] = df[column].clip(lower=q1 - k * iqr, upper=q3 + k * iqr)
        return df

    X_train_out = remove_outliers_consistent(X_train.copy(), X_train, numeric_cols)
    X_val_out = remove_outliers_consistent(X_val.copy(), X_train, numeric_cols)
    X_test_out = remove_outliers_consistent(X_test.copy(), X_train, numeric_cols)

    return X_train_out, X_val_out, X_test_out

# =========================================
# LightGBM preprocessing
# =========================================

def preprocess_for_lightgbm(X_train, X_val, X_test, categorical_cols):
    """
    Preprocessing pipeline for LightGBM:
    - Remove outliers from numeric columns (using training set quantiles)
    - Set categorical columns' dtype to 'category'
    
    Returns:
    - X_train, X_val, X_test with categorical columns correctly typed and outliers handled
    """
    X_train_lgb = X_train.copy()
    X_val_lgb = X_val.copy()
    X_test_lgb = X_test.copy()
    
    numeric_cols = X_train_lgb.select_dtypes(include=['int64', 'float64']).columns.tolist()
    def remove_outliers_consistent(df, ref_df, columns, k=1.5):
        for column in columns:
            q1 = ref_df[column].quantile(0.25)
            q3 = ref_df[column].quantile(0.75)
            iqr = q3 - q1
            df[column] = df[column].clip(lower=q1 - k * iqr, upper=q3 + k * iqr)
        return df
    X_train_lgb = remove_outliers_consistent(X_train_lgb, X_train_lgb, numeric_cols)
    X_val_lgb = remove_outliers_consistent(X_val_lgb, X_train_lgb, numeric_cols)
    X_test_lgb = remove_outliers_consistent(X_test_lgb, X_train_lgb, numeric_cols)

 
    for col in categorical_cols:
        for df in [X_train_lgb, X_val_lgb, X_test_lgb]:
            df[col] = df[col].astype('category')
    
    return X_train_lgb, X_val_lgb, X_test_lgb

# =========================================
# Dimensionality Reduction
# =========================================

def apply_tsne(X, y=None, n_components=2, sample_size=5000, random_state=42, 
               perplexity=30, n_iter=1000, learning_rate='auto'):
    """
    Apply t-SNE dimensionality reduction to the data.
    
    Parameters:
    -----------
    X : array-like or DataFrame
        Input features
    y : array-like, optional
        Target variable for coloring the plot
    n_components : int, default=2
        Number of dimensions for t-SNE output
    sample_size : int, default=5000
        Number of samples to use (t-SNE is computationally expensive)
    random_state : int, default=42
        Random seed for reproducibility
    perplexity : int, default=30
        t-SNE perplexity parameter
    n_iter : int, default=1000
        Number of iterations for t-SNE
    learning_rate : str or float, default='auto'
        Learning rate for t-SNE
        
    Returns:
    --------
    tuple
        - X_tsne : array-like
            t-SNE transformed data
        - tsne_df : DataFrame
            DataFrame containing t-SNE results and target variable (if provided)
    """
    from sklearn.manifold import TSNE
    import pandas as pd
    import numpy as np
    
    # Convert to numpy array if DataFrame
    if isinstance(X, pd.DataFrame):
        X = X.values
    
    # Sample the data if needed
    if len(X) > sample_size:
        indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[indices]
        y_sample = y[indices] if y is not None else None
    else:
        X_sample = X
        y_sample = y
    
    # Apply t-SNE
    tsne = TSNE(
        n_components=n_components,
        random_state=random_state,
        perplexity=perplexity,
        n_iter=n_iter,
        learning_rate=learning_rate
    )
    
    X_tsne = tsne.fit_transform(X_sample)
    
    # Create DataFrame with results
    tsne_df = pd.DataFrame({
        f'TSNE{i+1}': X_tsne[:, i] for i in range(n_components)
    })
    
    if y_sample is not None:
        tsne_df['Target'] = y_sample
    
    return X_tsne, tsne_df

def plot_tsne(tsne_df, target_col='Target', figsize=(10, 8), title='t-SNE Visualization'):
    """
    Plot t-SNE results.
    
    Parameters:
    -----------
    tsne_df : DataFrame
        DataFrame containing t-SNE results and target variable
    target_col : str, default='Target'
        Name of the target column for coloring
    figsize : tuple, default=(10, 8)
        Figure size
    title : str, default='t-SNE Visualization'
        Plot title
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=figsize)
    sns.scatterplot(
        data=tsne_df,
        x='TSNE1',
        y='TSNE2',
        hue=target_col,
        palette='Set1',
        alpha=0.7
    )
    plt.title(title)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(title=target_col)
    plt.grid(True, alpha=0.3)
    plt.show()

def save_tsne_results(tsne_df, output_path):
    """
    Save t-SNE results to CSV file.
    
    Parameters:
    -----------
    tsne_df : DataFrame
        DataFrame containing t-SNE results
    output_path : str or Path
        Path to save the results
    """
    tsne_df.to_csv(output_path, index=False)
    print(f"t-SNE results saved to {output_path}")
