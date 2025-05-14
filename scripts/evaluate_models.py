
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# =============================
# Error Sample Analysis
# =============================
def analyze_error_samples(X_val, y_val, y_pred, error_type='FN'):
    """
    Analyze feature distributions for error samples in a validation set.
    error_type: 'FN' for false negatives, 'FP' for false positives.
    Returns descriptive statistics and feature importance (difference between error and correct samples).
    """
    if error_type == 'FN':
        mask = (y_val == 1) & (y_pred == 0)
    else:  # FP
        mask = (y_val == 0) & (y_pred == 1)
    error_samples = X_val[mask]
    correct_samples = X_val[~mask]
    error_stats = error_samples.describe()
    correct_stats = correct_samples.describe()
    error_importance = pd.DataFrame({
        'feature': X_val.columns,
        'error_mean': error_samples.mean(),
        'correct_mean': correct_samples.mean(),
        'diff': error_samples.mean() - correct_samples.mean()
    })
    return error_stats, correct_stats, error_importance

# =============================
# Feature Distribution Visualization
# =============================
def plot_feature_distributions(X_val, y_val, y_pred, error_type='FN', top_n=3, feature_list=None):
    """
    Visualize the distributions of the top N features where error and correct samples differ most.
    If feature_list is provided, plot those features instead of selecting by mean difference.
    """
    _, _, error_importance = analyze_error_samples(X_val, y_val, y_pred, error_type=error_type)
    if feature_list is None:
        top_features = error_importance['diff'].abs().sort_values(ascending=False).head(top_n).index.tolist()
    else:
        top_features = feature_list
    if error_type == 'FN':
        mask = (y_val == 1) & (y_pred == 0)
    else:
        mask = (y_val == 0) & (y_pred == 1)
    error_samples = X_val[mask]
    correct_samples = X_val[~mask]
    for feat in top_features:
        plt.figure(figsize=(6, 4))
        sns.kdeplot(error_samples[feat], label=error_type, fill=True, color='red', alpha=0.5)
        sns.kdeplot(correct_samples[feat], label='Correct', fill=True, color='blue', alpha=0.5)
        plt.title(f'Feature Distribution: {feat} ({error_type} vs Correct)')
        plt.xlabel(feat)
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        plt.show()

# =============================
# Error Type Visualization (Confusion Matrix & PCA)
# =============================
def visualize_error_types(X_val, y_val, y_pred):
    """
    Visualize error type distribution and PCA projection for error types.
    """
    TP_idx = (y_val == 1) & (y_pred == 1)
    TN_idx = (y_val == 0) & (y_pred == 0)
    FP_idx = (y_val == 0) & (y_pred == 1)
    FN_idx = (y_val == 1) & (y_pred == 0)
    val_error_df = X_val.copy().reset_index(drop=True)
    val_error_df['error_type'] = 'Other'
    val_error_df.loc[TP_idx.values, 'error_type'] = 'TP'
    val_error_df.loc[TN_idx.values, 'error_type'] = 'TN'
    val_error_df.loc[FP_idx.values, 'error_type'] = 'FP'
    val_error_df.loc[FN_idx.values, 'error_type'] = 'FN'
    plt.figure(figsize=(6, 4))
    sns.countplot(data=val_error_df, x='error_type', order=['TP', 'TN', 'FP', 'FN'])
    plt.title('Error Type Distribution (Validation Set)')
    plt.xlabel('Error Type')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()
    pca = PCA(n_components=2)
    X_val_pca = pca.fit_transform(X_val)
    val_error_df['pca1'] = X_val_pca[:, 0]
    val_error_df['pca2'] = X_val_pca[:, 1]
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=val_error_df, x='pca1', y='pca2', hue='error_type', alpha=0.4, palette='Set2')
    plt.title('PCA Projection of Validation Samples by Error Type')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.legend()
    plt.tight_layout()
    plt.show()

# =============================
# Error Pattern Analysis (Pairplot)
# =============================
def analyze_error_patterns(X_val, y_val, y_pred, feature_importance, top_n=3):
    """
    Analyze patterns in error samples using top important features (pairplot).
    """
    top_features = feature_importance.head(top_n)['feature'].tolist() if isinstance(feature_importance, pd.DataFrame) else feature_importance[:top_n]
    error_type = pd.Series('Correct', index=X_val.index)
    error_type[(y_val == 1) & (y_pred == 0)] = 'FN'
    error_type[(y_val == 0) & (y_pred == 1)] = 'FP'
    plot_data = X_val[top_features].copy()
    plot_data['error_type'] = error_type
    sns.pairplot(plot_data, hue='error_type', diag_kind='kde')
    plt.show()
    return plot_data

# =============================
# Local Explanation for Error Samples (SHAP)
# =============================
def analyze_local_explanations(X_val, y_val, y_pred, explainer, n_samples=3):
    """
    Generate local SHAP explanations for random FN and FP samples.
    """
    import shap
    def convert_object_to_category(df):
        object_cols = df.select_dtypes(include=['object']).columns
        for col in object_cols:
            df[col] = df[col].astype('category')
        return df
    X_val = convert_object_to_category(X_val)
    fn_indices = X_val[(y_val == 1) & (y_pred == 0)].index
    fp_indices = X_val[(y_val == 0) & (y_pred == 1)].index
    fn_sample_indices = np.random.choice(fn_indices, size=min(n_samples, len(fn_indices)), replace=False)
    fp_sample_indices = np.random.choice(fp_indices, size=min(n_samples, len(fp_indices)), replace=False)
    shap.initjs()
    for idx in fn_sample_indices:
        sample = X_val.loc[[idx]]
        print(f"\nFN Sample {idx}:")
        shap.force_plot(
            explainer.expected_value,
            explainer.shap_values(sample),
            sample,
            matplotlib=True
        )
    for idx in fp_sample_indices:
        sample = X_val.loc[[idx]]
        print(f"\nFP Sample {idx}:")
        shap.force_plot(
            explainer.expected_value,
            explainer.shap_values(sample),
            sample,
            matplotlib=True
        )

# =============================
# Complete Error Analysis Pipeline
# =============================
def run_error_analysis(X_val, y_val, y_pred, explainer=None, shap_top_n=5):
    """
    Run complete error analysis pipeline. If explainer is provided, will also generate SHAP-based plots.
    Returns a dictionary of analysis results.
    """
    print("Visualizing Error Types (Confusion Matrix & PCA)...")
    visualize_error_types(X_val, y_val, y_pred)
    print("\n Analyzing Error Samples (FN and FP)...")
    fn_stats, fn_correct_stats, fn_importance = analyze_error_samples(X_val, y_val, y_pred, 'FN')
    fp_stats, fp_correct_stats, fp_importance = analyze_error_samples(X_val, y_val, y_pred, 'FP')
    print("\n Plotting Feature Distributions (FN)...")
    plot_feature_distributions(X_val, y_val, y_pred, error_type='FN', top_n=3)
    print("\n Plotting Feature Distributions (FP)...")
    plot_feature_distributions(X_val, y_val, y_pred, error_type='FP', top_n=3)
    feature_importance = None
    if explainer is not None:
        print("\n SHAP Feature Importance and Distributions...")
        shap_values = explainer.shap_values(X_val)
        mean_shap = np.abs(shap_values).mean(axis=0)
        feature_importance = pd.DataFrame({
            'feature': X_val.columns,
            'importance': mean_shap
        }).sort_values('importance', ascending=False)
        plot_feature_distributions(X_val, y_val, y_pred, error_type='FN', top_n=shap_top_n, feature_list=feature_importance.head(shap_top_n)['feature'].tolist())
        plot_feature_distributions(X_val, y_val, y_pred, error_type='FP', top_n=shap_top_n, feature_list=feature_importance.head(shap_top_n)['feature'].tolist())
        print("\n6. Analyzing Error Patterns (Pairplot)...")
        analyze_error_patterns(X_val, y_val, y_pred, feature_importance, top_n=3)
        print("\n7. Generating Local Explanations (SHAP)...")
        analyze_local_explanations(X_val, y_val, y_pred, explainer)
    return {
        'fn_stats': fn_stats,
        'fp_stats': fp_stats,
        'fn_importance': fn_importance,
        'fp_importance': fp_importance,
        'feature_importance': feature_importance
    }