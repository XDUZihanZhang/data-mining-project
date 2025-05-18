# Feature Engineering
import pandas as pd
from pathlib import Path
import pandas as pd

def create_features(df):
    """
    Create new features based on domain knowledge, including metabolic risk indicators.

    Features created:
    - BMI (Body Mass Index)
    - Pulse Pressure (SBP - DBP)
    - Mean Arterial Pressure (DBP + 1/3 Pulse Pressure)
    - Vision Average (average of left and right sight)
    - Hearing Average (average of left and right hearing)
    - AST/ALT Ratio (liver function marker)
    - Metabolic risk indicators (waist, blood pressure, triglyceride, cholesterol, glucose)
    - Metabolic risk count (sum of all risk indicators)
    """
    df_new = df.copy()
    
    # BMI: weight (kg) / (height (m))^2
    df_new['BMI'] = df_new['weight'] / ((df_new['height'] / 100) ** 2)
    
    # Pulse Pressure: Systolic BP - Diastolic BP
    df_new['pulse_pressure'] = df_new['SBP'] - df_new['DBP']
    
    # Mean Arterial Pressure: DBP + 1/3 * Pulse Pressure
    df_new['mean_arterial_pressure'] = df_new['DBP'] + (df_new['pulse_pressure'] / 3)
    
    # Vision Average: average of sight_left and sight_right
    df_new['vision_avg'] = (df_new['sight_left'] + df_new['sight_right']) / 2
    
    # Hearing Average: average of hear_left and hear_right
    df_new['hearing_avg'] = (df_new['hear_left'] + df_new['hear_right']) / 2
    
    # AST/ALT Ratio: Liver function indicator
    df_new['AST_ALT_ratio'] = df_new['SGOT_AST'] / df_new['SGOT_ALT']
    
    # Metabolic risk indicators
    waist_thresh = df_new['sex'].map({'Male': 90, 'Female': 80})
    df_new['risk_waist'] = (df_new['waistline'] > waist_thresh).astype(int)
    df_new['risk_bp'] = ((df_new['SBP'] >= 130) | (df_new['DBP'] >= 85)).astype(int)
    df_new['risk_tg'] = (df_new['triglyceride'] >= 150).astype(int)
    df_new['risk_chole'] = (df_new['tot_chole'] >= 200).astype(int)
    df_new['risk_glu'] = (df_new['BLDS'] >= 100).astype(int)
    
    # Metabolic risk count (sum of all risk indicators)
    df_new['metabolic_risk_count'] = (
        df_new['risk_waist'] +
        df_new['risk_bp'] +
        df_new['risk_tg'] +
        df_new['risk_chole'] +
        df_new['risk_glu']
    )
    
    return df_new