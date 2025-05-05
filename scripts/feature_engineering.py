# Feature Engineering Module
# =========================================
import pandas as pd
from pathlib import Path

def create_features(df):
    """
    Create new features based on domain knowledge.

    Features created:
    - BMI (Body Mass Index)
    - Pulse Pressure (SBP - DBP)
    - Mean Arterial Pressure (DBP + 1/3 Pulse Pressure)
    - Vision Average (average of left and right sight)
    - Hearing Average (average of left and right hearing)
    - AST/ALT Ratio (liver function marker)
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
    
    return df_new