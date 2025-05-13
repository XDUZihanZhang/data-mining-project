# Predicting Alcohol Consumption Patterns Using Health and Demographic Data

This project aims to predict individual alcohol consumption patterns based on various health indicators and demographic features, such as height, weight, waistline, blood pressure, liver function markers, and vision/hearing data.

We explore multiple machine learning models, including Logistic Regression, Random Forest, XGBoost, and LightGBM, to perform binary classification (drinking vs. non-drinking).

---

## Project Structure

```plaintext
data-mining-project/
│
├── README.md               # Project overview and instructions
├── requirements.txt        # Python dependencies
├── .gitignore              # Files and folders ignored by Git
│
├── data/                   # Datasets
│   ├── raw/                # Raw original data
│   ├── interim/            # Intermediate cleaned data
│   ├── processed/          # Final datasets for modeling (train/val/test)
│
├── notebooks/              # Exploratory and modeling notebooks
│   ├── 01_data_cleaning.ipynb
│   ├── 02_data_split.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_preprocessing.ipynb
│   ├── 05_model_training.ipynb
│   ├── 06_model_evaluation.ipynb
│   ├── 07_feature_importance.ipynb
│
├── scripts/                # Reusable Python modules
│   ├── data_cleaning.py
│   ├── split_dataset.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── train_models.py
│   ├── evaluate_models.py
│
├── models/                 # Saved trained model artifacts
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   ├── lightgbm_model.pkl
│
├── results/                # Outputs and analysis
│   ├── figures/            # Visualizations (heatmaps, ROC curves, etc.)
│   ├── tables/             # Metrics tables and summaries
│   ├── reports/            # Project reports and conclusions
│
└── config/                 # Configuration files (YAML format)
    ├── params_logistic.yaml
    ├── params_randomforest.yaml
    ├── params_xgboost.yaml
    ├── params_lightgbm.yaml
```

## Workflow Overview

1. **Data Cleaning**  
   - Handle placeholder values (e.g., 999, 9.9).
   - Standardize data types and correct invalid entries.

2. **Data Splitting**  
   - Split the cleaned dataset into training, validation, and test sets before advanced preprocessing.

3. **Feature Engineering**  
   - Create new features:
     - BMI = weight / (height/100)^2
     - Pulse pressure = SBP - DBP
     - Mean arterial pressure = DBP + (Pulse pressure / 3)
     - Vision average = (sight_left + sight_right) / 2
     - AST/ALT ratio = SGOT_AST / SGOT_ALT

4. **Preprocessing**  
   - Impute missing values (e.g., waistline imputed by sex group median).
   - Scale numerical features (StandardScaler for Logistic Regression only).
   - Encode categorical features (One-Hot Encoding for `sex`).

5. **Model Training and Evaluation**  
   - Train four models:
     - Logistic Regression
     - Random Forest
     - XGBoost
     - LightGBM
   - Evaluate models using:
     - Accuracy
     - F1-score
     - ROC-AUC

6. **Feature Importance Analysis**  
   - Analyze feature importances using:
     - Logistic Regression coefficients
     - Tree-based feature importances
     - SHAP values

## Authors

- **Anyi Zhu** — *distribution*
- **Guanglongjia Li** — *distribution*
- **Zihan Zhang** — *distribution*
- **Ola Hagerupsen** - *distribution*
- **Lars Ostertag** - *distribution*

## License

This project is licensed under the **MIT License**.  
For more details, see the [LICENSE](LICENSE) file.