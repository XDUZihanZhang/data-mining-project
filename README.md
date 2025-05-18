# Predicting Alcohol Consumption Patterns Using Health and Demographic Data

This project aims to predict individual alcohol consumption patterns based on various health indicators and demographic features, such as height, weight, waistline, blood pressure, liver function markers, and vision/hearing data.

We explore multiple machine learning models, including Logistic Regression, Random Forest, XGBoost, LightGBM and MLP, to perform binary classification (drinking vs. non-drinking).

## Dataset

The dataset used in this project can be downloaded from:  
- [Kaggle Download Link](https://www.kaggle.com/datasets/sooyoungher/smoking-drinking-dataset/data)  
- [Google Drive Download Link](https://drive.google.com/drive/folders/1RSMcVVyuzdNXgV9icpH8_hyDMmzcRu3X?usp=sharing)
---

## Project Structure

```plaintext
DATA-MINING-PROJECT/
│
├── config/                        # Configuration files (if any)
│
├── data/                          # Datasets
│   ├── raw/                       # Raw original data
│   ├── interim/                   # Intermediate cleaned data
│   └── processed/                 # Final datasets for modeling (train/val/test)
│
├── notebooks/                     # Jupyter notebooks for each pipeline step
│   ├── 0_data_exploration.ipynb
│   ├── 1_data_cleaning.ipynb
│   ├── 2_data_split.ipynb
│   ├── 3_feature_engineering.ipynb
│   ├── 4_preprocessing.ipynb
│   ├── 5.1_model_training_logistic_regression.ipynb
│   ├── 5.2_model_training_random_forest.ipynb
│   ├── 5.3_model_training_xgboost.ipynb
│   ├── 5.4_model_training_lightgbm.ipynb
│   └── 5.5_model_training_MLP.ipynb
│
├── scripts/                       # Reusable Python scripts
│   ├── evaluate_models.py
│   ├── feature_engineering.py
│   └── preprocessing.py
│
├── .gitignore                     # Files and folders ignored by Git
├── README.md                      # Project overview and instructions
```

## Workflow Overview

1. **Data Cleaning**  
   - Handle placeholder values.
   - Standardize data types and correct invalid entries.

2. **Data Splitting**  
   - Split the cleaned dataset into training, validation, and test sets before advanced preprocessing.

3. **Feature Engineering**  
   - Create domain-specific features:

4. **Preprocessing**  
   - Impute missing values.
   - Scale numerical features.
   - Encode categorical features.
   - Preprocessing specific to MLP

5. **Model Training and Evaluation**  
   - Five models:
     - Logistic Regression
     - Random Forest
     - XGBoost
     - LightGBM
     - MLP

## Authors

- **Guanglongjia Li** — *Data Exploration*
- **Lars Ostertag** - *Preprocessing, Training Logistic Regression Model*
- **Ola Hagerupsen** - *Preprocessing, Training Random Forest Model*
- **Zihan Zhang** — *Preprocessing, Training XGBoost and LightGBM Model*
- **Anyi Zhu** — *Preprocessing, Training MLP Model*

## License

This project is licensed under the **MIT License**.  
For more details, see the [LICENSE](LICENSE) file.