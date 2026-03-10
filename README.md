# 🥮 Nata Visionaries: Machine Learning Quality Prediction

## 📌 Project Overview
This project was developed to predict the quality of Portugal's famous Pastéis de Nata (Portuguese Custard Tarts) based on ingredient proportions and baking processes. Using a dataset collected from traditional bakeries in Lisbon and Porto, the goal was to build a machine learning model capable of classifying a pastry's quality as either **"OK"** (meets exceptional standards) or **"KO"** (subpar quality). 

By successfully predicting pastry quality, this model prevents unnecessary physical quality testing ("destruction without consumption") and provides actionable insights for bakeries to optimize their recipes.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Scikit-Learn, Pandas, NumPy, Seaborn, Matplotlib, SciPy
* **Techniques:** Exploratory Data Analysis (EDA), Feature Engineering, Ensemble Feature Selection, Hyperparameter Tuning (GridSearchCV)

## 🧠 Methodology & Pipeline

### 1. Exploratory Data Analysis (EDA)
* Analyzed over 5,000 production records containing both numerical (temperatures, durations, ingredient concentrations) and categorical features.
* Identified extreme physical anomalies (e.g., sugar content exceeding 100g/100g) and highly skewed distributions.
* Conducted multivariate analysis, revealing a distinct failure mode: baking durations exceeding 50 minutes consistently resulted in "KO" pastries (dried out/burnt).

### 2. Preprocessing & Feature Engineering
* Built a robust preprocessing pipeline evaluating multiple combinations of imputers, outlier handling, and scalers across baseline models.
* **Final Preprocessing Strategy:** Applied the IQR method for outliers, Median Imputation for missing values, Log1p transformations for skewed data, and RobustScaler for normalization.
* **Feature Engineering:** Created derived features to capture baking dynamics, such as `cream_x_egg`, `sugar_per_egg`, `total_stress` (duration × temperature), and an `is_burnt` binary flag.

### 3. Feature Selection
* Implemented a strict "Ensemble Voting" feature selection process to eliminate noise and prevent overfitting.
* Features were evaluated using four different methods: **Spearman Correlation, Recursive Feature Elimination (RFE) with Logistic Regression, RFE with Random Forest, and LassoCV**.
* Only features that survived a majority vote were retained for the final modeling phase.

### 4. Modeling & Hyperparameter Tuning
* Evaluated 9 distinct algorithms, ranging from basic Logistic Regression and KNN to complex ensembles like AdaBoost, Random Forest, and a custom Stacking Classifier.
* **Random Forest** and **Gradient Boosting** emerged as the top performers. 
* Conducted exhaustive hyperparameter tuning via `GridSearchCV` to minimize the generalization gap (overfitting). 

## 📊 Results
The **Gradient Boosting Classifier** was selected as the final model due to its optimal balance between training performance and robust validation generalization. 
* **Validation Accuracy:** ~78%
* **Validation F1-Score (Macro):** ~0.76
* **Kaggle Submission Accuracy:** Achieved a competitive final score of 0.791 on unseen test data.

## 📂 Repository Structure
* `ML07_NB1_DATAEXPLORATION.ipynb`: Initial EDA, distribution checks, and multivariate visualizations.
* `ML07_NB2_PREP_FEAT_MODEL.ipynb`: Combined pipeline covering exhaustive preprocessing evaluation, feature selection voting, and multi-model benchmarking.
* `ML07_NB9_FINAL.ipynb`: The final inference notebook applying the optimal preprocessing pipeline and tuned Gradient Boosting model to generate predictions for Kaggle.

---
*Project developed by Group 7: Carolina Arez, Henrique Madureira, Miguel Melo, Noa Penas for the Machine Learning I course at Nova Information Management School (2025-26).*