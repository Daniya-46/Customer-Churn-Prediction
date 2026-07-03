# Telecom Customer Churn Prediction
This project focuses on predicting customer churn—a critical business metric for telecom and SaaS companies. By analyzing customer demographics, account information, and service usage, we build a machine learning model capable of identifying "at-risk" customers before they cancel their subscriptions.

🚀 Key Objectives
EDA & Storytelling: Uncover behavioral patterns (e.g., how contract types and tenure impact churn).

Data Cleaning: Handle missing values (e.g., hidden spaces in charges) and remove redundant data.

Feature Engineering: Transform noisy continuous data (tenure) into meaningful categorical groups.

Model Optimization: Address the inherent class imbalance of churn data using scale_pos_weight and perform hyperparameter tuning with GridSearchCV.

🛠 Tech Stack
Language: Python

Libraries: pandas, numpy, matplotlib, seaborn (for EDA), scikit-learn, xgboost (for modeling).

📊 Pipeline Overview
EDA Phase: Visualized churn distributions, correlation heatmaps, and categorical behavior.

Data Cleaning: Addressed inconsistencies and encoded categorical text into binary math.

Modeling: Performed a "bake-off" between Logistic Regression, Random Forest, and XGBoost.

Tuning: Optimized the champion model (XGBoost) using GridSearchCV to maximize Recall for churned customers.

📈 Key Results
After hyperparameter tuning, the XGBoost model achieved a Recall of 0.83 on churned customers, meaning the model successfully identifies 83% of users at risk of leaving, allowing the business to proactively intervene with retention strategies.

📂 Project Structure
customer_churn_prediction_3.ipynb: The complete end-to-end Jupyter Notebook.

Telco-Customer-Churn.csv: The raw dataset.
