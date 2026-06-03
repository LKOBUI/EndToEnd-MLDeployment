import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc,precision_score,recall_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score,f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import argparse
import warnings
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pickle
from pathlib import Path
import joblib
warnings.filterwarnings("ignore", category=UserWarning)
import sys
sys.stdout.reconfigure(encoding='utf-8')
from mlflow import MlflowClient
from mlflow.entities import ViewType

# Function: drop rows with non-numeric values
def drop_non_numeric(df_frame):
    cleaned_df = df_frame.copy()
    for col in cleaned_df.columns:
        # Try to convert column to numeric
        cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
    # Drop rows where conversion failed (NaN introduced)
    cleaned_df = cleaned_df.dropna()
    return cleaned_df

def categorize_chol(val):
    if val < 200:
        return "Normal", 0
    elif 200 <= val <= 239:
        return "Medium", 1
    else:
        return "High", 1
 
def get_cleaned_data():
    # Get current working directory
    cwd = os.getcwd()
    # Specify dataset filename
    filename = "dataset_2190_cholesterol.csv"
    file_path = os.path.join(cwd, filename)
    df = pd.read_csv(file_path)
    null_counts = df.isnull().sum()
    #if null_counts.sum() > 0:
    #    print("\n✅ Null values are present in the dataset.")
    #else:
    #    print("\n❌ No null values found in the dataset.")
    # Apply cleaning
    df_clean = drop_non_numeric(df)
    #print("Cleaned shape:", df_clean.shape)
    df_clean[['chol_category_label', 'chol_category_code']] = df_clean['chol'].apply(
        lambda x: pd.Series(categorize_chol(x))
        )
    # Define columns and target
    columns = ['age', 'sex', 'cp', 'trestbps', 'fbs', 'restecg', 'thalach', 'exang',
           'oldpeak', 'slope', 'ca', 'thal', 'num', 'chol_category_code']
    target = "chol_category_code"
    binary_vars = ['sex', 'fbs', 'exang']
    # Separate features and target
    X = df_clean[columns].drop(columns=[target])
    y = df_clean[target]
    
    # Identify numeric columns to scale (exclude binary + target)
    numeric_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in binary_vars]
    
    # Apply StandardScaler
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    # Store result in df_scale (features + target)
    df_scale = X_scaled.copy()
    df_scale[target] = y
    
    #print("Scaled dataset preview:")
    #print(df_scale.columns)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    #print("\nTrain set shape:", X_train.shape, y_train.shape)
    #print("Test set shape:", X_test.shape, y_test.shape)
    return X_train, X_test, y_train, y_test


def logistic_model_1(X_train, X_test, y_train, y_test):
    # Logistic Regression with elasticnet penalty, saga solver
    # 1. saga + elasticnet + l1_ratio=0.1, C=10
    C_value = 10
    L_1_ratio = 0.1
    Solver = 'saga'
    Panality = 'elasticnet'
    model = LogisticRegression(
    penalty=Panality,
    solver=Solver,
    l1_ratio=L_1_ratio,
    C=C_value,
    max_iter=10000
    )
    model.fit(X_train, y_train)

    # Predictions
    y_pred_prob = model.predict_proba(X_test)[:, 1]  # probability of class 1
    y_pred_class = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred_class)
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    mse = mean_squared_error(y_test, y_pred_prob)
    mae = mean_absolute_error(y_test, y_pred_prob)

    # Store results in DataFrame
    results_df = pd.DataFrame([{
        "C": C_value,
        "l1_ratio": L_1_ratio,
        'solver': Solver,
        'penalty' : Panality,
        "Accuracy": acc,
        "ROC_AUC": roc_auc,
        "MSE": mse,
        "MAE": mae
    }])
    print("================ Logistice Model ================")
    print(f'Logistice accureacy {acc}')
    return model, results_df


def logistic_model_2(X_train, X_test, y_train, y_test):
    # Logistic Regression with elasticnet penalty, saga solver
    # 1. saga + elasticnet + l1_ratio=0.1, C=10
    C_value = 100
    L_1_ratio = 0.4
    Solver = 'saga'
    Panality = 'elasticnet'
    model = LogisticRegression(
    penalty=Panality,
    solver=Solver,
    l1_ratio=L_1_ratio,
    C=C_value,
    max_iter=10000
    )
    model.fit(X_train, y_train)

    # Predictions
    y_pred_prob = model.predict_proba(X_test)[:, 1]  # probability of class 1
    y_pred_class = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred_class)
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    mse = mean_squared_error(y_test, y_pred_prob)
    mae = mean_absolute_error(y_test, y_pred_prob)

    # Store results in DataFrame
    results_df = pd.DataFrame([{
        "C": C_value,
        "l1_ratio": L_1_ratio,
        'solver': Solver,
        'penalty' : Panality,
        "Accuracy": acc,
        "ROC_AUC": roc_auc,
        "MSE": mse,
        "MAE": mae
    }])
    print("================ Logistice Model ================")
    print(f'Logistice accureacy {acc}')
    return model, results_df

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

# Uses of set_tracking_uri():
mlflow.set_tracking_uri("http://127.0.0.1:5000")

client = MlflowClient()
run_id = "bfd36ebdd74e4973aeac6dfc8a5a969d"
print(f'Access rin id {run_id}')
run = client.get_run(run_id)

# Check lifecycle stage
if run.info.lifecycle_stage != "active":
    raise Exception(f"Run {run_id} is not active. Current stage: {run.info.lifecycle_stage}")

print(f"Run tags: {run.data.tags}")
print(f"Experiment id: {run.info.experiment_id}")
print(f"Run id: {run.info.run_id}")
print(f"Run name: {run.info.run_name}")
print(f"lifecycle_stage: {run.info.lifecycle_stage}")
print(f"status: {run.info.status}")

run = client.create_run (
    experiment_id=run.info.experiment_id,
    tags={
        "Version": "v1",
        "Priority": "P1"
    },
    run_name="run from client 2"
)
print('=========== After update Tages ============')
print(f"Run tags: {run.data.tags}")
print(f"Experiment id: {run.info.experiment_id}")
print(f"Run id: {run.info.run_id}")
print(f"Run name: {run.info.run_name}")
print(f"lifecycle_stage: {run.info.lifecycle_stage}")
print(f"status: {run.info.status}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    # Load data
    X_train, X_test, y_train, y_test = get_cleaned_data()
    model, results_df = logistic_model_1(X_train, X_test, y_train, y_test)
    predicted_qualities = model.predict(X_test)
    (rmse, mae, r2) = eval_metrics(y_test, predicted_qualities)
    joblib.dump(model, "linear_regression.pkl")
    # Log parameters
    param = ["C", "l1_ratio", "solver",'penalty']        
    C_val = client.log_param(run.info.run_id, param[0], results_df.loc[0, param[0]])
    l1_ratio_val = client.log_param(run.info.run_id, param[1], results_df.loc[0, param[1]])
    solver_val = client.log_param(run.info.run_id, param[2], results_df.loc[0, param[2]])
    penality_val = client.log_param(run.info.run_id, param[3], results_df.loc[0, param[3]])

    # Log metrics
    for metric in ["Accuracy", "ROC_AUC", "MSE", "MAE"]:
        if metric in results_df.columns:
            client.log_metric(run.info.run_id,metric, results_df.loc[0, metric])
    
    client.log_artifact(run.info.run_id, "linear_regression.pkl")
    client.log_artifact(run.info.run_id,"dataset_2190_cholesterol.csv")
    print("Logged parameter values:")
    print(f"C_val: {C_val}")
    print(f"l1_ratio_val: {l1_ratio_val}")
    print(f"solver_val: {solver_val}")
    print(f"penality_val: {penality_val}")
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)  

    #Mlflow Client Update
    # ✅ Update parameters with new values
    #  Updates the run record in the MLflow Tracking Serve
    # client.update_run(run.info.run_id, status='FINISHED', name='Mlflow Client Run')
    # A fundamental MLflow rule: once a parameter is logged for a run, 
    # its value cannot be changed. 
    # Correct ways to handle updates creates a separate run with the new parameters and metrics.
    # i.e. mlflow.start_run()
   