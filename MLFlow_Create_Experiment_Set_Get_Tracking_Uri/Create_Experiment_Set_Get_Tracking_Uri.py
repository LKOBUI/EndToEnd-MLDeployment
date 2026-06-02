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
warnings.filterwarnings("ignore", category=UserWarning)

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

def glm_binomial(X_train, X_test, y_train, y_test, threshold=0.5):
    # Add constant term for intercept
    X_train_const = sm.add_constant(X_train)
    X_test_const = sm.add_constant(X_test)

    # Fit GLM with binomial family (logistic regression)
    glm_binom = sm.GLM(y_train, X_train_const, family=sm.families.Binomial())
    glm_result = glm_binom.fit()

    # Predict probabilities on test set
    y_pred_prob = glm_result.predict(X_test_const)
    y_pred_class = (y_pred_prob >= threshold).astype(int)

    # Collect metrics
    metrics = {
        "MSE": mean_squared_error(y_test, y_pred_prob),
        "MAE": mean_absolute_error(y_test, y_pred_prob),
        "R2": r2_score(y_test, y_pred_prob),
        "ROC_AUC": roc_auc_score(y_test, y_pred_prob),
        "Accuracy": accuracy_score(y_test, y_pred_class)
    }

    # Return structured output
    results = {
        "Parameters": {"threshold": threshold},
        "Metrics": metrics,
        "Model": glm_result
    }
    print('================= GLM RESULT =======================')
    print(f"Model Accuracy: {results['Metrics']['Accuracy']:.4f}")
    return results

def logistic_model(X_train, X_test, y_train, y_test, C_value, l1_ratio=0.912):
    # Logistic Regression with elasticnet penalty, saga solver
    model = LogisticRegression(
        penalty='elasticnet',
        solver='saga',        # saga supports elasticnet
        l1_ratio=l1_ratio,
        max_iter=1000,
        C=C_value
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
        "l1_ratio": l1_ratio,
        "Accuracy": acc,
        "ROC_AUC": roc_auc,
        "MSE": mse,
        "MAE": mae
    }])
    print("================ Logistice Model ================")
    print(f'Logistice accureacy {acc}')
    return model, results_df

def mlp_clasifications(X_train, X_test, y_train, y_test,
                        hidden_layer_sizes=(100,),   # single tuple
                        activation='relu',           # single string
                        solver='adam',               # single string
                        max_iter=5000,
                        tol=0.0001,
                        alpha=0.0001):
    try:
        batch_size = min(200, X_train.shape[0])
        mlp = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            max_iter=max_iter,
            tol=tol,
            random_state=42
        )
        mlp.fit(X_train, y_train)

        # Predictions
        y_pred_prob = mlp.predict_proba(X_test)[:, 1]
        y_pred_class = mlp.predict(X_test)

        # Metrics
        results = {
            "Hidden_Layers": hidden_layer_sizes,
            "Activation": activation,
            "Solver": solver,
            "Accuracy": accuracy_score(y_test, y_pred_class),
            "ROC_AUC": roc_auc_score(y_test, y_pred_prob),
            "MSE": mean_squared_error(y_test, y_pred_prob),
            "MAE": mean_absolute_error(y_test, y_pred_prob)
        }
    except Exception as e:
        results = {
            "Hidden_Layers": hidden_layer_sizes,
            "Activation": activation,
            "Solver": solver,
            "Accuracy": f"Error: {str(e)}",
            "ROC_AUC": None,
            "MSE": None,
            "MAE": None
        }
    print("================ mlp model ================")
    print("Accuracy:", np.round(results["Accuracy"],6))
    return mlp, results

def get_mlflow_uri_path() -> str:
    """
    Construct MLflow tracking URI path dynamically.
    Returns: file:/<current_working_directory>/mlruns
    """
    cwd = os.getcwd()   # current working directory
    uri_path = f"file:{os.path.join(cwd, 'mlruns')}"
    return uri_path

def run_xgboost_classifier(X_train, X_test, y_train, y_test,
                           max_depth=3,
                           learning_rate=0.1,
                           n_estimators=100,
                           subsample=1.0,
                           colsample_bytree=1.0,
                           random_state=42):
    # Initialize model
    model = xgb.XGBClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=random_state
    )

    # Fit model
    model.fit(X_train, y_train)

    # Predictions
    y_pred_class = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    xgboot_acc = accuracy_score(y_test, y_pred_class)
    # Metrics
    results = {
        "Accuracy": xgboot_acc,
        "Precision": precision_score(y_test, y_pred_class, zero_division=0),
        "Recall": recall_score(y_test, y_pred_class, zero_division=0),
        "F1": f1_score(y_test, y_pred_class, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, y_pred_prob)
    }
    print("================ Xgb boot ================")
    print(f'Xgboot Accureacy:{np.round(xgboot_acc,5)}')
    return model,results

# Create parser
parser = argparse.ArgumentParser(description="Classification model runner")

# Add experiment name argument
parser.add_argument(
    "--experiment_name",
    type=str,
    default="classification",
    help="Name of the MLflow experiment (default: classification)"
)
# 1. Select model type
parser.add_argument("--model", type=str, choices=["glm", "logistic", "mlp", "xgboost"],
                    required=True, help="Choose classification model")

# 2. GLM arguments
parser.add_argument("--threshold", type=float, default=0.5,
                    help="Threshold for GLM classification (default: 0.5)")

# 3. Logistic Regression arguments
parser.add_argument("--C_value", type=float, default=1.0,
                    help="Inverse of regularization strength for Logistic Regression (default: 1.0)")
parser.add_argument("--l1_ratio", type=float, default=0.5,
                    help="ElasticNet mixing parameter for Logistic Regression (default: 0.5)")

# 4. MLP arguments
parser.add_argument("--hidden_layer", type=int, default=100,
                    help="Number of neurons in hidden layer (default: 100)")
parser.add_argument("--activation", type=str, choices=["identity", "logistic", "tanh", "relu"],
                    default="relu", help="Activation function for MLP (default: relu)")
parser.add_argument("--solver", type=str, choices=["lbfgs", "sgd", "adam"],
                    default="adam", help="Solver for MLP (default: adam)")
parser.add_argument("--alpha", type=float, default=0.0001,
                    help="L2 regularization term for MLP (default: 0.0001)")

# 5. XGBoost arguments
parser.add_argument("--learning_rate", type=float, default=0.1,
                    help="Learning rate for XGBoost (default: 0.1)")

args = parser.parse_args()   # no CLI args, just defaults

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Set tracking directory explicitly
    mlflow.set_tracking_uri(get_mlflow_uri_path())
    print("Tracking URI:", mlflow.get_tracking_uri())

    # Load data
    X_train, X_test, y_train, y_test = get_cleaned_data()

    print("The set tracking uri is ", mlflow.get_tracking_uri())

    # ✅ Use set_experiment to avoid duplicate errors
    exp = mlflow.set_experiment(args.experiment_name)
    exp_id = exp.experiment_id
    get_exp = mlflow.get_experiment(exp_id)

    print("Name:", get_exp.name)
    print("Experiment_id:", get_exp.experiment_id)
    print("Artifact Location:", get_exp.artifact_location)
    print("Tags:", get_exp.tags)
    print("Lifecycle_stage:", get_exp.lifecycle_stage)
    print("Creation timestamp:", get_exp.creation_time)
    #=============================================
    mlflow.start_run(experiment_id=exp_id, run_name=f"{args.model}_run")
    tags = {
        "Work": "Embedded Platform",
        "release.candidate":"RELAY_01",
        "release.version": "1.0.10",
        "dataset": "Cholesterol",
        "experiment.stage": "hyperparameter_tuning",
        "owner": "abhishek",
        "framework": "scikit-learn and staemodels",
        "model.type": "mix mode"
    }
    mlflow.set_tags(tags)
    # Start MLflow run
    if args.model == "glm":
        results = glm_binomial(X_train, X_test, y_train, y_test, args.threshold)
        mlflow.log_param("threshold", results["Parameters"]["threshold"])
        for metric_name, metric_value in results["Metrics"].items():
            mlflow.log_metric(metric_name, metric_value)
        with open("glm_model.pkl", "wb") as f:
            pickle.dump(results["Model"], f)
        mlflow.log_artifact("glm_model.pkl", artifact_path="glm_model")

    elif args.model == "logistic":
        model, results_df = logistic_model(
            X_train, X_test, y_train, y_test,
            C_value=args.C_value,
            l1_ratio=args.l1_ratio
        )
        mlflow.log_param("C", results_df.loc[0, "C"])
        mlflow.log_param("l1_ratio", results_df.loc[0, "l1_ratio"])
        for metric in ["Accuracy", "ROC_AUC", "MSE", "MAE"]:
            mlflow.log_metric(metric, results_df.loc[0, metric])
        mlflow.sklearn.log_model(model, name="logistic_model", serialization_format="skops")

    elif args.model == "mlp":
        mlp, results = mlp_clasifications(
            X_train, X_test, y_train, y_test,
            hidden_layer_sizes=args.hidden_layer,
            activation=args.activation,
            solver=args.solver,
            max_iter=10000,
            tol=0.0001,
            alpha=args.alpha
        )
        mlflow.log_param("Hidden_Layers", results["Hidden_Layers"])
        mlflow.log_param("Activation", results["Activation"])
        mlflow.log_param("Solver", results["Solver"])
        if isinstance(results["Accuracy"], (int, float)):
            for metric in ["Accuracy", "ROC_AUC", "MSE", "MAE"]:
                mlflow.log_metric(metric, results[metric])
        mlflow.sklearn.log_model(mlp, name="mlp_model", serialization_format="skops")

    elif args.model == "xgboost":
        model, results = run_xgboost_classifier(
            X_train, X_test, y_train, y_test,
            max_depth=3,
            learning_rate=args.learning_rate,
            n_estimators=100,
            subsample=1.0,
            colsample_bytree=1.0,
            random_state=42
        )
        for param in ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree"]:
            mlflow.log_param(param, model.get_params()[param])
        for metric in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]:
            mlflow.log_metric(metric, results[metric])
        mlflow.xgboost.log_model(model, name="xgb_model")

    # ✅ Print artifact URI and run info
    artifacts_uri = mlflow.get_artifact_uri()
    print("The artifact path is", artifacts_uri)

    # End run explicitly
    mlflow.end_run()

    # Get last active run info
    run = mlflow.last_active_run()
    if run:
        print("Active run id is {}".format(run.info.run_id))
        print("Active run name is {}".format(run.info.run_name))
