MLflow workflow combines model training functions (like logistic_model_1/2/3) with a reusable run_experiment() API that connects to the MLflow Tracking Server, ensuring parameters, metrics, artifacts, and models are logged and versioned. This makes experiments reproducible and traceable in the MLflow UI at http://127.0.0.1:5000/#/experiments.
## Models
	- logistic_model_1 → Trains a logistic regression model with ElasticNet penalty and saga solver.
	- Hyperparameters: C=10, l1_ratio=0.1.
	- Outputs: Accuracy, ROC‑AUC, MSE, MAE.
	- Returns: trained model + DataFrame of results.
	
Other models (GLM, MLP, XGBoost) can be defined similarly, each returning a model and results DataFrame.

## Experiment Runner (run_experiment)
This function is the bridge to the MLflow Tracking Server:
	- `Tracking URI` → sets where MLflow stores metadata/artifacts (sqlite:///mlflow.db, ./mlruns, or remote storage).
	- `Experiment Management` → restores deleted experiments, creates new ones if missing, or reuses existing ones.
	- `Run Lifecycle` → starts a run, logs parameters, metrics, models, and artifacts.
	- `Metadata` → prints experiment details (ID, name, artifact location, lifecycle stage).
	- `Outputs` → trained model, logged metrics, and run identifiers (run_id, run_name).
	
#### MLflow Tracking Server API that automates the full lifecycle of an experiment

**Tracking URI Setup** → It begins by setting the MLflow tracking URI (`mlflow.set_tracking_uri(uri)`), which tells MLflow where to store metadata and artifacts. In a tracking server setup, this URI points to your backend store (e.g., SQLite, PostgreSQL) and artifact store (e.g., local folder, S3).

**Experiment Management** → Using the `MlflowClient`, it checks if the experiment already exists. If it’s marked as deleted, the function restores it (`client.restore_experiment`). If it doesn’t exist, it creates a new one. This ensures that experiments are reusable and consistent across multiple runs.

**Experiment Metadata** → It retrieves and prints experiment details from the tracking server (name, ID, artifact location, tags, lifecycle stage, creation time). This is exactly what the tracking server stores in its backend database.

**Run Lifecycle** → Inside `mlflow.start_run()`, the function starts a run under the chosen experiment. Runs are the atomic unit in MLflow tracking: they capture parameters, metrics, tags, and artifacts.

**Logging to Server →**

- Parameters (`mlflow.log_param`) → hyperparameters like C, l1_ratio, solver, penalty.
- Metrics (`mlflow.log_metric`) → accuracy, ROC‑AUC, MSE, MAE.
- Model (`mlflow.sklearn.log_model`) → serialized and stored in the artifact store.
- Artifacts (`mlflow.log_artifacts`) → additional files (e.g., data, plots).

**Artifact URI** → It prints the artifact path, which is managed by the tracking server and points to where files are stored (local folder or remote storage).

**Run Closure & Metadata** → After ending the run (`mlflow.end_run()`), it fetches the last active run (mlflow.last_active_run) and prints its run ID and run name. These identifiers are critical for querying results later in the MLflow UI or API.