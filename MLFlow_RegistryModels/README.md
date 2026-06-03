## MLflow Model Registry
========================

MLflow Model Registry is the part of `MLflow` that helps you manage the lifecycle of
your machine learning models after they’ve been trained and logged. Think of it 
as a central hub where models are `versioned, staged, and promoted for production` 
use.

The `MLflow Model Registry` is essentially GitHub for models: it versions them, tracks 
their lifecycle, and lets you promote or archive them as they move from experimentation 
to production.

## MLflow Model Registry:
----------------------
- Stores models: Every time you log a model, you can register it in the 
                 registry under a chosen name.
- Versions models: Each new registration creates a new version 
                  (Version 1, Version 2, etc.).
- Tracks metadata: Keeps notes, tags, and descriptions about models.
- Controls lifecycle: Lets you move models through different 
                      stages (e.g., from development to production).

## Stages in the Registry:
========================

- --------------------------------------------------------------------------------------------
- |Stage          |Purpose                                                                   |
- |---------------|--------------------------------------------------------------------------|
- |None           |Default stage when a model is first registered. No lifecycle assigned.    |
- --------------------------------------------------------------------------------------------
- |Staging        |For testing and validation. Models here are candidates for promotion.     |
- --------------------------------------------------------------------------------------------
- |Production     |The "live" model serving real traffic or being used in production systems.|
- --------------------------------------------------------------------------------------------
- |Archived       |Old or deprecated models. They're kept for record but not actively used.  |
- |------------------------------------------------------------------------------------------|

`MLflow Model Registry` is useful when Team collaboration, Multiple data scientists can register 
models under the same name and track versions. `Deployment pipelines`, You can promote a model 
from `Staging → Production` once it passes validation. Governance, Keeps a clear history of 
which model was used in production at any point in time.

Not Use It when If your have already uses another model registry (like AWS SageMaker or Azure ML), 
you may skip MLflow’s built‑in registry.

## Additional API:
==================

`last_active_run():` 

Returns the most recent run object that was active in the current Python process.
After finished a run (e.g., inside or just after a with `mlflow.start_run()`: block), 
you can call this to quickly get the `run ID`, name, and metadata of the last run you 
executed.

If no run has been started yet, it returns None. If multiple runs were started, it 
always points to the most recent one that was active. Commonly used for printing or 
debugging.

Note*:  `last_active_run()` always last `run’s unique ID` (run.info.run_id) even you sequence
`active_run() -> end_run() -> Call last_active_run()` -> will return last run’s unique ID 
(run.info.run_id)

## wrapper around MLflow’s Tracking and Model Registry APIs:
===========================================================

##Why It’s Useful in Model Registry:
====================================
- `Experiment Tracking` → Logs parameters, metrics, artifacts, and models into MLflow’s backend. This ensures reproducibility and traceability.
- `Model Registration` → After each run, the trained model is registered under a consistent name (BinaryClassifications). This creates versioned entries in the MLflow Model Registry.
- `Dynamic Reloading` → Instead of hard‑coding a version, the API fetches the latest registered version and reloads it. This makes workflows resilient and avoids errors when versions change.
- `Evaluation` → Once reloaded, the model is tested again on held‑out data, confirming that the registered artifact behaves as expected.

## Sub APIs Explained:
======================

- `mlflow.set_tracking_uri(uri)`: Defines where MLflow stores metadata/artifacts (local folder, DB, or remote server).
- `mlflow.set_experiment(experiment_name)`: Creates or reuses an experiment. Each experiment groups multiple runs.
- `mlflow.start_run()`: Starts a run under the experiment. Runs are the atomic unit of logging.
- `mlflow.log_param() / mlflow.log_metric()`: Records hyperparameters and evaluation metrics for the run.
- `mlflow.sklearn.log_model()`: Saves the trained model as an artifact, with optional serialization format (skops).
- `mlflow.log_artifacts()`: Stores additional files (datasets, plots, configs) alongside the run.
- `mlflow.register_model()`: Registers the model into the Model Registry, creating a new version under a given name.
- `MlflowClient().get_latest_versions()`: Queries the registry for the most recent version of a model. This avoids hard‑coding version numbers.
- `mlflow.pyfunc.load_model()`: Reloads the registered model (latest version or by stage), making it available for inference.
- `eval_metrics()`: Custom helper to compute RMSE, MAE, and R² on predictions from the reloaded model.

## Path Tree Structure:
=======================

- mlruns/
- ├── 0/                          # Default experiment (ID = 0)
- │   └── <run_id>/               # Each run has a unique ID
- │       ├── artifacts/          # Logged artifacts (models, datasets, plots)
- │       │   ├── model/          # Saved model folder
- │       │   │   ├── MLmodel     # Metadata file (includes signature if logged)
- │       │   │   ├── model.pkl   # Serialized model
- │       │   │   ├── conda.yaml  # Environment dependencies
- │       │   │   └── requirements.txt
- │       │   └── other_artifacts/
- │       ├── meta.yaml           # Run metadata (start/end time, status)
- │       └── metrics/            # Logged metrics (Accuracy, ROC_AUC, etc.)
- ├── <experiment_id>/            # Custom experiments (e.g., "mlflow_registery_model")
- │   └── <run_id>/               # Runs under this experiment
- │       ├── artifacts/
- │       ├── meta.yaml
- │       └── metrics/
- └── .trash/                     # Deleted experiments/runs are moved here











