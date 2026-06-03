## MlflowClient :
-------------

MLflow’s MlflowClient is a low‑level Python API that talks directly to the MLflow 
Tracking Server and Model Registry via REST calls. It gives you fine‑grained control 
over experiments, runs, and registered models, unlike the higher‑level mlflow module 
which manages only the “active run.”

Provides CRUD (Create, Read, Update, Delete) operations for, Experiments (create, list, 
delete, restore, set tags).
Runs (start, log params/metrics/artifacts, search, delete).
Model Registry (create registered models, add versions, transition stages, set aliases).

## set_tracking_uri():
-------------------
This API tells MLflow where to store and retrieve experiment data — parameters, 
metrics, artifacts, and models. Think of it as setting the “database connection” 
for MLflow’s tracking system.Without setting the tracking URI, MLflow defaults 
to a local folder (./mlruns). By explicitly calling set_tracking_uri(),
Centralize experiment tracking across multiple machines. Enable collaboration 
among team members. Integrate with CI/CD pipelines and MLOps workflows.

## joblib.dump():
-------------
joblib.dump() is how you persist trained ML models (or other Python objects) to disk and
a serialization API from the joblib library in Python. It’s used to save Python 
objects to disk so they can be reloaded later without retraining or re-computation.

## client.set_experiment_tag():
===============================
`client.set_experiment_tag()` is a low level MLflow API that lets you attach metadata 
directly to an experiment rather than to a run. While `mlflow.set_tags()` applies tags 
to a specific run, `set_experiment_tag()` applies tags at the experiment level, 
meaning all runs inside that experiment can be grouped or filtered by those tags. 
This is useful for organizing experiments with labels 
like `"team": "NLP", "priority": "P1", or "dataset": "Cholesterol",` 
so you can later query or filter experiments in the `MLflow UI` or 
via the `client`. In short, it’s a way to add descriptive, searchable 
metadata to experiments, improving governance and discoverability across multiple runs.

## client.create_run():
====================
`client.create_run()` is a low‑level MLflow Tracking API that programmatically starts 
a new run inside a given experiment. Unlike the high level `mlflow.start_run()` context 
manager, this method gives you direct control over run creation and metadata. You 
pass an `experiment_id` to specify which experiment the run belongs to, and you can 
also attach tags `like "Version": "v1" or "Priority": "P1")` and a human‑friendly 
`run_name`. The method returns a Run object containing two parts:
- `run.info` → structural metadata (run ID, experiment ID, run name, lifecycle stage, status).
- `run.data` → logged data such as tags, parameters, and metrics.

## set_experiment_tag() & get_experiment():
===========================================
This method lets you attach metadata tags directly to an experiment. 
Unlike run‑level tags (which describe a single run), experiment tags 
describe the entire experiment and apply across all runs inside it.

`client.get_experiment()` This retrieves the full metadata of an experiment by its ID. 
It returns details such as:

- experiment_id → numeric ID
- name → experiment name
- artifact_location → where artifacts are stored
- lifecycle_stage → ACTIVE or DELETED
- tags → any tags set via set_experiment_tag()