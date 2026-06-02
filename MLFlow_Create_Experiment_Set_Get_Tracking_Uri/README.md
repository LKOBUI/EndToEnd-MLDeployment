# MLflow Experiment Setup & Tracking URI
Goal: Demonstrate reproducible experiment management and artifact logging
## 📌 How to Run Code
Run experiments with different models using command-line arguments:


python Create_Experiment_Set_Get_Tracking_Uri.py --experiment_name GLM --model glm --threshold 0.5
python Create_Experiment_Set_Get_Tracking_Uri.py --experiment_name LOGISTIC --model logistic --C_value 1.0 --l1_ratio 0.7
python Create_Experiment_Set_Get_Tracking_Uri.py --experiment_name MLP --model mlp --hidden_layer 10 --activation relu --solver lbfgs --alpha 0.0001
python Create_Experiment_Set_Get_Tracking_Uri.py --experiment_name XGBOOT --model xgboost --learning_rate 0.001

## Windows PowerShell Cleanup

powershell
Remove-Item -Recurse -Force "D:\ML\MLFlow\MLFlow-1\mlruns"

## Tracking URI
- Definition: The Tracking URI tells MLflow where to store experiments, runs, metrics, parameters, and artifacts.
- Default: ./mlruns if not explicitly set.

## Directory structure:

Code
mlruns/
  └── <experiment_id>/
      └── <run_id>/
          ├── metrics/
          ├── params/
          ├── artifacts/
          └── tags/

## Experiments
An experiment is a container for multiple runs. Each experiment has:
- Name → human-readable identifier
- ID → unique identifier
- Tags → metadata labels (e.g., version, priority)
- Artifact location → storage path for artifacts (local folder, cloud storage, or network file system)

#### Artifacts can be stored in:

- Local filesystem (./mlruns, myartifacts)
- Cloud storage (Amazon S3, Azure Blob, Google Cloud Storage)
- Network file systems (NFS, SFTP)

#### Important:  
create_experiment() can only be called once per unique name.
If rerun with the same name, you’ll get an “Experiment already exists” error.

Use instead:

`mlflow.set_experiment(args.experiment_name)` This reuses the experiment if it exists, or creates it if it doesn’t.

## Useful MLflow APIs
- `mlflow.end_run()` → Ends the current MLflow run (like closing a session).
- `mlflow.last_active_run()` → Retrieves the most recent run object.
- `run.info.run_id` → Unique run identifier (UUID-style string).
- `run.info.run_name` → Human-readable run name (auto-generated if not set).
- These APIs make it easy to trace runs in the MLflow UI or programmatically.

## Workflow Notes
- Always print the run ID and run name after training/logging metrics → makes it easy to trace results in MLflow UI.
- Artifacts (models, plots, logs) are stored in the artifact_location defined by the experiment.
- Cloud storage options: Amazon S3, Azure Blob, Google Cloud Storage
- Network storage options: NFS, SFTP