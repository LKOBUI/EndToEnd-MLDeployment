# Reproducible ML Deployment Pipeline

### Learning Objectives

* **Achieve Complete Reproducibility**: Master the ability to reconstruct any machine learning model by precisely mapping the exact dataset version, code version, and environment configuration used during training.
* **Implement Data Versioning with LakeFS**: Learn to apply Git-like workflows (commits, branches, merges) to large-scale datasets, ensuring immutable data snapshots for every experiment.
* **Enforce Code Trackability with Git SHA**: Automate the process of capturing and tagging MLflow runs with the exact Git commit SHA to ensure code states are never lost.
* **Master MLflow Lifecycle Management**: Practice setting up tracking servers, logging models with explicit signatures, managing experiments, and registering production-ready models in the MLflow Model Registry.

### Repository Directory Tree

```text
.
├── .github/
│   └── workflows/
├── data/
├── notebooks/
│   ├── 1. MLFlow_Create_Experiment_Set_Get_Tracking_Uri.ipynb
│   ├── 2. MLFlowTrackingServer.ipynb
│   ├── 3. MLFlow_Client.ipynb
│   ├── 4. MLflow_Code_Versioning_GitSHA.ipynb
│   ├── 5. MLFlow-ModelSignature.ipynb
│   ├── 6. MLFlow_RegistryModels.ipynb
│   └── 7. End‑to‑End_Data_Versioning_LakeFS.ipynb
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

### Detailed Summary of Project Notebooks

#### 1. MLFlow_Create_Experiment_Set_Get_Tracking_Uri
* Focuses on the fundamental setup of the MLflow tracking environment.
* Demonstrates how to programmatically set and retrieve the MLflow Tracking URI to connect local training scripts to remote or local tracking servers.
* Details the creation and management of isolated MLflow experiments, preventing run pollution and organizing different model training phases.

#### 2. MLFlowTrackingServer
* Covers the configuration, launching, and administration of a centralized MLflow Tracking Server.
* Explains how to handle backend stores (such as relational databases for metadata) and artifact stores (such as S3-compatible storage for models and logs).
* Guides the user on how team members can concurrently log metrics, parameters, and artifacts to a unified dashboard.

#### 3. MLFlow_Client
* Explores advanced interaction with the MLflow ecosystem using the `MlflowClient` low-level API.
* Demonstrates how to search, filter, and retrieve historical runs, parameters, and metrics programmatically without using the standard fluent API.
* Teaches how to manage experiment states, delete runs, and query metadata directly from python scripts.

#### 4. MLflow_Code_Versioning_GitSHA
* Bridges the gap between code modifications and model outputs by leveraging Git version control.
* Implements automated scripts to extract the current Git commit SHA at the exact moment a training run starts.
* Logs the Git SHA as a core tag within the MLflow run, providing an audit trail back to the precise line of code that generated the model.

#### 5. MLFlow-ModelSignature
* Highlights the critical importance of data contract safety in production deployments.
* Shows how to automatically infer or explicitly define model signatures (input schemas and output formats) using MLflow.
* Logs the model signature alongside model artifacts, ensuring that downstream deployment pipelines or API endpoints can automatically validate incoming data shapes and types.

#### 6. MLFlow_RegistryModels
* Focuses on the transition from an experiment run to a deployable asset.
* Covers the MLflow Model Registry workflow, including model registration, semantic versioning, and state transitions (e.g., Staging to Production).
* Teaches how to manage model aliases and tags to seamlessly serve the correct model version to end-users.

#### 7. End‑to‑End_Data_Versioning_LakeFS
* The capstone notebook that unifies data tracking with model tracking.
* Integrates LakeFS API calls into the ML pipeline to manage data repositories like source code.
* Demonstrates creating a data branch for an experiment, committing the data snapshot, and merging it back once the experiment succeeds.

### MLflow Code and Data Versioning Using LakeFS

The core strength of this repository is its strict integration of data and code tracking within MLflow to guarantee end-to-end reproducibility.

#### Data Versioning with LakeFS
Rather than pointing to a mutable bucket path (e.g., `s3://my-bucket/data.csv`), LakeFS introduces a Git-like structure to object storage. 
* **Branching**: The pipeline creates a specific data branch (e.g., `experiment-branch`) to isolate data changes or snapshots.
* **Committing**: Once the training dataset is finalized, a LakeFS commit is generated. This produces a unique, immutable LakeFS Commit ID representing the exact state of the data.
* **Tagging in MLflow**: The LakeFS Commit ID and repository URI are injected directly into the active MLflow run using custom tags:
  `mlflow.set_tag("lakefs.commit_id", lakefs_commit_sha)`
  `mlflow.set_tag("lakefs.dataset_uri", "lakefs://repo/branch/dataset.csv")`

#### Code Versioning with Git SHA
To ensure the training script itself is versioned, the workflow uses Git automation.
* **SHA Extraction**: The Python script programmatically reads the active repository HEAD commit using Git libraries or subprocesses.
* **Tagging in MLflow**: The resulting Git SHA is attached as a metadata tag to the MLflow run:
  `mlflow.set_tag("git.commit_sha", current_git_sha)`

By combining the **LakeFS Commit ID** (Data Snapshot) and the **Git Commit SHA** (Code Snapshot) as immutable tags within **MLflow** (Experiment Tracking), any engineer can look at a registered model, pull the exact code revision, download the identical dataset version, and perfectly replicate the training results.
