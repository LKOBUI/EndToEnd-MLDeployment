# MLflow Experiment & Model Signature Workflow
==============================================

This project demonstrates how to train logistic regression models, clean and scale datasets, and log experiments into the MLflow Tracking Server with full metadata, parameters, metrics, artifacts, and model signatures.


## Model Signature, Signature Enforcement, and Logging Signatures:
===============================================================

## Model Signature:
----------------
A model signature describes the expected input and output schema of a model.

-It defines:
	-Input types (e.g., DataType.float, DataType.string, TensorSpec)
	-Output types (e.g., scalar, array, tensor)
	-Stored inside the model’s MLmodel file when you log the model.

## Signature Enforcement:
---------------------

When enabled, MLflow validates inputs at runtime against the stored signature.
If the input data doesn’t match (wrong type, missing column, etc.), MLflow raises 
an error. This prevents silent failures where a model might produce nonsense because
of mismatched input.
Example: If your model expects age:int and chol:float, but you pass a 
string "twenty", MLflow will block execution.

You can see your signature @this path

MLflow Tracking Server (http://127.0.0.1:5000)

└── Experiments

    └── Experiment ID: 14
	
        └── Run ID: a77040ba7bd14d89ad61b93e80197be1
		
            └── Artifacts
			
                ├── MLmodel
				
				      └── Signature
					  
                ├── model.pkl / model.skops
				
                ├── conda.yaml / requirements.txt
				
                ├── dataset_2190_cholesterol.csv
				
                └── other logged files

MLflow built in model flavors are standardized formats that let you save and load models across different 
machine learning libraries. They ensure interoperability, so a model trained in one framework can be deployed 
or served using MLflow’s common tools.

A flavor is a way MLflow packages a model so it can be understood by downstream tools. Enables models from 
different libraries (e.g., scikit‑learn, PyTorch, TensorFlow) to be logged, loaded, and served consistently.
Each MLflow model directory contains an MLmodel YAML file listing supported flavors.

## MLflow storage format:
----------------------

MLflow models are stored in a standardized directory format that makes them portable
across frameworks and deployment environments. This storage format is what allows 
MLflow to support multiple “flavors” and consistent loading/serving.

## MLflow Model Storage Format:
----------------------------

When you log a model with MLflow, it creates a directory with the following structure:

- model-directory/
	- │
	- ├── MLmodel                # Core metadata file (YAML)
	- ├── model.pkl              # Serialized model file (framework-specific)
	- ├── conda.yaml             # Conda environment specification (optional)
	- ├── requirements.txt       # Python dependencies (optional)
	- ├── python_env.yaml        # Alternative environment spec (optional)
	- └── artifacts/             # Extra files (plots, vocab, configs, etc.)

## Serialized model file:
----------------------

- Framework‑specific file (e.g., 
- model.pkl for scikit‑learn, 
- model.pt for PyTorch, 
- model.h5 for Keras

## Environment files:
------------------
- (conda.yaml, requirements.txt, python_env.yaml)
- Capture dependencies needed to run the model.
- Ensure reproducibility across machines.

## Artifacts directory:
-------------------
- tores additional files like preprocessing scripts, vocabularies, or plots.
- seful when models depend on external resources.

## API in details

#### Data Preparation API:
==========================

- `drop_non_numeric(df_frame)` → Converts all columns to numeric, drops rows with invalid values. 
- `categorize_chol(val)` → Categorizes cholesterol values into Normal, Medium, or High.
- `get_cleaned_data()` → Loads dataset, cleans it, applies StandardScaler to numeric features, and splits into train/test sets.

#### Model Training APIs:
========================

Each logistic regression variant trains with different hyperparameters:

- `logistic_model_1` → ElasticNet penalty, saga solver, C=10, l1_ratio=0.1.
- `logistic_model_2` → ElasticNet penalty, saga solver, C=1, l1_ratio=0.5.
- `logistic_model_3` → ElasticNet penalty, saga solver, C=1, l1_ratio=0.9.
- All return:
	- Trained model
	- Results DataFrame with parameters + metrics (Accuracy, ROC_AUC, MSE, MAE).
	
## Experiment Runner API (run_experiment):
=========================================
- Tracking URI: `mlflow.set_tracking_uri(uri)`. Defines where MLflow stores metadata/artifacts (local folder, SQLite DB, or remote server).
- Experiment Management: exp = `mlflow.set_experiment(experiment_name)` Creates or reuses experiments safely. Prints metadata (ID, name, artifact location, lifecycle stage).
- Run Lifecycle: with mlflow.start_run(experiment_id=exp_id, run_name=run_name): Starts a run under the experiment. Runs are atomic units in MLflow.
- Logging Parameters → `mlflow.log_param()` 
- Metrics → `mlflow.log_metric()`
- Model → `mlflow.sklearn.log_model()`
- Artifacts → `mlflow.log_artifacts()`
- Model Signature
	- input_schema = Schema([ColSpec(col["type"], col['name']) for col in input_data])
	- output_schema = Schema([ColSpec(col['type']) for col in output_data])
	- signature = ModelSignature(inputs=input_schema, outputs=output_schema)
	- mlflow.sklearn.log_model(model, name=model_name, signature=signature, input_example=input_example)
	- Inputs Schema → defines feature names and types.
	- Outputs Schema → defines prediction type.
- Signature → enforces schema consistency when serving models.
- Input Example → provides sample input for validation.
- Run Metadata  
- Prints artifact URI, run ID, and run name for traceability.























