import mlflow
from mlflow import MlflowClient

# Set the tracking server URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Initialize client
client = MlflowClient()

exp_id = client.create_experiment(
    name="my_experiment_30_v2",   # new unique name
    artifact_location="http://127.0.0.1:5000/artifacts",
    tags={"purpose": "Exp_Id30_v2", "owner": "abhishek"}
)
print("New experiment created with ID:", exp_id)

# Tag the experiment to indicate your desired logical ID
client.set_experiment_tag(exp_id, "desired_id", "30")

print("Experiment created with ID:", exp_id)

# Verify experiment details
experiment = client.get_experiment(exp_id)
print("Name:", experiment.name)
print("Experiment_id:", experiment.experiment_id)
print("Artifact Location:", experiment.artifact_location)
print("Lifecycle_stage:", experiment.lifecycle_stage)
