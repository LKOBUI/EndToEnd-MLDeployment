import mlflow
from mlflow import MlflowClient
from mlflow.entities import ViewType

mlflow.set_tracking_uri("http://127.0.0.1:5000")

client = MlflowClient()

run = client.create_run (
    experiment_id=11,
    tags={
        "Version": "v1",
        "Priority": "P1"
    },
    run_name="run from client 2"
)

print(f"Run tags: {run.data.tags}")
print(f"Experiment id: {run.info.experiment_id}")
print(f"Run id: {run.info.run_id}")
print(f"Run name: {run.info.run_name}")
print(f"lifecycle_stage: {run.info.lifecycle_stage}")
print(f"status: {run.info.status}")

# Uses of set_tracking_uri():

# mlflow.set_tracking_uri("http://127.0.0.1:5000")

client = MlflowClient()
print('Access rin id {bfd36ebdd74e4973aeac6dfc8a5a969d}')
run = client.get_run("bfd36ebdd74e4973aeac6dfc8a5a969d")
print(f"Run tags: {run.data.tags}")
print(f"Experiment id: {run.info.experiment_id}")
print(f"Run id: {run.info.run_id}")
print(f"Run name: {run.info.run_name}")
print(f"lifecycle_stage: {run.info.lifecycle_stage}")
print(f"status: {run.info.status}")