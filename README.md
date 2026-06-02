#Git SHA + LakeFS Commit Tracking for MLflow Experiments
# 📘 Reproducible ML Deployment Pipeline

This repository demonstrates an **end‑to‑end ML deployment workflow** that integrates:

- **LakeFS** → Git‑style data versioning (commits, branches, merges for datasets)  
- **Git SHA** → Code versioning for reproducibility  
- **MLflow** → Experiment tracking, model registry, and deployment  

## 🔑 Why This Repo?
The goal is to showcase a **production‑ready MLOps pipeline** where every experiment is fully traceable:
- **Data snapshot** → LakeFS commit ID  
- **Code snapshot** → Git commit SHA  
- **Environment snapshot** → Conda `environment.yml` or Docker image digest  
- **Experiment results** → MLflow runs with parameters, metrics, and artifacts  

## 📂 Structure
EndToEnd-MLDeployment/
│── src/                  # Python source code (LakeFS + MLflow integration)
│── notebooks/            # Jupyter notebooks for experiments
│── configs/              # Environment.yml and Dockerfile
│── artifacts/            # Saved models, logs, metrics
│── README.md             # Project documentation


## 🚀 Workflow
1. **Version datasets** with LakeFS commits.  
2. **Track experiments** in MLflow with Git SHA + LakeFS commit tags.  
3. **Log environment** using Conda spec or Docker digest.  
4. **Register models** in MLflow Model Registry for deployment.  

## ✅ Outcome
This pipeline ensures **end to end reproducibility**: you can always answer *which dataset, which code, and which environment produced this model*. Perfect for **interviews, portfolio projects, and production MLOps**.

