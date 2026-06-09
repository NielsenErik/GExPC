import mlflow
import os

def set_up_mlflow(**kwargs):
    if kwargs["mlflow"] == "aida":
        experiment_name = kwargs["experiment"]
        mlflow.set_tracking_uri("https://mlflow.dev.aida-labs.it")
        mlflow.set_experiment(experiment_name=experiment_name)
        print(f"MLflow experiment set to {experiment_name} at AIDA MLflow server")
    #return experiment

