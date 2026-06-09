import os
import mlflow
import argparse
import json

from setup import set_up_mlflow
from experiment import run_experiment

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path of the config file to use")
    parser.add_argument("--debug", action="store_true", help="Debug flag")
    parser.add_argument("seed", type=int, help="Random seed to use")
    args = parser.parse_args()
    config = json.load(open(args.config))
    if "crackseg9k_split" in config["data_path"]:
        dataset_name = config["data_path"].split("/")[2]
    else:
        dataset_name = config["data_path"].split("/")[1]
    config["seed"] = args.seed
    if "preprocessing" not in config:
        config["preprocessing"] = False
    if config["preprocessing"]==True:
        config["log_dir"] = os.path.join("logs", config["algorithm"], "preprocessing", dataset_name, str(config["seed"]))
    else:
        config["log_dir"] = os.path.join("logs", config["algorithm"], dataset_name, str(config["seed"]))
    os.makedirs(config["log_dir"], exist_ok=True)
    json.dump(config, open(os.path.join(config["log_dir"], "config.json"), "w"), indent=4)
    run_experiment(config)

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()