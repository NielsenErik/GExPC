import argparse
import json
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate configuration files for Tipi crack segmentation experiments.")
    parser.add_argument("template", help="Path of the base template JSON file to use")
    args = parser.parse_args()

    print("Launching Tipi config generator...")

    # Load the base configuration
    with open(args.template, "r") as f:
        base_config = json.load(f)

    # Define datasets and algorithms to generate configs for
    datasets = [
        "data/crackseg9k_split/AEL",
        "data/crackseg9k_split/CCIC",
        "data/crackseg9k_split/Ceramic",
        "data/crackseg9k_split/CRACK500",
        "data/crackseg9k_split/cracktree200",
        "data/crackseg9k_split/DeepCrack",
        "data/crackseg9k_split/GAPS384",
        "data/crackseg9k_split/Rissbilder",
        "data/crackseg9k_split/Volker",
        "data/CrackSeg9k"
    ]

    algorithms = ["unet", "unetplus", "deepcrackz", "crackformer", "nnunet"]
    base_path = "src/config/jawa/"

    # Generate one config file per (dataset, algorithm) combination
    for dataset in datasets:
        dataset_name = os.path.basename(dataset)
        for algo in algorithms:
            config = base_config.copy()

            # Update fields dynamically
            config["algorithm"] = algo
            config["data_path"] = os.path.join(dataset, "train")
            config["test_path"] = os.path.join(dataset, "test")
            config["model_name"] = algo
            config["experiment"] = f"jawa_{dataset_name}"

            # Build output path
            output_dir = os.path.join(base_path, dataset_name)
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{algo}.json")

            # Save JSON file
            print(f"Writing config file to {output_file}")
            with open(output_file, "w") as f:
                json.dump(config, f, indent=4)

    print("Tipi config generator done.")
