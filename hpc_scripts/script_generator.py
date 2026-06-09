import os
import glob

if __name__ == "__main__":
    print("Launching HPC script generator...")

    # Paths
    CONFIG_ROOT = "src/config/hpc"
    TEMPLATE_PATH = "hpc_scripts/template.sh"
    OUTPUT_DIR = "hpc_scripts/generated"

    # Load the PBS script template
    with open(TEMPLATE_PATH, "r") as f:
        template = f.read()

    # Find all generated config files recursively
    config_files = glob.glob(os.path.join(CONFIG_ROOT, "**", "*.json"), recursive=True)

    if not config_files:
        print("⚠️ No config files found in:", CONFIG_ROOT)
        exit(1)

    # Make output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for config_path in config_files:
        # Example: src/config/hpc/CRACK500/pcnet_ge.json
        rel_path = os.path.relpath(config_path, CONFIG_ROOT)
        dataset = os.path.dirname(rel_path)
        experiment = os.path.splitext(os.path.basename(config_path))[0]

        # Prepare output directories
        log_dir = os.path.join("logs", "hpc", dataset, experiment)
        os.makedirs(log_dir, exist_ok=True)

        # Create job script path
        job_script_path = os.path.join(OUTPUT_DIR, f"{dataset}_{experiment}.sh")
        os.makedirs(os.path.dirname(job_script_path), exist_ok=True)

        # Replace placeholders in the template
        script_content = (
            template
            .replace("{dataset}", dataset)
            .replace("{experiment}", experiment)
            .replace("{config_path}", config_path)
        )

        # Write the job script
        with open(job_script_path, "w") as f:
            f.write(script_content)

        print(f"🧩 Wrote script: {job_script_path}")

    print("✅ HPC script generator done.")
