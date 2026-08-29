import yaml
import subprocess
import os
import sys

EXPERIMENTS = [
    # {
    #     "name": "Exp 1: Baseline Loop",
    #     "overrides": {
    #         "extraction": {
    #             "pad_mode": "loop",
    #             "strategy": "uniform",
    #             "pooling": "spatiotemporal"
    #         }
    #     },
    #     "depths": ["1", "2"]
    # },
    # {
    #     "name": "Exp 2: Static Padding",
    #     "overrides": {
    #         "extraction": {
    #             "pad_mode": "last_frame",
    #             "strategy": "uniform",
    #             "pooling": "spatiotemporal"
    #         }
    #     },
    #     "depths": ["1", "2"]
    # },
    # {
    #     "name": "Exp 3: Shuffle Ablation",
    #     "overrides": {
    #         "extraction": {
    #             "pad_mode": "last_frame",
    #             "strategy": "shuffle",
    #             "pooling": "spatiotemporal"
    #         }
    #     },
    #     "depths": ["1", "2"]
    # },
    {
        "name": "Exp 4: Spatial Only",
        "overrides": {
            "extraction": {
                "pad_mode": "last_frame",
                "strategy": "uniform",
                "pooling": "spatial"
            }
        },
        "depths": ["1", "2"]
    },
]


def update_dict(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def main():
    base_config_path = "config.yaml"
    temp_config_path = "config_temp.yaml"

    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    for i, exp in enumerate(EXPERIMENTS):
        print(f"\n{'='*60}")
        print(f"Starting {exp['name']} ({i+1}/{len(EXPERIMENTS)})")
        print(f"{'='*60}")

        # Create temporary config
        config = base_config.copy()
        update_dict(config, exp["overrides"])
        
        with open(temp_config_path, "w") as f:
            yaml.dump(config, f)

        # 1. Extract Features
        print(f"\n--- Running feature extraction for {exp['name']} ---")
        cmd_extract = [sys.executable, "extract_features.py", "--config", temp_config_path]
        subprocess.run(cmd_extract, check=True)

        # 2. Run Probes
        print(f"\n--- Running probes for {exp['name']} (depths {exp['depths']}) ---")
        cmd_probes = [sys.executable, "run_probes.py", "--config", temp_config_path, "--depth"] + exp["depths"]
        subprocess.run(cmd_probes, check=True)

    # Cleanup
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)
    
    print("\nAll experiments finished successfully!")


if __name__ == "__main__":
    main()
