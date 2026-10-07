import os
import argparse
import subprocess
import pandas as pd

# Define paths and seeds
SEEDS = [42, 0, 7]
MODELS = ["baseline", "cbam"]
RESULTS_DIR = "results"
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"

def main():
    parser = argparse.ArgumentParser(description="Wrapper script to run classification across seeds.")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Force a clean re-run of all seeds (including seed 42, which already exists)."
    )
    args = parser.parse_args()

    # Track runs
    completed_runs = []
    skipped_runs = []

    print("=" * 60)
    print("3-SEED TRAINING WRAPPER")
    print("=" * 60)

    for model in MODELS:
        model_slug = "baseline_resnet18" if model == "baseline" else "cbam_resnet18"
        for seed in SEEDS:
            result_path = os.path.join(RESULTS_DIR, f"classification_{model}_seed{seed}.csv")
            
            # For seed 42, check if the file exists. If it is a fresh run (no --clean), skip it.
            # For other seeds, also check if they exist to support resuming.
            if os.path.exists(result_path) and not args.clean:
                print(f"Skipping: {model} seed={seed} (results exist at {result_path})")
                # Read result path to verify metadata
                try:
                    df = pd.read_csv(result_path)
                    ckpt = df.iloc[0]["checkpoint_path"]
                    log = df.iloc[0]["log_path"]
                except Exception:
                    ckpt = os.path.join(CHECKPOINT_DIR, f"{model_slug}_seed{seed}_best.pth" if seed != 42 else f"{model_slug}_best.pth")
                    log = os.path.join(LOG_DIR, f"{model}_seed{seed}.csv" if seed != 42 else f"{model}_training.csv")
                
                skipped_runs.append({
                    "model": model,
                    "seed": seed,
                    "result_path": result_path,
                    "checkpoint_path": ckpt,
                    "log_path": log
                })
                continue
            
            # Otherwise run the training
            print(f"Starting: {model} seed={seed}")
            cmd = ["python", "train_classifier.py", "--model", model, "--seed", str(seed)]
            
            try:
                subprocess.run(cmd, check=True)
                print(f"Finished: {model} seed={seed}")
                
                # Verify output file
                if os.path.exists(result_path):
                    df = pd.read_csv(result_path)
                    ckpt = df.iloc[0]["checkpoint_path"]
                    log = df.iloc[0]["log_path"]
                else:
                    ckpt = os.path.join(CHECKPOINT_DIR, f"{model_slug}_seed{seed}_best.pth")
                    log = os.path.join(LOG_DIR, f"{model}_seed{seed}.csv")
                
                completed_runs.append({
                    "model": model,
                    "seed": seed,
                    "result_path": result_path,
                    "checkpoint_path": ckpt,
                    "log_path": log
                })
            except subprocess.CalledProcessError as e:
                print(f"ERROR: training failed for {model} seed={seed} with exit code {e.returncode}")
                return

    # Print final summary
    print("\n" + "=" * 60)
    print("RUN SUMMARY")
    print("=" * 60)
    print(f"Total runs completed: {len(completed_runs)}")
    print(f"Total runs skipped (already exist): {len(skipped_runs)}")
    print("-" * 60)
    
    if skipped_runs:
        print("Skipped Runs:")
        for r in skipped_runs:
            print(f" - {r['model'].upper()} (seed {r['seed']}):")
            print(f"   * Results:    {r['result_path']}")
            print(f"   * Checkpoint: {r['checkpoint_path']}")
            print(f"   * Log:        {r['log_path']}")
            
    if completed_runs:
        print("\nCompleted Runs:")
        for r in completed_runs:
            print(f" - {r['model'].upper()} (seed {r['seed']}):")
            print(f"   * Results:    {r['result_path']}")
            print(f"   * Checkpoint: {r['checkpoint_path']}")
            print(f"   * Log:        {r['log_path']}")
    print("=" * 60)

if __name__ == "__main__":
    main()
