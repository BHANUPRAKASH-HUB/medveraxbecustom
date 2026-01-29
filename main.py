import subprocess
import os
import sys
import time

def run_script(script_path, script_dir):
    """
    Executes a Python script from terminal directly allowing interaction
    """
    script_name = os.path.basename(script_path)
    print("\n" + "="*50)
    print(f"--- RUNNING: {script_name}")
    print("="*50)
    sys.stdout.flush()

    start_time = time.time()
    try:
        # Run directly in the shell to preserve interactive terminal behavior and output streaming
        # This is necessary because some scripts (09_final_test.py) require input()
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=script_dir,
            env={**os.environ, "PYTHONUNBUFFERED": "1"}
        )

        duration = time.time() - start_time

        if result.returncode == 0:
            print(f"\n[OK] COMPLETED: {script_name} ({duration:.2f}s)")
            return True, duration
        else:
            print(f"\n[ERROR] in {script_name}")
            print(f"Status: Failed with return code {result.returncode}")
            return False, 0

    except Exception as e:
        print(f"\n[CRITICAL ERROR] Failed to start {script_name}: {str(e)}")
        return False, 0

def main():
    # Setup paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # The scripts are in 'notebooks/notebook python'
    SCRIPT_FILE_DIR = os.path.join(BASE_DIR, "notebooks", "notebook python")
    # INTERPRETATION: All notebook code assumes relative path starting from 'notebooks/' 
    # (e.g., '../data/raw/'). So we MUST run from the 'notebooks' folder.
    CWD_DIR = os.path.join(BASE_DIR, "notebooks")
    
    # Ensure directories exist
    if not os.path.exists(SCRIPT_FILE_DIR):
        print(f"[ERROR] Error: Script directory not found at {SCRIPT_FILE_DIR}")
        sys.exit(1)
    if not os.path.exists(CWD_DIR):
        os.makedirs(CWD_DIR)

    # Strictly numerical sequence as requested
    scripts = [
        "01_data_collection.py",
        "02_data_cleaning.py",
        "03_eda_feature_engineering.py",
        "04_models_all_train_test.py",
        "05_statistical_validation.py",
        "06_threshold_tuning.py",
        "07_hybrid_training_all.py",
        "08_hyperparameter_tuning.py",
        "09_final_test.py"
    ]

    print("\n" + "*"*50)
    print("MEDVERAX FULL ML PIPELINE RUNNER")
    print("*"*50)
    print(f"Base Directory: {BASE_DIR}")
    print(f"Scripts File Directory: {SCRIPT_FILE_DIR}")
    print(f"Execution CWD: {CWD_DIR}")
    print(f"Total Scripts: {len(scripts)}")

    results = []
    pipeline_start = time.time()

    for script in scripts:
        script_path = os.path.join(SCRIPT_FILE_DIR, script)
        
        if not os.path.exists(script_path):
            print(f"\n[SKIP] Skipping: {script} (File not found)")
            results.append((script, "NOT FOUND", 0))
            continue

        success, duration = run_script(script_path, CWD_DIR)
        results.append((script, "SUCCESS" if success else "FAILED", duration))
        
        if not success:
            print("\n[HALT] Pipeline halted due to error.")
            break

    pipeline_duration = time.time() - pipeline_start

    # Final Summary Report
    print("\n" + "="*50)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*50)
    print(f"{'Script Name':<35} | {'Status':<10} | {'Time':<10}")
    print("-" * 60)
    for name, status, duration in results:
        duration_str = f"{duration:.2f}s" if duration > 0 else "-"
        print(f"{name:<35} | {status:<10} | {duration_str:<10}")
    
    print("-" * 60)
    print(f"Total Pipeline Duration: {pipeline_duration:.2f}s")
    
    if all(r[1] == "SUCCESS" for r in results):
        print("\n[SUCCESS] ALL STEPS COMPLETED SUCCESSFULLY!")
    else:
        print("\n[FAIL] PIPELINE FINISHED WITH ERRORS.")
        sys.exit(1)

if __name__ == "__main__":
    main()
