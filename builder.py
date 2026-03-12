import sys
import subprocess
from pathlib import Path

# Configuration: script to use and directories
python_script = "full_build.py"
dataset_dir = Path("dataset")
models_dir = dataset_dir / "onnx_models"
configs_dir = dataset_dir / "config_files"
results_dir = dataset_dir / "results_synth"

print("Starting automated FINN batch execution...")

# Get a list of all ONNX models in alphabetical order
# .glob() finds the files, sorted() ensures alphabetical order
models = sorted(models_dir.glob("*.onnx"))

# Check if any models were found
if not models:
    print(f"Error: No ONNX models found in {models_dir}")
    sys.exit(1)

# Iterate through every model
for model_path in models:
    model_name = model_path.name       # e.g., "my_model.onnx"
    model_base_name = model_path.stem  # e.g., "my_model"
    model_config_dir = configs_dir / model_base_name
    
    # Check if the config directory for this model exists
    if not model_config_dir.is_dir():
        print(f"Warning: No config directory found for {model_name}. Skipping...")
        continue
        
    # For each model, get the list of config files, in alphabetical order
    configs = sorted(model_config_dir.glob("*.json"))
    
    if not configs:
        print(f"Warning: Config directory {model_config_dir} is empty. Skipping...")
        continue
        
    # Iterate through every config file for this model
    for config_path in configs:
        config_name = config_path.name
        config_base_name = config_path.stem
        
        # Calculate the exact output directory the Python script will use
        target_output_dir = results_dir / model_base_name / config_base_name
        
        # Check if the output directory already exists to avoid overwriting
        if target_output_dir.is_dir():
            print("\n=========================================================")
            print(f" SKIPPING: Model = {model_name} | Config = {config_name}")
            print(" Reason: Output folder already exists!")
            print("=========================================================")
            continue
            
        # If the folder doesn't exist, proceed with the build
        print("\n=========================================================")
        print(f" RUNNING: Model = {model_name} | Config = {config_name}")
        print("=========================================================")
        
        # Prepare the command list
        command = [
            "python3", python_script, 
            "onnx_models", 
            "-v", 
            "-m", model_name, 
            "-fc", config_name
        ]
        
        # Check for errors during the build
        try:
            # subprocess.run executes the command. By not capturing stdout/stderr, 
            # it prints directly to your terminal exactly like TCL's >@ stdout
            subprocess.run(command, check=True)
            print(f"\n*** SUCCESS *** Completed {model_name} with {config_name}.")
            
        except subprocess.CalledProcessError as e:
            print(f"\n*** ERROR *** Build failed for {model_name} with {config_name}!")
            print(f"Error details: Command returned non-zero exit status {e.returncode}.")
            
        except KeyboardInterrupt:
            # Allows you to Ctrl+C out of the batch loop safely
            print("\nBatch execution cancelled by user.")
            sys.exit(1)

print("\nAll model-config combinations have been processed!")