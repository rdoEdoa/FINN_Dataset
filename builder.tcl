#!/usr/bin/tclsh

# Configuration: script to use and directories
set python_script "full_build.py"
set dataset_dir "dataset"
set models_dir "$dataset_dir/onnx_models"
set configs_dir "$dataset_dir/config_files"
set results_dir "$dataset_dir/results_synth" 

puts "Starting automated FINN batch execution..."

# Get a list of all ONNX models in alphabetical order
set models [lsort -dictionary [glob -nocomplain -directory $models_dir *.onnx]]

# Check if any models were found
if {[llength $models] == 0} {
    puts "Error: No ONNX models found in $models_dir"
    exit 1
}

# Iterate through every model
foreach model_path $models {
    
    set model_name [file tail $model_path]
    set model_base_name [file rootname $model_name]
    set model_config_dir [file join $configs_dir $model_base_name]
    
    # Check if the config directory for this model exists
    if {![file isdirectory $model_config_dir]} {
        puts "Warning: No config directory found for $model_name. Skipping..."
        continue
    }
    
    # For each model, get the list of config files, in alphabetical order
    set configs [lsort -dictionary [glob -nocomplain -directory $model_config_dir *.json]]
    
    if {[llength $configs] == 0} {
        puts "Warning: Config directory $model_config_dir is empty. Skipping..."
        continue
    }
    
    # Iterate through every config file for this model
    foreach config_path $configs {
        set config_name [file tail $config_path]
        set config_base_name [file rootname $config_name]
        
        # Calculate the exact output directory the Python script will use
        set target_output_dir [file join $results_dir $model_base_name $config_base_name]
        
        # Check if the output directory already exist. If it does, skip this model-config combination to avoid overwriting results.
        if {[file isdirectory $target_output_dir]} {
            puts "\n========================================================="
            puts " SKIPPING: Model = $model_name | Config = $config_name"
            puts " Reason: Output folder already exists!"
            puts "========================================================="
            continue ; # Skip to the next config immediately
        }
        
        # If the folder doesn't exist, proceed with the build
        puts "\n========================================================="
        puts " RUNNING: Model = $model_name | Config = $config_name"
        puts "========================================================="
        
        # Check for errors during the build
        if {[catch {exec -ignorestderr python3 $python_script "onnx_models" -v -m $model_name -fc $config_name >@ stdout} result]} {
            puts "\n*** ERROR *** Build failed for $model_name with $config_name!"
            puts "Error details: $result"
        } else {
            puts "\n*** SUCCESS *** Completed $model_name with $config_name."
        }
    }
}

puts "\nAll model-config combinations have been processed!"