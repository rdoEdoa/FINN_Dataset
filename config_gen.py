import os
import torch
import onnx
import json
import random
import glob
import numpy as np
from tqdm import tqdm
from onnx import numpy_helper

DATA_DIR = "./dataset/weights"
ONNX_DIR = "./dataset/onnx_models"
OUTPUT_JSON_DIR = "./dataset/config_files"

# Maximum number of configurations generated per model
N_CONF = 12

if not os.path.exists(OUTPUT_JSON_DIR):
    os.makedirs(OUTPUT_JSON_DIR)

def get_div(n):
    """Returns a list of power-of-2 divisors for a given number."""
    if n is None or n <= 0:
        return [1]
    n = int(n)
    return [i for i in range(1, n + 1) if n % i == 0 and (i & (i - 1)) == 0]

def find_weights(start_tensor_name, initializer_map, producer_map, max_depth=6):
    """Searches the weights associated to a layer"""
    # The max depth prevents the search from going to previous layers
    queue = [(start_tensor_name, 0)]
    visited = set()
    while queue:
        curr_name, depth = queue.pop(0)
        if depth > max_depth: continue
        if curr_name in visited: continue
        visited.add(curr_name)
        if curr_name in initializer_map:
            w = numpy_helper.to_array(initializer_map[curr_name])
            if len(w.shape) == 4 or (len(w.shape) == 2 and w.shape[0]>1): return w
        if curr_name in producer_map:
            for inp in producer_map[curr_name].input: queue.append((inp, depth + 1))
    return None

def get_layer_properties(node, initializer_map, producer_map):
    """Returns the properties of a layer (MH, MW) based on its weights. Defaults to (1, 1) if not found."""
    mh, mw = 1, 1
    if len(node.input) > 1:
        w = find_weights(node.input[1], initializer_map, producer_map)
        if w is not None:
            if len(w.shape) == 4: # Conv
                mh = w.shape[0]
                mw = w.shape[1] * w.shape[2] * w.shape[3] # In * K * K 
            elif len(w.shape) == 2: # MatMul
                mh = w.shape[0]
                mw = w.shape[1]
    return max(1, mh), max(1, mw)

def main():
    os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.pth')]
    print(f"Configuration files generation on {len(files)} models...")
    
    for f in tqdm(files):
        pt_path    = os.path.join(DATA_DIR, f)
        onnx_path  = os.path.join(ONNX_DIR, f.replace('.pth', '.onnx'))
        model_name = f.replace('.pth', '')

        if not os.path.exists(onnx_path):
            continue

        model_output_dir = os.path.join(OUTPUT_JSON_DIR, model_name)
        os.makedirs(model_output_dir, exist_ok=True)

        # Count how many config files already exist for this model
        existing_configs = sorted(glob.glob(os.path.join(model_output_dir, "config_*.json")))
        num_existing     = len(existing_configs)

        if num_existing >= N_CONF:
            print(f"  {model_name}: already has {num_existing}/{N_CONF} configs, skipping.")
            continue

        num_to_generate = N_CONF - num_existing
        # The next config index starts from where we left off
        next_config_idx = num_existing

        print(f"  {model_name}: {num_existing} existing configs, generating {num_to_generate} more.")

        # Open the .pth file and the corresponding ONNX model
        data  = torch.load(pt_path)
        model = onnx.load(onnx_path)
        graph = model.graph
        initializer_map = {init.name: init for init in graph.initializer}
        producer_map    = {node.output[0]: node for node in graph.node
                           for out in node.output}

        targets_perf = []
        targets_area = []
        mva_counter  = 0
        swg_counter  = 0
        layer_options = {}

        for node in graph.node:
            mh, mw = get_layer_properties(node, initializer_map, producer_map)

            if node.op_type in ['Conv', 'MatMul', 'Gemm']:
                targets_perf.append([float(mh), float(mw), float(mw)])
                targets_area.append([1.0, 1.0, float(mw)])

                valid_configs_for_layer = []
                for pe in get_div(mh):
                    for simd in get_div(mw):
                        config_line = {
                            f"MVAU_hls_{mva_counter}": {"PE": pe, "SIMD": simd, "ram_style": "auto"},
                            f"MVAU_rtl_{mva_counter}": {"PE": pe, "SIMD": simd, "ram_style": "auto"}
                        }
                        if node.op_type == 'Conv':
                            config_line[f"ConvolutionInputGenerator_hls_{swg_counter}"] = {"SIMD": simd, "ram_style": "auto"}
                            config_line[f"ConvolutionInputGenerator_rtl_{swg_counter}"] = {"SIMD": simd, "ram_style": "auto"}

                        valid_configs_for_layer.append(config_line)

                if valid_configs_for_layer:
                    layer_options[f"layer_{mva_counter}"] = valid_configs_for_layer

                mva_counter += 1
                if node.op_type == 'Conv':
                    swg_counter += 1
            else:
                targets_perf.append([1.0, 1.0, 1.0])
                targets_area.append([1.0, 1.0, 1.0])

        data.y_max_perf = torch.tensor(targets_perf, dtype=torch.float)
        data.y_min_area = torch.tensor(targets_area, dtype=torch.float)
        if hasattr(data, 'y'):
            del data.y
        torch.save(data, pt_path)

        if not layer_options:
            continue

        layer_keys = list(layer_options.keys())

        # Calculate max possible unique combinations
        max_possible_combos = 1
        for key in layer_keys:
            max_possible_combos *= len(layer_options[key])

        # Load already-generated combos to avoid duplicates
        existing_combo_strs = set()
        for cfg_path in existing_configs:
            try:
                with open(cfg_path) as fin:
                    existing_combo_strs.add(json.dumps(json.load(cfg_path), sort_keys=True))
            except Exception:
                pass

        # Cap at what is actually possible minus what already exists
        num_to_generate = min(num_to_generate, max_possible_combos - num_existing)

        if num_to_generate <= 0:
            print(f"  {model_name}: no new unique combinations possible, skipping.")
            continue

        total_generated  = 0
        unique_combos    = existing_combo_strs.copy()
        max_attempts     = num_to_generate * 100  # avoid infinite loop
        attempts         = 0

        while total_generated < num_to_generate and attempts < max_attempts:
            attempts += 1
            chosen_combo = tuple(random.choice(layer_options[key]) for key in layer_keys)
            combo_str    = json.dumps(chosen_combo, sort_keys=True)

            if combo_str not in unique_combos:
                unique_combos.add(combo_str)

                final_json_dict = {"Defaults": {}}
                for layer_dict in chosen_combo:
                    final_json_dict.update(layer_dict)

                # Use next_config_idx to continue numbering from where we left off
                file_path = os.path.join(
                    model_output_dir,
                    f"config_{next_config_idx + total_generated:05d}.json"
                )
                with open(file_path, "w") as fout:
                    json.dump(final_json_dict, fout, indent=4)

                total_generated += 1

        if total_generated < num_to_generate:
            print(f"  WARNING: {model_name}: only generated {total_generated}/{num_to_generate} "
                  f"new configs after {max_attempts} attempts.")

if __name__ == "__main__":
    main()