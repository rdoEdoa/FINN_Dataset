# EDA flows and Datasets for AI-assisted FPGA designs

This repository contains a dataset to help synthesize quantized convolutional neural networks with the FINN framework. It is built to suggest optimal folding configuration files that are crucial to obtain a synthesized netlist with the needed area/performance ratio.

## Scripts

- `mod_gen.py`: script that randomly generates a certain number of small and large models using PyTorch and Brevitas in `.py` format. It also automatically converts said models to `.onnx` format (so that they can be accessed by the FINN framework) and saves their weights, because their shapes are essential to generate valid configuration files.

*Note*: the layers that are supported at the moment are reported in the Appendix, at the end of the file.

- `config_gen.py`: script that randomly generates, for each model, a certain amount of configutation files that will then be used in FINN.

## Usage

*Note*: To generate a dataset, FINN is required; therefore, it is necessary to have it installed and to clone this repository in a location that can be easily accessed while inside the docker container in which FINN runs.

First, it is needed to create and access an environment with Python 3.11 installed:

```sh
python3.11 -m venv environment_name
source ./environment_name/bin/activate
```

Then, install the required packages through the requirements file:

```sh
pip install -r requirements.txt
```

Once the workspace is set up, the first script, `mod_gen.py`, to generate the models can be run. To specify the number of models to be generated, it is sufficient to modify the values of the variables `NUM_SMALL` and `NUM_LARGE` respectively at lines 17 and 18.
Then, simply run:

```sh
python3.11 mod_gen.py
```

Then, it is possible to run `config_gen.py` to generate the associated configuration files; again, it possible to specify the number of configurations to be generated for each model by modifying the value of the variable `N_CONF` at line 15.
Run the script with:

```sh
python3.11 config_gen.py
```

## Appendix
The currently supported layers are:
- Input formatting: `QuantIdentity`;
- Convolution: `QuantConv2d`;
- Activation: `QuantReLU`;
- Pooling: `MaxPool2d`;
- Flatten: `Flatten`;
- Linearization: `QuantLinear`;