# EDA flows and Datasets for AI-assisted FPGA designs

This repository contains a dataset to help synthesize quantized convolutional neural networks with the FINN framework. It is built to suggest optimal folding configuration files that are crucial to obtain a synthesized netlist with the needed area/performance ratio.

## Scripts

## Usage

*Note*: To generate a dataset, FINN will be required; therefore, it is necessary to have it installed and to clone this repository in a location that can be easily accessed while inside the docker container in which FINN runs.

First, it is needed to create and access an environment with Python 3.11 installed:

```sh
python3.11 -m venv environment_name
source ./environment_name/bin/activate
```

Then, install the required packages through the requirements file:

```sh
pip install -r requirements.txt
```
