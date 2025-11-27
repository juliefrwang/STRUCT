# STRUCT (previously named as SPLASH-structure)

This repository contains the official implementation accompanying our manuscript: STRUCT: a statistical approach to identify RNA secondary structures from raw sequencing data, bypassing multiple sequence alignment. Read the paper on [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.10.03.616574v2).

## Setting up the Python Environment

It is recommended to use a virtual environment to manage dependencies. Ensure you have Python `>= 3.9` installed. Then, follow these steps:

```bash
# Create a virtual environment (you can replace 'my_env' with any name)
python3 -m venv my_env

# Activate the virtual environment
source my_env/bin/activate
```

## Installing STRUCT
You can install STRUCT by cloning the repository and using pip to install the package and its dependencies:

```bash
# Clone the repository
git clone https://github.com/juliefrwang/splash-structure.git

# Navigate to the project directory
cd splash-structure

# Install the package
pip install .
```

## Installing Julia programming language
STRUCT relies on Julia for some computations. Follow the steps below to install Julia and set up the required packages.

### Installing Julia
You can download and install Julia from the official website: https://julialang.org/downloads/.
After installing, verify your Julia installation:
```bash
julia --version
```

### Installing Julia Packages
Install the required Julia packages for this project, run:
```bash
julia -e 'using Pkg; Pkg.add("DataFrames"); Pkg.add("CSV"); Pkg.add("Combinatorics")'
```

## Usage
After installation, you can run STRUCT directly from the command line. STRUCT provides two handy commands: `ss-target` for executing target mode and `ss-compactor` for running compactor mode.
### Target mode syntax:
```bash
ss-target <splash_output_file> <output_prefix>
```
__Positional Arguments__
1. `<splash_output_file>`: Path to the SPLASH output file. For SPLASH, please see: https://github.com/refresh-bio/SPLASH.
2. `<output_prefix>`: Prefix for naming the output result folder. 

### Compactor mode syntax:
```bash
ss-compactor <compactor_file> <output_prefix>
```
__Positional Arguments__
1. `<compactor_file>`:Path to the compactor file.
2. `<output_prefix>`: Prefix for naming the output result folder.

## Example runs on test data
There are two files in `tests/test_data/`: `test.after_correction.scores.tsv`, a test SPLASH output file, and `test_compactor.tsv`, a test compactor file. To run STRUCT from `splash-structure` folder with an output folder prefix `new_test`:
### Run target mode
```bash
ss-target new_test tests/test_data/test.after_correction.scores.tsv
```
### Run compactor mode
```bash
ss-compactor new_test tests/test_data/test.compactor.tsv
```
The output will be saved in the `new_test_results` folder. The file `structure_on_targets.tsv` contains the target mode results, and `structure_on_compactors.tsv` contains the compactor mode results. The subfolder `interm_compactor` contains an intermediate file for processed compactors before the algorithm searches for compensatory stems.
