# ML-Template

This is a template for Machine Learning projects.

## How to use

To instantiate your project using this template, on the [template's GitHub page](https://github.com/ignasa007/ML-Template.git), click `Use this template`, then `Create a new repository`.

Otheriwse, if you already have a local git repository, run
```bash
cd <local-git-repo>
git remote add ml-template https://github.com/ignasa007/ML-Template.git
git fetch ml-template
git merge --allow-unrelated-histories ml-template/main
git remote remove ml-template
```

## Directory Structure

- `assets` - plots generated from different experiments.
- `config` - configuration files for different datasets and models.
- `data` - raw datasets store.
- `data_classes` - Python classes to handle different datasets, and make them suitable for training.
- `model_classes` - Python classes to handle different models.
- `results` - results of the different runs. <br>
    - `directory structure` - `<dataset>` -> `<model>` -> `<run-date>` -> `logs` and `<data-split>_results`
- `utils` - utility functions for running the transformer experiments.
- `main.py` - main file for training the models.
- `inference.py` - main file for testing the models.

## Setup

```bash
conda create --name <env-name> --file requirements.txt python=3.8
conda activate <env-name>
```

## Execution

To run the transformer experiments, execute
```bash
python3 -B main.py \
    --dataset <dataset> \
    --model <model>
```
You can also override default configurations using the command line.<br>

For inference, execute
```bash
python3 -B inference.py \
    --dataset <dataset> \
    --model <model> \
    --weights <path-to-weights>
```

Note: Make sure to set the device index to <i>None</i> if you do not wish to use the GPU.