# Deep Learning Template

This is a template for deep learning projects.

## How to use

To instantiate your project using this template, on the [template's GitHub page](https://github.com/ignasa007/ML-Template.git), click `Use this template`, then `Create a new repository`.

Otherwise, if you already have a local git repository, run
```bash
cd ${local-git-repo}
git remote add ml-template https://github.com/ignasa007/ML-Template.git
git fetch ml-template
git merge --allow-unrelated-histories ml-template/main
git remote remove ml-template
```

## Directory Structure

- `assets` - plots generated from different experiments.
- `config` - configuration files for different architectures, datasets, and optimizers.
- `data` - Python classes to handle different datasets, and make them suitable for training.
- `datastore` - raw datasets store.
- `models` - Python classes to assemble models.
- `results` - results of the different runs.
    - `directory structure` - `./${dataset}/${architecture}/${timestamp}/...`
- `utils` - utility functions for running the transformer experiments.
- `main.py` - main file for training the models.

## Setup

```bash
conda create --name ${env_name} --file requirements.txt
conda activate ${env_name}
```

## Execution

To run the transformer experiments, execute
```bash
python3 -m main \
    --dataset ${dataset} \
    --architecture ${model} \
    --device_index ${CUDA_VISIBLE_DEVICES}
```
You can override configurations in `configs` using the command line, e.g. 
```bash
python3 -m main \
    --dataset ${dataset} \
    --architecture ${architecture} \
    --device_index ${CUDA_VISIBLE_DEVICES} \
    data.batch_size 64 \
    model.width 256
```

Notes: 
- Make sure to omit `--device_index` if you do not wish to use a GPU.
- Currently, the project supports up to 1 GPU per run, since I don't know how to distribute computing over multiple GPUs :').