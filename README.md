# accent-embeddings
EECS  225D Project

# Installation
To install the necessary packages, use pip.
```
pip3 install -r requirements.txt
```

If you would like to specify a path to the VCTK dataset using the environment,
create a `.env` file with the following contents.  
```
DATASET_PATH=<path-to-dataset> 
```
or, if running from a notebook, use the `%env` magic.
`DATASET_PATH` should be the directory which contains the VCTK-Corpus-0.92 directory.
If you have not downloaded the VCTK dataset, then you should download it from
[the source](https://datashare.ed.ac.uk/handle/10283/2950).

# Hyper Parameters
You can set hyperparameters for experiments by using the dataclasses
provided in `hyper_params.py`.
Note that the `accent_embed_dim` of each task network must match the `out_dim`
of `MultiTaskParams`.

In order to train a different head at a specified epoch interval, set
`alternate_epoch_interal` in `MultiTaskParams` to be non-zero.
If you would like to use dynamic weight averaging while training, then select
`loss_weighting_strategy = "dwa"` in `MultiTaskParams`.

# Training

To run training, modify `train.py` with the experiment hyperparameters and run
```
python3 train.py
```
By default, losses and model checkpoints are [WandB](https://wandb.ai)
```
