# accent-embeddings
EECS  225D Project

# Installation
To install the necessary packages, use pip.
```
pip3 install -r requirements.txt
```

To specify a path to the dataset, create a `.env` file with the following
contents.
```
DATASET_PATH=<path-to-dataset>
```
This should be the directory which contains the VCTK-Corpus-0.92.zip file (which
will be downloaded automatically if it does not exist at `DATASET_PATH`. If
`DATASET_PATH` is not specified, then the dataset will be downloaded/searched
for in the current directory.
