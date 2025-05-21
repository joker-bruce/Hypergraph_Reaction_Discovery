# Data Preprocessing
## SMILES from USPTO
unzipped data from data.zip
## Environment Setup
Activate the environment from the upper directory
```
source ../chemhgnn_env/bin/activate
```

## Run
To preprocess the data for general purpose, run the following command in the upper directory:
```
python data_chem_ns.py --datapoint 10000 --output_dir ./data_neg_balanced_smiles
```
