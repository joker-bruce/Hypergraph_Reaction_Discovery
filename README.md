# 📘 ChemHGNN: A Hierarchical Hypergraph Neural Network for Reaction Virtual Screening and Discovery

ChemHGNN is a scalable framework for chemical reaction modeling using hierarchical hypergraph neural networks. This repository supports reproducibility of experiments presented in our study, including data preprocessing, training, and evaluation.

---

## 🔧 Setup Instructions

### ✅ Environment Requirements

- Python 3.9.21  
- CUDA 11.8  
- PyTorch 2.2.1  
- DGL for CUDA 11.8  

### 📦 Installation

1. Create a python virtual environment and activate it with:
```
python -m venv chemhgnn_env
source chemhgnn_env/bin/activate
```
2. Please install the DGL library first using command below:
```
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.2/cu118/repo.html
```

3. Then install torch version 2.2.1 directly:
```
pip install torch==2.2.1
```
4. Then install the requirements.txt using the command below:
```
pip install -r requirements.txt
```
**Note**: Installation via the recommended method is essential. Errors may occur if installed differently.

## 📁 Dataset
Download the datasets from the [link](https://drive.google.com/drive/folders/1MIXddERcv0scsVF5j6LFL8fooKtL6wdx?usp=drive_link) and put it under the [data_neg_balanced_smiles](\data_neg_balanced_smiles)

preprocessing
The data is preprocessed already and dataset can be found in the data\_neg\_balanced\_smiles. Nevertheless, if you want to preprocess the data again, use the following command:


There are five splits automated generated for five-fold cross validation

## 🧠 Model architecture
[📄 View the Model architecture](pictures/pipeline_overview.pdf)
![Page 1](pictures/pipeline_overview.png)

## 🚀 Training
Please use the following command:
```
python HGNN_ns_new_loss.py --dataset_name hyperedge_1000  --epochs 50 --train_label mix --aggre_method sum --loss_type mse --folder_name exp_bal_1000_ecfp6_mix_sum_mse --encoding ecfp6_feat
```

After training, the result and checkpoint of the best epoch can be found at data\_neg\_balanced\_smiles/checkpoints/\$folder\_name\$

## 📓  Notebooks
Jupyter notebooks with additional experiments and ablation studies are available in directory [nbs](nbs/).
