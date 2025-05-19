# Hypergraph_Reaction_Discovery

##How to Setup
###Environment
Please install the DGL library first using command below:

Then install the requirement.txt using the command below:


###Data preprocessing
The data is preprocessed already and dataset can be found in the data\_neg\_balanced\_smiles. Nevertheless, if you want to preprocess the data again, use the following command:


There are five splits automated generated for five-fold cross validation

###Model architecture

###Training
Please use the following command:


After training, the result and checkpoint of the best epoch can be found at data\_neg\_balanced\_smiles/checkpoints/\$model\_file\$
