# Unsupervised protein embeddings outperform hand-crafted sequence and structure features at predicting molecular function
This repository contains the source code and data needed to reproduce the results of the pre-print:
https://www.biorxiv.org/content/10.1101/2020.04.07.028373v1

## Datasets
For each dataset (_PDB_, _SP_ and _CAFA_) there is a data_* directory with the training/validation/test protein IDs, the information content (IC) vector and the MFO GO term matrix. All the models_* directories and the feats_pdb for the _PDB_ dataset are available in the 4TU.Centre for Research Data (https://data.4tu.nl/) repository:

https://data.4tu.nl/repository/uuid:b88d84e1-7408-40d9-a8fc-d734f852dd7a

The _SP_ and _CAFA_ features are not provided due to their large size. They can be generated using the example code or provided upon request.

## Main dependencies
The pip environment in which the code was tested can be found in requirements.txt. The main dependencies are:
* Python 3.6
* Pytorch 1.2.0
* Pytorch-geometric 1.3.1
* Numpy 1.16.4
* Scikit-learn 0.21.2
* Biopython 1.74

## Feature generation example
The code in Run_example.sh generates a feature dictionary for each protein sample. It contains the protein sequence, the amino acid-level ELMo embeddings, the MFO GO term labels and the protein contact map (only for the _PDB_ dataset).

## Neural network training and test example
Train the MLP_E model in the _PDB_ dataset:
```
python scripts/main.py --phase='train' \
--batch_size=64 --num_epochs=100 --init_lr=0.0005 --lr_sched='True' \
--net_type='mlp' --feats_type='embeddings' --input_dim=1024 --fc_dim=512 \
--num_classes=256 --model_dir=models_pdb/MLP_E \
--train_file=datasets/data_pdb/train.names \--valid_file=datasets/data_pdb/valid.names \
--feats_dir=feats_pdb --icvec_file=datasets/data_pdb/icVec.npy
```

Test the trained MLP_E model in the _PDB_ dataset:
```
python scripts/main.py --phase='test' \
--net_type='mlp' --feats_type='embeddings' --input_dim=1024 --fc_dim=512 \
--num_classes=256 --feats_dir=feats_pdb --icvec_file=datasets/data_pdb/icVec.npy \
--model_file=models_pdb/MLP_E/model.pth.tar --test_file=datasets/data_pdb/test.names \
--save_file=models_pdb/MLP_E/test_pred.pkl
```
