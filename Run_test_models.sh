#!/bin/sh


# PDB dataset
./Run_BASELINE.sh pdb onehot
./Run_BASELINE.sh pdb embeddings

./Run_MLP.sh pdb onehot test
./Run_MLP.sh pdb embeddings test

./Run_1DCNN.sh pdb onehot test
./Run_1DCNN.sh pdb embeddings test

./Run_GCN3.sh onehot test
./Run_GCN3.sh embeddings test

./Run_GCN1.sh nofeats test
./Run_GCN1.sh onehot test
./Run_GCN1.sh embeddings test

./Run_GCN1_perturb.sh random test
./Run_GCN1_perturb.sh identity test

./Run_CHEBCN.sh 2 nofeats test
./Run_CHEBCN.sh 2 onehot test
./Run_CHEBCN.sh 2 embeddings test
./Run_CHEBCN.sh 10 nofeats test
./Run_CHEBCN.sh 10 onehot test
./Run_CHEBCN.sh 10 embeddings test

./Run_GINCN.sh nofeats test
./Run_GINCN.sh onehot test
./Run_GINCN.sh embeddings test

./Run_GMMCN.sh nofeats test
./Run_GMMCN.sh onehot test
./Run_GMMCN.sh embeddings test

./Run_2DCNN.sh test

./Run_1DCNN-2DCNN.sh onehot test
./Run_1DCNN-2DCNN.sh embeddings test



# SP dataset
./Run_BASELINE.sh sp onehot
./Run_BASELINE.sh sp embeddings

./Run_MLP.sh sp onehot test
./Run_MLP.sh sp embeddings test

./Run_1DCNN.sh sp onehot test
./Run_1DCNN.sh sp embeddings test



# CAFA dataset
./Run_BASELINE.sh cafa embeddings
./Run_MLP.sh cafa embeddings test
./Run_1DCNN.sh cafa embeddings test
