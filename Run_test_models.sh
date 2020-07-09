#!/bin/sh


# PDB dataset
./Run_baseline.sh pdb onehot
./Run_baseline.sh pdb embeddings

./Run_mlp.sh pdb onehot test
./Run_mlp.sh pdb embeddings test

./Run_conv.sh pdb cnn1d onehot test
./Run_conv.sh pdb cnn1d embeddings test
./Run_conv.sh pdb cnn2d nofeats test
./Run_conv.sh pdb cnn1d2d onehot test
./Run_conv.sh pdb cnn1d2d embeddings test

./Run_graphconv.sh gcn 3 onehot normal test
./Run_graphconv.sh gcn 3 embeddings normal test
./Run_graphconv.sh gcn 1 degree normal test
./Run_graphconv.sh gcn 1 onehot normal test
./Run_graphconv.sh gcn 1 embeddings normal test
./Run_graphconv.sh gcn 1 embeddings random test
./Run_graphconv.sh gcn 1 embeddings identity test

./Run_graphconv.sh chebcn 1 onehot normal test 2
./Run_graphconv.sh chebcn 1 embeddings normal test 2
./Run_graphconv.sh chebcn 1 onehot normal test 10
./Run_graphconv.sh chebcn 1 embeddings normal test 10
./Run_graphconv.sh gmmcn 1 onehot normal test
./Run_graphconv.sh gmmcn 1 embeddings normal test
./Run_graphconv.sh gincn 1 onehot normal test
./Run_graphconv.sh gincn 1 embeddings normal test


# SP dataset
./Run_baseline.sh sp onehot
./Run_baseline.sh sp embeddings

./Run_mlp.sh sp onehot test
./Run_mlp.sh sp embeddings test

./Run_conv.sh sp cnn1d onehot test
./Run_conv.sh sp cnn1d embeddings test


# CAFA dataset
./Run_baseline.sh cafa embeddings
./Run_mlp.sh cafa embeddings test
./Run_conv.sh cafa cnn1d embeddings test
