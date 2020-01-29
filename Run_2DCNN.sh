#!/bin/sh


PHASE=$1

SET='pdb'
NUMCLASS=256
DATADIR=data_${SET}
FEATSDIR=feats_${SET}

MODELDIR=models_${SET}/2DCNN_CM
mkdir -p ${MODELDIR}


# 2D-CNN
python scripts/main.py --phase=${PHASE} \
--batch_size=64 --num_epochs=100 --init_lr=0.0005 --lr_sched='True' \
--net_type='cnn2d' --feats_type='nofeats' --input_dim=1 \
--channel_dims='64_512' --filter_sizes='5_5' --fc_dim=256 --num_classes=${NUMCLASS} \
--model_dir=${MODELDIR} --train_file=${DATADIR}/train.list --valid_file=${DATADIR}/valid.names \
--feats_dir=${FEATSDIR} --icvec_file=${DATADIR}/icVec.npy \
--model_file=${MODELDIR}/model.pth.tar --test_file=${DATADIR}/test.names --save_file=${MODELDIR}/test_pred.pkl \
>> ${MODELDIR}/${PHASE}.txt
