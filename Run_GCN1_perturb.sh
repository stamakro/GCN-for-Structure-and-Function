#!/bin/sh


EDGESTYPE=$1
PHASE=$2

FEATSTYPE='embeddings'
INPUTDIM=1024

SET='pdb'
NUMCLASS=256
DATADIR=data_${SET}
FEATSDIR=feats_${SET}

if [[ $EDGESTYPE == 'random' ]]; then
	FCDIM=0
	MODELDIR=models_${SET}/GCN1_E_R

elif [[ $EDGESTYPE == 'identity' ]]; then
	FCDIM=0
	MODELDIR=models_${SET}/GCN1_E_I
fi

mkdir -p ${MODELDIR}


# 1-layer GCN
python scripts/main.py --phase=${PHASE} \
--batch_size=64 --num_epochs=100 --init_lr=0.0005 --lr_sched='True' \
--net_type='gcn' --feats_type=${FEATSTYPE} --edges_type=${EDGESTYPE} --input_dim=${INPUTDIM} \
--channel_dims='512' --fc_dim=${FCDIM} --num_classes=${NUMCLASS} \
--model_dir=${MODELDIR} --train_file=${DATADIR}/train.names --valid_file=${DATADIR}/valid.names \
--feats_dir=${FEATSDIR} --icvec_file=${DATADIR}/icVec.npy \
--model_file=${MODELDIR}/model.pth.tar --test_file=${DATADIR}/test.names --save_file=${MODELDIR}/test_pred.pkl \
>> ${MODELDIR}/${PHASE}.txt
