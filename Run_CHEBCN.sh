#!/bin/sh


CHEBORDER=$1
FEATSTYPE=$2
PHASE=$3

SET='pdb'
NUMCLASS=256
DATADIR=data_${SET}
FEATSDIR=feats_${SET}

if [[ $FEATSTYPE == 'nofeats' ]]; then
	INPUTDIM=1
	FCDIM=256
	MODELDIR=models_${SET}/CHEBCN_K${CHEBORDER}_CM
elif [[ $FEATSTYPE == 'onehot' ]]; then
	INPUTDIM=26
	FCDIM=256
	MODELDIR=models_${SET}/CHEBCN_K${CHEBORDER}_1h_CM
elif [[ $FEATSTYPE == 'embeddings' ]]; then
	INPUTDIM=1024
	FCDIM=0
	MODELDIR=models_${SET}/CHEBCN_K${CHEBORDER}_E_CM
fi

mkdir -p ${MODELDIR}


# 1-layer ChebCN
python scripts/main.py --phase=${PHASE} --cheb_order=${CHEBORDER} \
--batch_size=64 --num_epochs=100 --init_lr=0.0005 --lr_sched='True' \
--net_type='chebcn' --feats_type=${FEATSTYPE} --input_dim=${INPUTDIM} \
--channel_dims='512' --fc_dim=${FCDIM} --num_classes=${NUMCLASS} \
--model_dir=${MODELDIR} --train_file=${DATADIR}/train.names --valid_file=${DATADIR}/valid.names \
--feats_dir=${FEATSDIR} --icvec_file=${DATADIR}/icVec.npy \
--model_file=${MODELDIR}/model.pth.tar --test_file=${DATADIR}/test.names --save_file=${MODELDIR}/test_pred.pkl \
>> ${MODELDIR}/${PHASE}.txt
