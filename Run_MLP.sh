#!/bin/sh


SET=$1
FEATSTYPE=$2
PHASE=$3

if 	 [[ $SET == 'pdb' ]];  then NUMCLASS=256;
elif [[ $SET == 'sp' ]];   then NUMCLASS=441;
elif [[ $SET == 'cafa' ]]; then NUMCLASS=679;
fi

DATADIR=data_${SET}
FEATSDIR=feats_${SET}

if [[ $FEATSTYPE == 'onehot' ]]; then
	INPUTDIM=26
	MODELDIR=models_${SET}/MLP_1h
elif [[ $FEATSTYPE == 'embeddings' ]]; then
	INPUTDIM=1024
	MODELDIR=models_${SET}/MLP_E
fi

mkdir -p ${MODELDIR}


# MLP
python scripts/main.py --phase=${PHASE} \
--batch_size=64 --num_epochs=300 --init_lr=0.0005 --lr_sched='True' \
--net_type='mlp' --feats_type=${FEATSTYPE} --input_dim=${INPUTDIM} --fc_dim=512 --num_classes=${NUMCLASS} \
--model_dir=${MODELDIR} --train_file=${DATADIR}/train.names --valid_file=${DATADIR}/valid.names \
--feats_dir=${FEATSDIR} --icvec_file=${DATADIR}/icVec.npy \
--model_file=${MODELDIR}/model.pth.tar --test_file=${DATADIR}/test.names --save_file=${MODELDIR}/test_pred.pkl \
>> ${MODELDIR}/${PHASE}.txt
