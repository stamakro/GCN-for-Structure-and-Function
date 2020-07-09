#!/bin/sh


MODELTYPE=$1
# gcn, chebcn, gmmcn, gincn
NLAYERS=$2
FEATSTYPE=$3
EDGESTYPE=$4
PHASE=$5

SET='pdb'
NUMCLASS=256
DATADIR=data_${SET}
FEATSDIR=feats_${SET}

CHEBORDER=0
if [[ $MODELTYPE == 'gcn' ]]; then
    if [[ $NLAYERS == 1 ]]; then
        CHDIMS='512'
        PREFIX=GCN1
    elif [[ $NLAYERS == 3 ]]; then
        CHDIMS='256_256_512'
        PREFIX=GCN3
    fi
elif [[ $MODELTYPE == 'chebcn' ]]; then
    CHEBORDER=$6
    CHDIMS='512'
    PREFIX=CHEBCN_K${CHEBORDER}
elif [[ $MODELTYPE == 'gmmcn' ]]; then
    CHDIMS='512'
    PREFIX=GMMCN
elif [[ $MODELTYPE == 'gincn' ]]; then
    CHDIMS='512'
    PREFIX=GINCN
fi

if [[ $FEATSTYPE == 'degree' ]]; then
	INPUTDIM=1
	FCDIM=256
    SUFFIX1=''
elif [[ $FEATSTYPE == 'onehot' ]]; then
	INPUTDIM=26
	FCDIM=256
    SUFFIX1='_1h'
elif [[ $FEATSTYPE == 'embeddings' ]]; then
	INPUTDIM=1024
	FCDIM=0
    SUFFIX1='_E'
fi

if [[ $EDGESTYPE == 'normal' ]]; then SUFFIX2='_CM';
elif [[ $EDGESTYPE == 'random' ]]; then
    FCDIM=0
    SUFFIX2='_R'
elif [[ $EDGESTYPE == 'identity' ]]; then
    FCDIM=0
    SUFFIX2='_I'
fi


MODELDIR=models_${SET}/${PREFIX}${SUFFIX1}${SUFFIX2}
mkdir -p ${MODELDIR}


# GCN (1-layer, 3-layer), CHEBCN, GMMCN, GINCN
python scripts/main.py --phase=${PHASE} --cheb_order=${CHEBORDER} \
--batch_size=64 --num_epochs=100 --init_lr=0.0005 --lr_sched='True' \
--net_type=${MODELTYPE} --feats_type=${FEATSTYPE} --edges_type=${EDGESTYPE} \
--input_dim=${INPUTDIM} --channel_dims=${CHDIMS} --fc_dim=${FCDIM} \
--num_classes=${NUMCLASS} --model_dir=${MODELDIR} \
--train_file=${DATADIR}/train.names --valid_file=${DATADIR}/valid.names \
--feats_dir=${FEATSDIR} --icvec_file=${DATADIR}/icVec.npy \
--model_file=${MODELDIR}/model.pth.tar --test_file=${DATADIR}/test.names \
--save_file=${MODELDIR}/test_pred.pkl >> ${MODELDIR}/${PHASE}.txt
