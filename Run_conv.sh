#!/bin/sh


SET=$1
MODELTYPE=$2
# cnn1d, cnn2d, cnn1d2d
FEATSTYPE=$3
PHASE=$4

if [[ $SET == 'pdb' ]]; then NUMCLASS=256;
elif [[ $SET == 'sp' ]]; then NUMCLASS=441;
elif [[ $SET == 'cafa' ]]; then NUMCLASS=679;
fi

DATADIR=datasets/data_${SET}
FEATSDIR=feats_${SET}

if [[ $FEATSTYPE == 'nofeats' ]]; then
	INPUTDIM=1
    SUFFIX=''
elif [[ $FEATSTYPE == 'onehot' ]]; then
	INPUTDIM=26
	SUFFIX='_1h'
elif [[ $FEATSTYPE == 'embeddings' ]]; then
	INPUTDIM=1024
    SUFFIX='_E'
fi

if [[ $MODELTYPE == 'cnn1d' ]]; then
    FCDIM=0
    MODELDIR=models_${SET}/1DCNN${SUFFIX}
elif [[ $MODELTYPE == 'cnn1d2d' ]]; then
    FCDIM=0
    MODELDIR=models_${SET}/1DCNN${SUFFIX}+2DCNN_CM
elif [[ $MODELTYPE == 'cnn2d' ]]; then
    FCDIM=256
    MODELDIR=models_${SET}/2DCNN_CM
fi

mkdir -p ${MODELDIR}


# 1D-CNN, 2D-CNN, 1D-CNN+2D-CNN
python scripts/main.py --phase=${PHASE} \
--batch_size=64 --num_epochs=100 --init_lr=0.0005 --lr_sched='True' \
--net_type=${MODELTYPE} --feats_type=${FEATSTYPE} --input_dim=${INPUTDIM} \
--channel_dims='64_512' --filter_sizes='5_5' --fc_dim=${FCDIM} \
--num_classes=${NUMCLASS} --model_dir=${MODELDIR} \
--train_file=${DATADIR}/train.names --valid_file=${DATADIR}/valid.names \
--feats_dir=${FEATSDIR} --icvec_file=${DATADIR}/icVec.npy \
--model_file=${MODELDIR}/model.pth.tar --test_file=${DATADIR}/test.names \
--save_file=${MODELDIR}/test_pred.pkl >> ${MODELDIR}/${PHASE}.txt
