#!/bin/sh


ONTOLOGY=$1
PHASE=$2
NUMHIDDENLAYERS=$3
FEATSTYPE='embeddings'


if 	 [[ $ONTOLOGY == 'CCO' ]];  then NUMCLASS=847;
elif [[ $ONTOLOGY == 'MFO' ]];   then NUMCLASS=907;
elif [[ $ONTOLOGY == 'BPO' ]]; then NUMCLASS=4009;
fi


DATADIR=/tudelft.net/staff-bulk/ewi/insy/DBL/smakrod/cafa4/data/lists/$ONTOLOGY/
FEATSDIR=/tudelft.net/staff-bulk/ewi/insy/DBL/smakrod/cafa4/data/dataset/$ONTOLOGY/

INPUTDIM=1024

if [[ $NUMHIDDENLAYERS == 1 ]];  then FCDIM="512";
elif [[ $NUMHIDDENLAYERS == 2 ]];  then FCDIM="512_512";
fi


MODELDIR=models_${SET}/MLP_E

mkdir -p ${MODELDIR}


# MLP
python scripts/main.py --phase=${PHASE} \
--batch_size=64 --num_epochs=300 --init_lr=0.0005 --lr_sched='True' \
--net_type='mlp' --feats_type=${FEATSTYPE} --input_dim=${INPUTDIM} --fc_dim=$FCDIM --num_classes=${NUMCLASS} \
--model_dir=${MODELDIR} --train_file=${DATADIR}/train.names --valid_file=${DATADIR}/valid.names \
--feats_dir=${FEATSDIR} --icvec_file=${DATADIR}/icVec.npy \
--model_file=${MODELDIR}/model.pth.tar --test_file=${DATADIR}/test.names --save_file=${MODELDIR}/test_pred.pkl  --protein-level=1 \
>> ${MODELDIR}/${PHASE}.txt
