#!/bin/sh


SET=$1
FEATSTYPE=$2

if [[ $SET == 'pdb' ]]; then NUMCLASS=256;
elif [[ $SET == 'sp' ]]; then NUMCLASS=441;
elif [[ $SET == 'cafa' ]]; then NUMCLASS=679;
fi

DATADIR=datasets/data_${SET}
FEATSDIR=feats_${SET}

OUTDIR=models_${SET}/BASELINE
mkdir -p ${OUTDIR}

python scripts/naive_knn_lr.py ${DATADIR} ${FEATSDIR} ${NUMCLASS} ${FEATSTYPE} ${OUTDIR} >> ${OUTDIR}/test.txt
