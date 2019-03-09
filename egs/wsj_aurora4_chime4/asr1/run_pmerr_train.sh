#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
# Copyright 2019 Johns Hopkins University (Ruizhi Li)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# general configuration
backend=pytorch
stage=0        # start from 0 if you need to start from data preparation
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot
seed=1

# feature configuration
do_delta=false

##################### network architecture ###################
# model type [BlstmpOutFwd, BlstmpAvgFwd]
model_type=BlstmpAvgFwd
# blstmp layers
blayers=2
bunits=320
bprojs=320
subsample=1_2_2_1 # skip every n frame from input to nth layers
# feedforward layers
flayers=1
funits=300
# loss
loss_type=bceloss
# label type [wer, cer]
label_type=cer
# minibatch related
batchsize=30
maxlen_in=800  # if input length  > maxlen_in, batchsize is automatically reduced
# optimization related
opt=adadelta
epochs=30
# decoding
recog_model=model.loss.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'


# exp tag
tag="" # tag for managing experiments.

################ input and output related ##################
bnftype=
dumpdir=exp/wsj_train_si284-aurora4_train_si84_multi_pytorch_blstmp_e4_subsample1_1_1_1_1_unit320_proj320_d1_unit300_add_aconvc10_aconvf100_mtlalpha0.2_adadelta_sampprob0.0_bs15_mli800_mlo150_lsmunigram0.05/dump_bnfenc_beam30_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm0_1layer_unit1000_sgd_bs300_word65000   # directory to dump full features
# set up dump and out
idx=


. utils/parse_options.sh || exit 1;

. ./path.sh
. ./cmd.sh

# Set bash to ' debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

[ -z $idx ] && echo "Have to specify experiment index" && exit 1
[ -z $bnftype ] && echo "Have to specify bottleneck feature type" && exit 1
outdir=exp_pmerr/expt_${bnftype}_err_${label_type}_${idx}
ln -s $PWD/$outdir $dumpdir/

# data (PM)
train_set=aurora4_train_si84_clean-aurora4_train_si84_multi-chime4_tr05_simu_isolated_1ch_track-chime4_tr05_real_isolated_1ch_track
train_dev=wsj_test_dev93-aurora4_dev_0330-chime4_dt05_simu_isolated_1ch_track-chime4_dt05_real_isolated_1ch_track
recog_set="wsj_test_eval92 \
chime4_et05_real_isolated_1ch_track chime4_et05_simu_isolated_1ch_track \
aurora4_test_eval92_street_wv1 aurora4_test_eval92_street_wv2 \
aurora4_test_eval92_airport_wv1 aurora4_test_eval92_airport_wv2 \
aurora4_test_eval92_train_wv1 aurora4_test_eval92_train_wv2 \
aurora4_test_eval92_restaurant_wv1 aurora4_test_eval92_restaurant_wv2 \
aurora4_test_eval92_babble_wv1 aurora4_test_eval92_babble_wv2 \
aurora4_test_eval92_car_wv1 aurora4_test_eval92_car_wv2 \
aurora4_test_eval92_clean_wv1 aurora4_test_eval92_clean_wv2 \
dirha_real_KA6 dirha_real_LA6 dirha_sim_KA6 dirha_sim_LA6"

feat_tr_dir=${dumpdir}/${train_set}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_set}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}


if [ -z ${tag} ]; then
    expdir=${outdir}/expt_${backend}_${model_type}_b${blayers}_subsample${subsample}_unit${bunits}_proj${bprojs}_f${flayers}_unit${funits}_${opt}_${loss_type}_bs${batchsize}_mli${maxlen_in}_label${label_type}
    if ${do_delta}; then
        expdir=${expdir}_delta
    fi
else
    expdir=${outdir}/expt_${backend}_${tag}
fi
mkdir -p ${expdir}
echo $train_set > $expdir/info_train_set


if [ ${stage} -le 1 ]; then
    echo "stage 1: PM ERR Network Training"

    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train_pmerr.py \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --debugmode ${debugmode} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --seed ${seed} \
        --train-json ${feat_tr_dir}/data_err.json \
        --valid-json ${feat_dt_dir}/data_err.json \
        --label-type ${label_type} \
        --model-type ${model_type} \
        --blayers ${blayers} \
        --bunits ${bunits} \
        --bprojs ${bprojs} \
        --flayers ${flayers} \
        --funits ${funits} \
        --subsample ${subsample} \
        --loss-type ${loss_type} \
        --batch-size ${batchsize} \
        --maxlen-in ${maxlen_in} \
        --opt ${opt} \
        --epochs ${epochs}
fi


if [ ${stage} -le 2 ]; then
    echo "stage 2: Decoding"
    nj=32

    for rtask in ${recog_set} ${train_set}; do
    (
        decode_dir=decode/${rtask}
        feat_recog_dir=${dumpdir}/${train_set}/${rtask}/delta${do_delta}
        recog_opts=

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_err.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog_pmerr.py \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_err.JOB.json \
            --result-label ${expdir}/${decode_dir}/data_err.JOB.json \
            --model ${expdir}/results/${recog_model} \
            --batchsize 20 \
            $recog_opts &
        wait

        concatjson.py ${expdir}/${decode_dir}/data_err.*.json > ${expdir}/${decode_dir}/data_err.json
        json2text_pmerr.py ${expdir}/${decode_dir}/data_err.json $label_type ${expdir}/${decode_dir}/result.txt

    ) &
    done
    wait
    echo "Finished"
fi

if [ ${stage} -le 3 ]; then
    echo "stage 3: Organizing decoding results for train and test"
    nj=32

    # train
    result=exp_pmerr/expt_${bnftype}_err_${label_type}_${idx}.train.txt
    cp ${expdir}/decode/${train_set}/result.txt $result

    # test
    result=exp_pmerr/expt_${bnftype}_err_${label_type}_${idx}.test.txt
    [ -f $result ] && rm $result
    touch $result
    for rtask in ${recog_set}; do
        cat ${expdir}/decode/${rtask}/result.txt >> $result
    done
    echo "Finished"
fi

