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

# network architecture
# encoder related
etype=blstmp     # encoder architecture type
elayers=2
eunits=320
eprojs=320
subsample=1_2_2_1 # skip every n frame from input to nth layers
# decoder related
dlayers=1
dunits=300
# attention related
atype=add
adim=320
awin=5
aheads=4
aconv_chans=10
aconv_filts=100

# hybrid CTC/attention
mtlalpha=0.2

# label smoothing
lsm_type=unigram
lsm_weight=0.05

# minibatch related
batchsize=30
maxlen_in=800  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=150 # if output length > maxlen_out, batchsize is automatically reduced

# optimization related
opt=adadelta
epochs=30

# decoding parameter
lm_weight=0
beam_size=5 # TODO how many in total? csdi + <eos>
penalty=0.0
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0.3
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# scheduled sampling option
samp_prob=0.0


# exp tag
tag="" # tag for managing experiments.


################ input and output related ##################
errtype=wer
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
outdir=exp_csdiseq/expt_${bnftype}_csdiseq_${errtype}_${idx}
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
aurora4_test_eval92_clean_wv1 aurora4_test_eval92_clean_wv2"


feat_tr_dir=${dumpdir}/${train_set}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_set}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
dict=data/lang_1char/csdi.txt

if [ -z ${tag} ]; then
    expdir=${outdir}/expt_${backend}_${etype}_e${elayers}_subsample${subsample}_unit${eunits}_proj${eprojs}_d${dlayers}_unit${dunits}_${atype}_aconvc${aconv_chans}_aconvf${aconv_filts}_mtlalpha${mtlalpha}_${opt}_sampprob${samp_prob}_bs${batchsize}_mli${maxlen_in}_mlo${maxlen_out}
    if [ "${lsm_type}" != "" ]; then
        expdir=${expdir}_lsm${lsm_type}${lsm_weight}
    fi
    if ${do_delta}; then
        expdir=${expdir}_delta
    fi
else
    expdir=${outdir}/expt_${backend}_${tag}
fi
mkdir -p ${expdir}
echo $train_set > $expdir/info_train_set


if [ ${stage} -le 1 ]; then
    echo "stage 1: CSDIseq Network Training"

    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --seed ${seed} \
        --train-json ${feat_tr_dir}/data_csdiseq_${errtype}.json \
        --valid-json ${feat_dt_dir}/data_csdiseq_${errtype}.json \
        --etype ${etype} \
        --elayers ${elayers} \
        --eunits ${eunits} \
        --eprojs ${eprojs} \
        --subsample ${subsample} \
        --dlayers ${dlayers} \
        --dunits ${dunits} \
        --atype ${atype} \
        --adim ${adim} \
        --awin ${awin} \
        --aheads ${aheads} \
        --aconv-chans ${aconv_chans} \
        --aconv-filts ${aconv_filts} \
        --mtlalpha ${mtlalpha} \
        --lsm-type ${lsm_type} \
        --lsm-weight ${lsm_weight} \
        --batch-size ${batchsize} \
        --maxlen-in ${maxlen_in} \
        --maxlen-out ${maxlen_out} \
        --sampling-probability ${samp_prob} \
        --opt ${opt} \
        --epochs ${epochs}
fi


if [ ${stage} -le 2 ]; then
    echo "stage 2: Decoding"
    nj=32

    for rtask in ${recog_set}; do
    (
        decode_dir=decode_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}/${rtask}
        feat_recog_dir=${dumpdir}/${train_set}/${rtask}/delta${do_delta}
        recog_opts=

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_csdiseq_${errtype}.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_csdiseq_${errtype}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data_csdiseq_${errtype}.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            --beam-size ${beam_size} \
            --penalty ${penalty} \
            --maxlenratio ${maxlenratio} \
            --minlenratio ${minlenratio} \
            --ctc-weight ${ctc_weight} \
            --lm-weight ${lm_weight} \
            $recog_opts &
        wait

        [ -e ${expdir}/${decode_dir}/data_csdiseq_${errtype}.json ] && rm ${expdir}/${decode_dir}/data_csdiseq_${errtype}.json
        score_sclite_csdiseq.sh --wer true ${expdir}/${decode_dir} ${dict} data_csdiseq_${errtype}

    ) &
    done
    wait
    echo "Finished"
fi

