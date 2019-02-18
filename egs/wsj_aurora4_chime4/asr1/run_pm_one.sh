#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
# Copyright 2019 Johns Hopkins University (Ruizhi Li)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh


# general configuration
backend=chainer
stage=0        # start from 0 if you need to start from data preparation
ngpu=0         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot
seed=1

########### ASR configs #############
## feature configuration
do_delta=false

# network architecture
# encoder related
etype=vggblstmp     # encoder architecture type
elayers=6
eunits=320
eprojs=320
subsample=1_2_2_1_1 # skip every n frame from input to nth layers
# decoder related
dlayers=1
dunits=300
# attention related
atype=location
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
epochs=15

# rnnlm related
use_wordlm=true     # false means to train/use a character LM
lm_vocabsize=65000  # effective only for word LMs
lm_layers=1         # 2 for character LMs
lm_units=1000       # 650 for character LMs
lm_opt=sgd          # adam for character LMs
lm_batchsize=300    # 1024 for character LMs
lm_epochs=20        # number of epochs
lm_maxlen=40        # 150 for character LMs
lm_resume=          # specify a snapshot file to resume LM training
lmtag=              # tag for managing LMs

# decoding parameter
lm_weight=1.0
beam_size=30
penalty=0.0
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0.3
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# scheduled sampling option
samp_prob=0.0


# exp tag
tag="" # tag for managing experiments.

########### PM configs #############
# bnf components
bnf_component=enc
bnf_batchsize=5

# asr related
expdir=exp/wsj_train_si284-aurora4_train_si84_multi_pytorch_blstmp_e4_subsample1_1_1_1_1_unit320_proj320_d1_unit300_add_aconvc10_aconvf100_mtlalpha0.2_adadelta_sampprob0.0_bs15_mli800_mlo150_lsmunigram0.05

. utils/parse_options.sh || exit 1;

. ./path.sh
. ./cmd.sh

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# data (asr)
train_set=wsj_train_si284-aurora4_train_si84_multi
train_dev=wsj_test_dev93-aurora4_dev_0330
recog_set="wsj_test_eval92 \
chime4_et05_real_isolated_1ch_track chime4_et05_simu_isolated_1ch_track \
aurora4_test_eval92_street_wv1 aurora4_test_eval92_street_wv2 \
aurora4_test_eval92_airport_wv1 aurora4_test_eval92_airport_wv2 \
aurora4_test_eval92_train_wv1 aurora4_test_eval92_train_wv2 \
aurora4_test_eval92_restaurant_wv1 aurora4_test_eval92_restaurant_wv2 \
aurora4_test_eval92_babble_wv1 aurora4_test_eval92_babble_wv2 \
aurora4_test_eval92_car_wv1 aurora4_test_eval92_car_wv2 \
aurora4_test_eval92_clean_wv1 aurora4_test_eval92_clean_wv2"

train_dirs="wsj_train_si284 aurora4_train_si84_multi wsj_test_dev93 aurora4_dev_0330"
extra_set="aurora4_train_si84_clean" # for PM use



# data used for PM
pm_train_set=aurora4_train_si84_clean-aurora4_train_si84_multi-chime4_tr05_simu_isolated_1ch_track-chime4_tr05_real_isolated_1ch_track
pm_train_dev=wsj_test_dev93-aurora4_dev_0330-chime4_dt05_simu_isolated_1ch_track-chime4_dt05_real_isolated_1ch_track
pm_train_dirs="aurora4_train_si84_clean aurora4_train_si84_multi \
wsj_test_dev93 aurora4_dev_0330 \
chime4_tr05_simu_isolated_1ch_track chime4_tr05_real_isolated_1ch_track \
chime4_dt05_simu_isolated_1ch_track chime4_dt05_real_isolated_1ch_track"
pm_extra_set=

# set up dump
dumpdir=${dumpdir}/${train_set}
pmdumpdir=${expdir}/dump_bnf_e${recog_model}_${bnf_component}

dict=data/lang_1char/${train_set}_units.txt
nlsyms=data/lang_1char/non_lang_syms.txt

if [ -z ${lmtag} ]; then
    lmtag=${lm_layers}layer_unit${lm_units}_${lm_opt}_bs${lm_batchsize}
    if [ $use_wordlm = true ]; then
        lmtag=${lmtag}_word${lm_vocabsize}
    fi
fi
lmexpdir=exp/train_rnnlm_${backend}_${lmtag}



if [ ${stage} -le 2 ]; then
    echo "stage 2: Decode on Train and Dev (ASR)"
    nj=32

#rtask=aurora4_dev_0330
#(
#    decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}_rnnlm${lm_weight}_${lmtag}
#    if [ $use_wordlm = true ]; then
#        recog_opts="--word-rnnlm ${lmexpdir}/rnnlm.model.best"
#    else
#        recog_opts="--rnnlm ${lmexpdir}/rnnlm.model.best"
#    fi
#    if [ $lm_weight == 0 ]; then
#        recog_opts=""
#    fi
#    feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
#
#    #### use CPU for decoding
#    ngpu=0
#
##    ${decode_cmd} JOB=23 ${expdir}/${decode_dir}/log/decode.JOB.log \
##        asr_recog.py \
##        --ngpu ${ngpu} \
##        --backend ${backend} \
##        --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
##        --result-label ${expdir}/${decode_dir}/data.JOB.json \
##        --model ${expdir}/results/${recog_model}  \
##        --beam-size ${beam_size} \
##        --penalty ${penalty} \
##        --maxlenratio ${maxlenratio} \
##        --minlenratio ${minlenratio} \
##        --ctc-weight ${ctc_weight} \
##        --lm-weight ${lm_weight} \
##        $recog_opts &
#
#    score_sclite.sh --wer true --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}
#) &
#
#
##rtask=aurora4_train_si84_clean
##(
##    decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}_rnnlm${lm_weight}_${lmtag}
##    if [ $use_wordlm = true ]; then
##        recog_opts="--word-rnnlm ${lmexpdir}/rnnlm.model.best"
##    else
##        recog_opts="--rnnlm ${lmexpdir}/rnnlm.model.best"
##    fi
##    if [ $lm_weight == 0 ]; then
##        recog_opts=""
##    fi
##    feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
##
##    #### use CPU for decoding
##    ngpu=0
##
##    ${decode_cmd} JOB=5 ${expdir}/${decode_dir}/log/decode.JOB.log \
##        asr_recog.py \
##        --ngpu ${ngpu} \
##        --backend ${backend} \
##        --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
##        --result-label ${expdir}/${decode_dir}/data.JOB.json \
##        --model ${expdir}/results/${recog_model}  \
##        --beam-size ${beam_size} \
##        --penalty ${penalty} \
##        --maxlenratio ${maxlenratio} \
##        --minlenratio ${minlenratio} \
##        --ctc-weight ${ctc_weight} \
##        --lm-weight ${lm_weight} \
##        $recog_opts &
##
##    score_sclite.sh --wer true --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}
##) &
#
#rtask=aurora4_train_si84_clean
#(
#    decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}_rnnlm${lm_weight}_${lmtag}
#    if [ $use_wordlm = true ]; then
#        recog_opts="--word-rnnlm ${lmexpdir}/rnnlm.model.best"
#    else
#        recog_opts="--rnnlm ${lmexpdir}/rnnlm.model.best"
#    fi
#    if [ $lm_weight == 0 ]; then
#        recog_opts=""
#    fi
#    feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
#
#    #### use CPU for decoding
#    ngpu=0
#
##    ${decode_cmd} JOB=26 ${expdir}/${decode_dir}/log/decode.JOB.log \
##        asr_recog.py \
##        --ngpu ${ngpu} \
##        --backend ${backend} \
##        --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
##        --result-label ${expdir}/${decode_dir}/data.JOB.json \
##        --model ${expdir}/results/${recog_model}  \
##        --beam-size ${beam_size} \
##        --penalty ${penalty} \
##        --maxlenratio ${maxlenratio} \
##        --minlenratio ${minlenratio} \
##        --ctc-weight ${ctc_weight} \
##        --lm-weight ${lm_weight} \
##        $recog_opts &
#
#    score_sclite.sh --wer true --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}
#) &


rtask=aurora4_train_si84_multi
(
    decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}_rnnlm${lm_weight}_${lmtag}
    if [ $use_wordlm = true ]; then
        recog_opts="--word-rnnlm ${lmexpdir}/rnnlm.model.best"
    else
        recog_opts="--rnnlm ${lmexpdir}/rnnlm.model.best"
    fi
    if [ $lm_weight == 0 ]; then
        recog_opts=""
    fi
    feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

    #### use CPU for decoding
    ngpu=0

    ${decode_cmd} JOB=21 ${expdir}/${decode_dir}/log/decode.JOB.log \
        asr_recog.py \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
        --result-label ${expdir}/${decode_dir}/data.JOB.json \
        --model ${expdir}/results/${recog_model}  \
        --beam-size ${beam_size} \
        --penalty ${penalty} \
        --maxlenratio ${maxlenratio} \
        --minlenratio ${minlenratio} \
        --ctc-weight ${ctc_weight} \
        --lm-weight ${lm_weight} \
        $recog_opts &

#    score_sclite.sh --wer true --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}
) &

rtask=chime4_tr05_simu_isolated_1ch_track
(
    decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}_rnnlm${lm_weight}_${lmtag}
    if [ $use_wordlm = true ]; then
        recog_opts="--word-rnnlm ${lmexpdir}/rnnlm.model.best"
    else
        recog_opts="--rnnlm ${lmexpdir}/rnnlm.model.best"
    fi
    if [ $lm_weight == 0 ]; then
        recog_opts=""
    fi
    feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

    #### use CPU for decoding
    ngpu=0

    ${decode_cmd} JOB=20 ${expdir}/${decode_dir}/log/decode.JOB.log \
        asr_recog.py \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
        --result-label ${expdir}/${decode_dir}/data.JOB.json \
        --model ${expdir}/results/${recog_model}  \
        --beam-size ${beam_size} \
        --penalty ${penalty} \
        --maxlenratio ${maxlenratio} \
        --minlenratio ${minlenratio} \
        --ctc-weight ${ctc_weight} \
        --lm-weight ${lm_weight} \
        $recog_opts &

#    score_sclite.sh --wer true --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}
) &


rtask=chime4_tr05_simu_isolated_1ch_track
(
    decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}_rnnlm${lm_weight}_${lmtag}
    if [ $use_wordlm = true ]; then
        recog_opts="--word-rnnlm ${lmexpdir}/rnnlm.model.best"
    else
        recog_opts="--rnnlm ${lmexpdir}/rnnlm.model.best"
    fi
    if [ $lm_weight == 0 ]; then
        recog_opts=""
    fi
    feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

    #### use CPU for decoding
    ngpu=0

    ${decode_cmd} JOB=25 ${expdir}/${decode_dir}/log/decode.JOB.log \
        asr_recog.py \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
        --result-label ${expdir}/${decode_dir}/data.JOB.json \
        --model ${expdir}/results/${recog_model}  \
        --beam-size ${beam_size} \
        --penalty ${penalty} \
        --maxlenratio ${maxlenratio} \
        --minlenratio ${minlenratio} \
        --ctc-weight ${ctc_weight} \
        --lm-weight ${lm_weight} \
        $recog_opts &

#    score_sclite.sh --wer true --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}
) &


fi


# test from here

