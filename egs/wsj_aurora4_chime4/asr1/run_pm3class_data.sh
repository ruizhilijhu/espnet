#!/bin/bash

# Copyright 2019 Johns Hopkins University (Ruizhi Li)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


# This script prepare bnf data and labels for PM3class training and train and decoding.
# 3 class: seen-indomain, unseen-indomain, unseen-outdomain

. ./path.sh
. ./cmd.sh

# general configuration
backend=pytorch
stage=0        # start from 0 if you need to start from data preparation
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
verbose=0      # verbose option
seed=1


########### ASR configs #############
asrdir=exp/wsj_train_si284-aurora4_train_si84_multi_pytorch_blstmp_e4_subsample1_1_1_1_1_unit320_proj320_d1_unit300_add_aconvc10_aconvf100_mtlalpha0.2_adadelta_sampprob0.0_bs15_mli800_mlo150_lsmunigram0.05
do_delta=false
dumpdir=dump

########### PM configs #############
# bnf components
bnf_component=enc
bnf_batchsize=5

############ decoding option for BNF and decoding #############
beam_size=30
recog_model=model.acc.best
penalty=0.0
minlenratio=0.0
maxlenratio=0.0
ctc_weight=0.3
lm_weight=0
lmtag=1layer_unit1000_sgd_bs300_word65000


. utils/parse_options.sh || exit 1;

. ./path.sh
. ./cmd.sh

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

######## data (asr) #######
asr_trn=wsj_train_si284-aurora4_train_si84_multi
asr_dev=wsj_test_dev93-aurora4_dev_0330

########## data (PM) ########
pm_trn=aurora4_train_si84_clean_trn-aurora4_train_si84_multi_trn-chime4_tr05_simu_isolated_1ch_track-chime4_tr05_real_isolated_1ch_track-wsj_test_dev93-aurora4_dev_0330

pm_dev=aurora4_train_si84_clean_cv-aurora4_train_si84_multi_cv-chime4_et05_simu_isolated_1ch_track-chime4_et05_real_isolated_1ch_track-wsj_test_eval92-aurora4_test_eval92

pm_trn_set="aurora4_train_si84_clean aurora4_train_si84_multi \
wsj_test_dev93 aurora4_dev_0330 \
chime4_tr05_simu_isolated_1ch_track chime4_tr05_real_isolated_1ch_track \
chime4_dt05_simu_isolated_1ch_track chime4_dt05_real_isolated_1ch_track"
pm_eval_set="wsj_test_eval92 \
chime4_et05_real_isolated_1ch_track chime4_et05_simu_isolated_1ch_track \
aurora4_test_eval92_street_wv1 aurora4_test_eval92_street_wv2 \
aurora4_test_eval92_airport_wv1 aurora4_test_eval92_airport_wv2 \
aurora4_test_eval92_train_wv1 aurora4_test_eval92_train_wv2 \
aurora4_test_eval92_restaurant_wv1 aurora4_test_eval92_restaurant_wv2 \
aurora4_test_eval92_babble_wv1 aurora4_test_eval92_babble_wv2 \
aurora4_test_eval92_car_wv1 aurora4_test_eval92_car_wv2 \
aurora4_test_eval92_clean_wv1 aurora4_test_eval92_clean_wv2 \
aurora4_train_si84_clean_cv aurora4_train_si84_multi_cv"
pm_extra_set=


### label SI:0, UI:1, UO:2 ####
# seen, in-domain
classSI="aurora4_train_si84_clean_trn aurora4_train_si84_multi_trn aurora4_train_si84_clean_cv aurora4_train_si84_multi_cv"
# unseen, in-domain
classUI="wsj_test_dev93 aurora4_dev_0330 wsj_test_eval92 aurora4_test_eval92 \
aurora4_test_eval92_street_wv1 aurora4_test_eval92_street_wv2 \
aurora4_test_eval92_airport_wv1 aurora4_test_eval92_airport_wv2 \
aurora4_test_eval92_train_wv1 aurora4_test_eval92_train_wv2 \
aurora4_test_eval92_restaurant_wv1 aurora4_test_eval92_restaurant_wv2 \
aurora4_test_eval92_babble_wv1 aurora4_test_eval92_babble_wv2 \
aurora4_test_eval92_car_wv1 aurora4_test_eval92_car_wv2 \
aurora4_test_eval92_clean_wv1 aurora4_test_eval92_clean_wv2"
# unseen, out-of-domain
classUO="chime4_tr05_simu_isolated_1ch_track chime4_tr05_real_isolated_1ch_track \
chime4_dt05_real_isolated_1ch_track chime4_dt05_simu_isolated_1ch_track \
chime4_et05_real_isolated_1ch_track chime4_et05_simu_isolated_1ch_track"


########### set up dict #########
dict=data/lang_1char/${asr_trn}_units.txt
nlsyms=data/lang_1char/non_lang_syms.txt

########### set up decode options #########
decode_id=beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}_rnnlm${lm_weight}_${lmtag}

########### set up dump dir #########
dumpdir=${dumpdir}
pmdumpdir=${asrdir}/dump_bnf${bnf_component}_${decode_id}

if [ ${stage} -le 1 ]; then
    echo "stage 1: BNF(ENCODER) Feature Extraction -- ${bnf_component}"
    nj=32
    tasks="${pm_trn_set} ${pm_eval_set} ${pm_extra_set}"
    for task in ${tasks}; do
    (
        # encoder features is not related to any decoder params
        featdir=${dumpdir}/$asr_trn/${task}/delta${do_delta} # in
        bnfdir=${asrdir}/bnf${bnf_component}_${decode_id}/${task} # out

        # split data
        splitjson.py --parts ${nj} ${featdir}/data.json

        #### use CPU for bnf extraction
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${bnfdir}/log/bnf.JOB.log \
            asr_bnf.py \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize ${bnf_batchsize} \
            --feat-json ${featdir}/split${nj}utt/data.JOB.json \
            --out ${bnfdir}/feats.JOB \
            --model ${asrdir}/results/${recog_model} \
            --bnf-component ${bnf_component} &
        wait

        [ -e ${bnfdir}/feats.scp ] && rm ${bnfdir}/feats.scp
        cat ${bnfdir}/feats.*.scp > ${bnfdir}/feats.scp || exit 1

        for file in text utt2spk wav.scp spk2utt; do
            cp data/${task}/${file} ${bnfdir}
        done

    ) &
    done
    wait
    echo "Finished: BNF -- ${bnf_component} extraction"
fi

if [ ${stage} -le 2 ]; then
    echo "stage 2: Label preparation"

    # split aurora4_train_si84_clean aurora4_train_si84_multi
    for task in aurora4_train_si84_clean aurora4_train_si84_multi; do
        bnfdir=${asrdir}/bnf${bnf_component}_${decode_id}/${task}
        echo -e "01u\n020\n02e\n40k" > ${bnfdir}/speakers_cv
        utils/subset_data_dir_tr_cv.sh --cv-spk-list ${bnfdir}/speakers_cv \
        ${bnfdir} ${bnfdir}_trn ${bnfdir}_cv
    done

    # Seen in-domain: label 0
    for task in $classSI; do
        bnfdir=${asrdir}/bnf${bnf_component}_${decode_id}/${task} # out
        awk '{print $1, 0}' $bnfdir/feats.scp > $bnfdir/utt23class
    done
    # unseen in-domain: label 1
    for task in $classUI; do
        bnfdir=${asrdir}/bnf${bnf_component}_${decode_id}/${task} # out
        if [ $task == 'aurora4_test_eval92' ];then
            aurora4_conds="airport_wv1 airport_wv2 babble_wv1 babble_wv2 car_wv1 car_wv2 clean_wv1 clean_wv2 restaurant_wv1 restaurant_wv2 street_wv1 street_wv2 train_wv1 train_wv2"
            for c in $aurora4_conds;do
                aurora4_strings+="${asrdir}/bnf${bnf_component}_${decode_id}/aurora4_test_eval92_$c "
            done
            utils/combine_data.sh --extra-files "utt2cer utt2csdiseqcer utt2wer utt2csdiseqwer" $bnfdir $aurora4_strings
        fi

        awk '{print $1, 1}' $bnfdir/feats.scp > $bnfdir/utt23class
    done
    # unseen out-of-domain: label 2
    for task in $classUO; do
        bnfdir=${asrdir}/bnf${bnf_component}_${decode_id}/${task} # out
        awk '{print $1, 2}' $bnfdir/feats.scp > $bnfdir/utt23class
    done
    echo "Finished: label preparation"
fi



pm_trn_dir=${asrdir}/bnf${bnf_component}_${decode_id}/${pm_trn} # data
pm_dev_dir=${asrdir}/bnf${bnf_component}_${decode_id}/${pm_dev} # data
pm_feat_trn_dir=${pmdumpdir}/${pm_trn}/${pm_trn}/delta${do_delta}; mkdir -p ${pm_feat_trn_dir} # dump data
pm_feat_dev_dir=${pmdumpdir}/${pm_trn}/${pm_dev}/delta${do_delta}; mkdir -p ${pm_feat_dev_dir} # dump data
if [ ${stage} -le 3 ]; then
    echo "stage 3: Dump Features (PM)"

    echo "combine data from WSJ aurora4 chime4"
    utils/combine_data.sh \
        --extra-files "utt2wer utt2csdiseqwer utt2cer utt2csdiseqcer utt23class" \
        ${pm_trn_dir} \
        ${asrdir}/bnf${bnf_component}_${decode_id}/aurora4_train_si84_clean_trn \
        ${asrdir}/bnf${bnf_component}_${decode_id}/aurora4_train_si84_multi_trn \
        ${asrdir}/bnf${bnf_component}_${decode_id}/chime4_tr05_simu_isolated_1ch_track \
        ${asrdir}/bnf${bnf_component}_${decode_id}/chime4_tr05_real_isolated_1ch_track \
        ${asrdir}/bnf${bnf_component}_${decode_id}/wsj_test_dev93 \
        ${asrdir}/bnf${bnf_component}_${decode_id}/aurora4_dev_0330

    utils/combine_data.sh \
        --extra-files "utt2wer utt2csdiseqwer utt2cer utt2csdiseqcer utt23class" \
        ${pm_dev_dir} \
        ${asrdir}/bnf${bnf_component}_${decode_id}/aurora4_train_si84_clean_cv \
        ${asrdir}/bnf${bnf_component}_${decode_id}/aurora4_train_si84_multi_cv \
        ${asrdir}/bnf${bnf_component}_${decode_id}/chime4_et05_simu_isolated_1ch_track \
        ${asrdir}/bnf${bnf_component}_${decode_id}/chime4_et05_real_isolated_1ch_track \
        ${asrdir}/bnf${bnf_component}_${decode_id}/wsj_test_eval92 \
        ${asrdir}/bnf${bnf_component}_${decode_id}/aurora4_test_eval92

    # compute global CMVN
    compute-cmvn-stats scp:${pm_trn_dir}/feats.scp ${pm_trn_dir}/cmvn.ark

    # dump features for training
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${pm_feat_trn_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{14,15,16}/${USER}/espnet-data/egs/wsj_aurora4_chime4/asr1/${pm_feat_trn_dir}/storage \
        ${pm_feat_trn_dir}/storage
    fi
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${pm_feat_dev_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{14,15,16}/${USER}/espnet-data/egs/wsj_aurora4_chime4/asr1/${pm_feat_dev_dir}/storage \
        ${pm_feat_dev_dir}/storage
    fi

    dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
        ${pm_trn_dir}/feats.scp \
        ${pm_trn_dir}/cmvn.ark \
        ${pm_feat_trn_dir}/log ${pm_feat_trn_dir}
    dump.sh --cmd "$train_cmd" --nj 4 --do_delta $do_delta \
        ${pm_dev_dir}/feats.scp \
        ${pm_trn_dir}/cmvn.ark \
        ${pm_feat_dev_dir}/log ${pm_feat_dev_dir}
    for rtask in ${pm_eval_set}; do
        pm_eval_dir=${asrdir}/bnf${bnf_component}_${decode_id}/${rtask} # data
        pm_feat_eval_dir=${pmdumpdir}/${pm_trn}/${rtask}/delta${do_delta}; mkdir -p ${pm_feat_eval_dir} # dump data
        dump.sh --cmd "$train_cmd" --nj 4 --do_delta $do_delta \
            ${pm_eval_dir}/feats.scp \
            ${pm_trn_dir}/cmvn.ark \
            ${pm_feat_eval_dir}/log ${pm_feat_eval_dir}
    done
fi


if [ ${stage} -le 4 ]; then
    echo "stage 4: PM Json Data Preparation for EER and 3class"
    data2json_err.sh --feat ${pm_feat_trn_dir}/feats.scp --nlsyms ${nlsyms} \
         ${pm_trn_dir} ${dict} > ${pm_feat_trn_dir}/data_err.json
    data2json_err.sh --feat ${pm_feat_dev_dir}/feats.scp --nlsyms ${nlsyms} \
         ${pm_dev_dir} ${dict} > ${pm_feat_dev_dir}/data_err.json
    for rtask in ${pm_eval_set}; do
        pm_eval_dir=${asrdir}/bnf${bnf_component}_${decode_id}/${rtask} # data
        pm_feat_eval_dir=${pmdumpdir}/${pm_trn}/${rtask}/delta${do_delta}; mkdir -p ${pm_feat_eval_dir} # dump data
        data2json_err.sh --feat ${pm_feat_eval_dir}/feats.scp \
            --nlsyms ${nlsyms} ${pm_eval_dir} ${dict} > ${pm_feat_eval_dir}/data_err.json
    done
fi


