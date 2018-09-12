#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import json
import logging
import numpy as np
import six
import sys


def end_detect(ended_hyps, i, M=3, D_end=np.log(1 * np.exp(-10))):
    '''End detection

    desribed in Eq. (50) of S. Watanabe et al
    "Hybrid CTC/Attention Architecture for End-to-End Speech Recognition"

    :param ended_hyps:
    :param i:
    :param M:
    :param D_end:
    :return:
    '''
    if len(ended_hyps) == 0:
        return False
    count = 0
    best_hyp = sorted(ended_hyps, key=lambda x: x['score'], reverse=True)[0]
    for m in six.moves.range(M):
        # get ended_hyps with their length is i - m
        hyp_length = i - m
        hyps_same_length = [x for x in ended_hyps if len(x['yseq']) == hyp_length]
        if len(hyps_same_length) > 0:
            best_hyp_same_length = sorted(hyps_same_length, key=lambda x: x['score'], reverse=True)[0]
            if best_hyp_same_length['score'] - best_hyp['score'] < D_end:
                count += 1

    if count == M:
        return True
    else:
        return False


# TODO(takaaki-hori): add different smoothing methods
def label_smoothing_dist(odim, lsm_type, transcript=None, blank=0):
    '''Obtain label distribution for loss smoothing

    :param odim:
    :param lsm_type:
    :param blank:
    :param transcript:
    :return:
    '''
    if transcript is not None:
        with open(transcript, 'rb') as f:
            trans_json = json.load(f)['utts']

    if lsm_type == 'unigram':
        assert transcript is not None, 'transcript is required for %s label smoothing' % lsm_type
        labelcount = np.zeros(odim)
        for k, v in trans_json.items():
            ids = np.array([int(n) for n in v['output'][0]['tokenid'].split()])
            # to avoid an error when there is no text in an uttrance
            if len(ids) > 0:
                labelcount[ids] += 1
        labelcount[odim - 1] = len(transcript)  # count <eos>
        labelcount[labelcount == 0] = 1  # flooring
        labelcount[blank] = 0  # remove counts for blank
        labeldist = labelcount.astype(np.float32) / np.sum(labelcount)
    else:
        logging.error(
            "Error: unexpected label smoothing type: %s" % lsm_type)
        sys.exit()

    return labeldist


# get output dim for latter BLSTM
def get_vgg2l_odim(idim, in_channel=3, out_channel=128):
    idim = idim / in_channel
    idim = np.ceil(np.array(idim, dtype=np.float32) / 2)  # 1st max pooling
    idim = np.ceil(np.array(idim, dtype=np.float32) / 2)  # 2nd max pooling
    return int(idim) * out_channel  # numer of channels


def get_maxpooling2_odim(idim, in_channel=3, out_channel=128, ceil_mode=False, mode='regular', dilation=1):

    idim = idim / in_channel
    fn = np.ceil if ceil_mode else np.floor
    if mode == 'regular':

        s, p, k = [1, dilation, 3]; idim = fn(((np.array(idim, dtype=np.float32) + 2 * p - dilation * (k-1)-1) / s) + 1) # in cnn
        s, p, k = [1, dilation, 3];
        idim = fn(((np.array(idim, dtype=np.float32) + 2 * p - dilation * (k - 1) - 1) / s) + 1)  # in cnn
        s, p, k = [2, 0, 2]; idim = fn(((np.array(idim, dtype=np.float32) + 2 * p - k) / s) + 1) # in maxpool
        s, p, k = [1, dilation, 3]; idim = fn(((np.array(idim, dtype=np.float32) + 2 * p - dilation * (k-1)-1) / s) + 1) # in cnn
        s, p, k = [1, dilation, 3];
        idim = fn(((np.array(idim, dtype=np.float32) + 2 * p - dilation * (k - 1) - 1) / s) + 1)  # in cnn
        s, p, k = [2, 0, 2]; idim = fn(((np.array(idim, dtype=np.float32) + 2 * p - k) / s) + 1) # in maxpool

    if mode == 'vgg8':

        s, p, k = [1, dilation, 3]; idim = fn(((np.array(idim, dtype=np.float32) + 2 * p - dilation * (k-1)-1) / s) + 1) # in cnn
        s, p, k = [1, dilation, 3];
        idim = fn(((np.array(idim, dtype=np.float32) + 2 * p - dilation * (k - 1) - 1) / s) + 1)  # in cnn
        s, p, k = [2, 0, 2]; idim = fn(((np.array(idim, dtype=np.float32) + 2 * p - k) / s) + 1) # in maxpool
        s, p, k = [1, dilation, 3]; idim = fn(((np.array(idim, dtype=np.float32) + 2 * p - dilation * (k-1)-1) / s) + 1) # in cnn
        s, p, k = [1, dilation, 3];
        idim = fn(((np.array(idim, dtype=np.float32) + 2 * p - dilation * (k - 1) - 1) / s) + 1)  # in cnn
        s, p, k = [2, 0, 2]; idim = fn(((np.array(idim, dtype=np.float32) + 2 * p - k) / s) + 1) # in maxpool
        s, p, k = [1, dilation, 3];
        idim = fn(((np.array(idim, dtype=np.float32) + 2 * p - dilation * (k - 1) - 1) / s) + 1)  # in cnn
        s, p, k = [1, dilation, 3];
        idim = fn(((np.array(idim, dtype=np.float32) + 2 * p - dilation * (k - 1) - 1) / s) + 1)  # in cnn
        s, p, k = [1, dilation, 3];
        idim = fn(((np.array(idim, dtype=np.float32) + 2 * p - dilation * (k - 1) - 1) / s) + 1)  # in cnn
        s, p, k = [1, dilation, 3];
        idim = fn(((np.array(idim, dtype=np.float32) + 2 * p - dilation * (k - 1) - 1) / s) + 1)  # in cnn

    elif mode =='resnetorig':
        s, p, k = [2, 3, 7]
        idim = fn(((np.array(idim, dtype=np.float32) + 2 * p - k) / s) + 1)
        s, p, k = [2, 1, 3]
        idim = fn(((np.array(idim, dtype=np.float32) + 2 * p - k) / s) + 1)

    return int(idim) * out_channel  # numer of channels


def get_RCNN_odim(idim, params, inplanes=1, planes=512):
    # in spectral axis
    idim = idim / inplanes
    idim = np.array(idim, dtype=np.float32)

    # params = [(1,2,3,4), (1,2,3,4)]
    for param in params:
        s, p, d, k = param # stride, padding, dilation, kernel_size
        idim = np.floor(((idim + 2 * p - d * (k - 1) - 1) / s) + 1)

    return int(idim) * planes  # numer of channels


def get_RCNN_ilens(ilens, params):

    # in time axis
    ilens = np.array(ilens, dtype=np.float32)

    # params = [(1,2,3,4), (1,2,3,4)]
    for param in params:
        s, p, d, k = param # stride, padding, dilation, kernel_size
        ilens = np.array(np.floor(((ilens + 2 * p - d * (k - 1) - 1) / s) + 1),dtype=np.int64)
    ilens = ilens.tolist()
    return ilens