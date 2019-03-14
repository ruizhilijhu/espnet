#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


from __future__ import division

import argparse
import logging
import math
import sys

from argparse import Namespace
import editdistance

import chainer
import numpy as np
import random
import six
import torch
import torch.nn.functional as F
import warpctc_pytorch as warp_ctc
import json

from chainer import reporter
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from espnet.nets.ctc_prefix_score import CTCPrefixScore
from espnet.nets.ctc_prefix_score import CTCPrefixScoreTH
from espnet.nets.e2e_asr_common import end_detect
from espnet.nets.e2e_asr_common import get_vgg2l_odim
from espnet.nets.e2e_asr_common import label_smoothing_dist


CTC_LOSS_THRESHOLD = 10000
CTC_SCORING_RATIO = 1.5
MAX_DECODER_OUTPUT = 5

# to be determined todo BEN
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


# ------------- Utility functions --------------------------------------------------------------------------------------
def to_cuda(m, x):
    """Function to send tensor into corresponding device

    :param torch.nn.Module m: torch module
    :param torch.Tensor x: torch tensor
    :return: torch tensor located in the same place as torch module
    :rtype: torch.Tensor
    """
    assert isinstance(m, torch.nn.Module)
    device = next(m.parameters()).device
    return x.to(device)


def pad_list(xs, pad_value):
    """Function to pad values

    :param list xs: list of torch.Tensor [(L_1, D), (L_2, D), ..., (L_B, D)]
    :param float pad_value: value for padding
    :return: padded tensor (B, Lmax, D)
    :rtype: torch.Tensor
    """
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, * xs[0].size()[1:]).fill_(pad_value)

    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]

    return pad

def unpad_list(xpads, xlens):
    """Function to pad values

    :param padded tensor (B, Lmax, D)
    :param tensor (B,)
    :return: list of torch.Tensor [(L_1, D), (L_2, D), ..., (L_B, D)]
    :rtype: list xs
    """

    xs = pack_padded_sequence(xpads, xlens, batch_first=True).data
    assert xs.size()[0] == sum(xlens)
    es = torch.cumsum(xlens, 0)
    ss = es - xlens

    xs = [xs[ss[i]:es[i],:] for i in range(len(xlens))]
    return xs

def make_pad_mask(lengths):
    """Function to make mask tensor containing indices of padded part

    e.g.: lengths = [5, 3, 2]
          mask = [[0, 0, 0, 0 ,0],
                  [0, 0, 0, 1, 1],
                  [0, 0, 1, 1, 1]]

    :param list lengths: list of lengths (B)
    :return: mask tensor containing indices of padded part (B, Tmax)
    :rtype: torch.Tensor
    """
    bs = int(len(lengths))
    maxlen = int(max(lengths))
    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    return seq_range_expand >= seq_length_expand


def mask_by_length(xs, length, fill=0):
    assert xs.size(0) == len(length)
    ret = xs.data.new(*xs.size()).fill_(fill)
    for i, l in enumerate(length):
        ret[i, :l] = xs[i, :l]
    return ret


def th_accuracy(pad_outputs, pad_targets, ignore_label):
    """Function to calculate accuracy

    :param torch.Tensor pad_outputs: prediction tensors (B*Lmax, D)
    :param torch.Tensor pad_targets: target tensors (B, Lmax, D)
    :param int ignore_label: ignore label id
    :retrun: accuracy value (0.0 - 1.0)
    :rtype: float
    """
    pad_pred = pad_outputs.view(
        pad_targets.size(0),
        pad_targets.size(1),
        pad_outputs.size(1)).argmax(2)
    mask = pad_targets != ignore_label
    numerator = torch.sum(pad_pred.masked_select(mask) == pad_targets.masked_select(mask))
    denominator = torch.sum(mask)
    return float(numerator) / float(denominator)


def get_last_yseq(exp_yseq):
    last = []
    for y_seq in exp_yseq:
        last.append(y_seq[-1])
    return last


def append_ids(yseq, ids):
    if isinstance(ids, list):
        for i, j in enumerate(ids):
            yseq[i].append(j)
    else:
        for i in range(len(yseq)):
            yseq[i].append(ids)
    return yseq

def append_states(yseq, states):
    for i, j in enumerate(states):
        yseq[i].append(j)
    return yseq


def expand_yseq(yseqs, next_ids):
    new_yseq = []
    for yseq in yseqs:
        for next_id in next_ids:
            new_yseq.append(yseq[:])
            new_yseq[-1].append(next_id)
    return new_yseq


def index_select_list(yseq, lst):
    new_yseq = []
    for l in lst:
        new_yseq.append(yseq[l][:])
    return new_yseq


def index_select_lm_state(rnnlm_state, dim, vidx):
    if isinstance(rnnlm_state, dict):
        new_state = {}
        for k, v in rnnlm_state.items():
            new_state[k] = [torch.index_select(vi, dim, vidx) for vi in v]
    elif isinstance(rnnlm_state, list):
        new_state = []
        for i in vidx:
            new_state.append(rnnlm_state[int(i)][:])
    return new_state


def process_subsample_string(elayers, etype, subsample_string):
    # +1 means input (+1) and layers outputs (args.elayer)
    subsample = np.ones(elayers + 1, dtype=np.int)
    if etype in ['blstmp', 'blstm']:
        ss = subsample_string.split("_")
        for j in range(min(elayers + 1, len(ss))):
            subsample[j] = int(ss[j])
    else:
        logging.warning(
            'Subsampling is not performed for this encoder type {}. If it is vgg*, it is performed in max pooling layers at CNN.'.format(etype))
    logging.info('subsample: ' + ' '.join([str(x) for x in subsample]))
    return subsample





def create_attention_module(eprojs, dunits, atype, adim, awin, aheads, aconv_chans, aconv_filts, store_hs=True):
    """
    create attention module
    """
    if atype == 'noatt':
        att = NoAtt()
    elif atype == 'dot':
        att = AttDot(eprojs, dunits, adim)
    elif atype == 'add':
        att = AttAdd(eprojs, dunits, adim, store_hs=store_hs)
    elif atype == 'location':
        att = AttLoc(eprojs, dunits, adim, aconv_chans, aconv_filts)
    elif atype == 'location2d':
        att = AttLoc2D(eprojs, dunits, adim, awin, aconv_chans, aconv_filts)
    elif atype == 'location_recurrent':
        att = AttLocRec(eprojs, dunits, adim, aconv_chans, aconv_filts)
    elif atype == 'coverage':
        att = AttCov(eprojs, dunits, adim)
    elif atype == 'coverage_location':
        att = AttCovLoc(eprojs, dunits, adim,aconv_chans, aconv_filts)
    elif atype == 'multi_head_dot':
        att = AttMultiHeadDot(eprojs, dunits,aheads, adim, adim)
    elif atype == 'multi_head_add':
        att = AttMultiHeadAdd(eprojs, dunits, aheads, adim, adim)
    elif atype == 'multi_head_loc':
        att = AttMultiHeadLoc(eprojs, dunits, aheads, adim, adim, aconv_chans, aconv_filts)
    elif atype == 'multi_head_multi_res_loc':
        att = AttMultiHeadMultiResLoc(eprojs, dunits, aheads, adim, adim, aconv_chans, aconv_filts)
    else:
        logging.error(
            "Error: need to specify an appropriate attention archtecture")
        sys.exit()
    return att

class Reporter(chainer.Chain):
    def report(self, loss_ctc, loss_att, acc, cer, wer, mtl_loss):
        reporter.report({'loss_ctc': loss_ctc}, self)
        reporter.report({'loss_att': loss_att}, self)
        reporter.report({'acc': acc}, self)
        reporter.report({'cer': cer}, self)
        reporter.report({'wer': wer}, self)
        logging.info('mtl loss:' + str(mtl_loss))
        reporter.report({'loss': mtl_loss}, self)

class ReporterMulEnc(chainer.Chain):
    def report(self, loss_ctcs, loss_att, acc, cer, wer, mtl_loss):
        # loss_ctcs = [weighted CTC, CTC1, CTC2, ... CTCN]
        num_enc = len(loss_ctcs) - 1
        reporter.report({'loss_ctc': loss_ctcs[0]}, self)
        for i in range(num_enc):
            reporter.report({'loss_ctc{}'.format(i+1): loss_ctcs[i+1]}, self)
        reporter.report({'loss_att': loss_att}, self)
        reporter.report({'acc': acc}, self)
        reporter.report({'cer': cer}, self)
        reporter.report({'wer': wer}, self)
        logging.info('mtl loss:' + str(mtl_loss))
        reporter.report({'loss': mtl_loss}, self)


# TODO(watanabe) merge Loss and E2E: there is no need to make these separately
class Loss(torch.nn.Module):
    """Multi-task learning loss module

    :param torch.nn.Module predictor: E2E model instance
    :param float mtlalpha: mtl coefficient value (0.0 ~ 1.0)
    """

    def __init__(self, predictor, mtlalpha):
        super(Loss, self).__init__()
        assert 0.0 <= mtlalpha <= 1.0, "mtlalpha shoule be [0.0, 1.0]"
        self.mtlalpha = mtlalpha
        self.loss = None
        self.accuracy = None
        self.predictor = predictor
        self.reporter = Reporter()

    def forward(self, xs_pad, ilens, ys_pad):
        '''Multi-task learning loss forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :return: loss value
        :rtype: torch.Tensor
        '''
        self.loss = None
        loss_ctc, loss_att, acc, cer, wer = self.predictor(xs_pad, ilens, ys_pad)
        alpha = self.mtlalpha
        if alpha == 0:
            self.loss = loss_att
            loss_att_data = float(loss_att)
            loss_ctc_data = None
        elif alpha == 1:
            self.loss = loss_ctc
            loss_att_data = None
            loss_ctc_data = float(loss_ctc)
        else:
            self.loss = alpha * loss_ctc + (1 - alpha) * loss_att
            loss_att_data = float(loss_att)
            loss_ctc_data = float(loss_ctc)

        loss_data = float(self.loss)
        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            self.reporter.report(loss_ctc_data, loss_att_data, acc, cer, wer, loss_data)
        else:
            logging.warning('loss (=%f) is not correct', loss_data)

        return self.loss


class E2E(torch.nn.Module):
    """E2E module

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param namespace args: argument namespace containing options
    """

    def __init__(self, idim, odim, args):
        super(E2E, self).__init__()
        self.etype = args.etype
        self.verbose = args.verbose
        self.char_list = args.char_list
        self.outdir = args.outdir
        self.mtlalpha = args.mtlalpha

        # below means the last number becomes eos/sos ID
        # note that sos/eos IDs are identical
        self.sos = odim - 1
        self.eos = odim - 1

        # subsample info
        # +1 means input (+1) and layers outputs (args.elayer)
        subsample = np.ones(args.elayers + 1, dtype=np.int)
        if args.etype == 'blstmp':
            ss = args.subsample.split("_")
            for j in range(min(args.elayers + 1, len(ss))):
                subsample[j] = int(ss[j])
        else:
            logging.warning(
                'Subsampling is not performed for vgg*. It is performed in max pooling layers at CNN.')
        logging.info('subsample: ' + ' '.join([str(x) for x in subsample]))
        self.subsample = subsample

        # label smoothing info
        if args.lsm_type:
            logging.info("Use label smoothing with " + args.lsm_type)
            labeldist = label_smoothing_dist(odim, args.lsm_type, transcript=args.train_json)
        else:
            labeldist = None

        # encoder
        self.enc = Encoder(args.etype, idim, args.elayers, args.eunits, args.eprojs,
                           self.subsample, args.dropout_rate)
        # ctc
        self.ctc = CTC(odim, args.eprojs, args.dropout_rate)
        # attention
        if args.atype == 'noatt':
            self.att = NoAtt()
        elif args.atype == 'dot':
            self.att = AttDot(args.eprojs, args.dunits, args.adim)
        elif args.atype == 'add':
            self.att = AttAdd(args.eprojs, args.dunits, args.adim)
        elif args.atype == 'location':
            self.att = AttLoc(args.eprojs, args.dunits,
                              args.adim, args.aconv_chans, args.aconv_filts)
        elif args.atype == 'location2d':
            self.att = AttLoc2D(args.eprojs, args.dunits,
                                args.adim, args.awin, args.aconv_chans, args.aconv_filts)
        elif args.atype == 'location_recurrent':
            self.att = AttLocRec(args.eprojs, args.dunits,
                                 args.adim, args.aconv_chans, args.aconv_filts)
        elif args.atype == 'coverage':
            self.att = AttCov(args.eprojs, args.dunits, args.adim)
        elif args.atype == 'coverage_location':
            self.att = AttCovLoc(args.eprojs, args.dunits, args.adim,
                                 args.aconv_chans, args.aconv_filts)
        elif args.atype == 'multi_head_dot':
            self.att = AttMultiHeadDot(args.eprojs, args.dunits,
                                       args.aheads, args.adim, args.adim)
        elif args.atype == 'multi_head_add':
            self.att = AttMultiHeadAdd(args.eprojs, args.dunits,
                                       args.aheads, args.adim, args.adim)
        elif args.atype == 'multi_head_loc':
            self.att = AttMultiHeadLoc(args.eprojs, args.dunits,
                                       args.aheads, args.adim, args.adim,
                                       args.aconv_chans, args.aconv_filts)
        elif args.atype == 'multi_head_multi_res_loc':
            self.att = AttMultiHeadMultiResLoc(args.eprojs, args.dunits,
                                               args.aheads, args.adim, args.adim,
                                               args.aconv_chans, args.aconv_filts)
        else:
            logging.error(
                "Error: need to specify an appropriate attention archtecture")
            sys.exit()
        # decoder
        self.dec = Decoder(args.eprojs, odim, args.dlayers, args.dunits,
                           self.sos, self.eos, self.att, self.verbose, self.char_list,
                           labeldist, args.lsm_weight, args.sampling_probability)

        # weight initialization
        self.init_like_chainer()

        # options for beam search
        if 'report_cer' in vars(args) and (args.report_cer or args.report_wer):
            recog_args = {'beam_size': args.beam_size, 'penalty': args.penalty,
                          'ctc_weight': args.ctc_weight, 'maxlenratio': args.maxlenratio,
                          'minlenratio': args.minlenratio, 'lm_weight': args.lm_weight,
                          'rnnlm': args.rnnlm, 'nbest': args.nbest,
                          'space': args.sym_space, 'blank': args.sym_blank}

            self.recog_args = argparse.Namespace(**recog_args)
            self.report_cer = args.report_cer
            self.report_wer = args.report_wer
        else:
            self.report_cer = False
            self.report_wer = False
        self.rnnlm = None

        self.logzero = -10000000000.0

    def init_like_chainer(self):
        """Initialize weight like chainer

        chainer basically uses LeCun way: W ~ Normal(0, fan_in ** -0.5), b = 0
        pytorch basically uses W, b ~ Uniform(-fan_in**-0.5, fan_in**-0.5)

        however, there are two exceptions as far as I know.
        - EmbedID.W ~ Normal(0, 1)
        - LSTM.upward.b[forget_gate_range] = 1 (but not used in NStepLSTM)
        """
        def lecun_normal_init_parameters(module):
            for p in module.parameters():
                data = p.data
                if data.dim() == 1:
                    # bias
                    data.zero_()
                elif data.dim() == 2:
                    # linear weight
                    n = data.size(1)
                    stdv = 1. / math.sqrt(n)
                    data.normal_(0, stdv)
                elif data.dim() == 4:
                    # conv weight
                    n = data.size(1)
                    for k in data.size()[2:]:
                        n *= k
                    stdv = 1. / math.sqrt(n)
                    data.normal_(0, stdv)
                else:
                    raise NotImplementedError

        def set_forget_bias_to_one(bias):
            n = bias.size(0)
            start, end = n // 4, n // 2
            bias.data[start:end].fill_(1.)

        lecun_normal_init_parameters(self)
        # exceptions
        # embed weight ~ Normal(0, 1)
        self.dec.embed.weight.data.normal_(0, 1)
        # forget-bias = 1.0
        # https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745
        for l in six.moves.range(len(self.dec.decoder)):
            set_forget_bias_to_one(self.dec.decoder[l].bias_ih)

    def forward(self, xs_pad, ilens, ys_pad):
        '''E2E forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :return: ctc loass value
        :rtype: torch.Tensor
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy in attention decoder
        :rtype: float
        '''
        # 1. encoder
        hs_pad, hlens = self.enc(xs_pad, ilens)

        # 2. CTC loss
        if self.mtlalpha == 0:
            loss_ctc = None
        else:
            loss_ctc = self.ctc(hs_pad, hlens, ys_pad)

        # 3. attention loss
        if self.mtlalpha == 1:
            loss_att = None
            acc = None
        else:
            loss_att, acc = self.dec(hs_pad, hlens, ys_pad)

        # 5. compute cer/wer
        if self.training or not (self.report_cer or self.report_wer):
            cer, wer = 0.0, 0.0
            # oracle_cer, oracle_wer = 0.0, 0.0
        else:
            if self.recog_args.ctc_weight > 0.0:
                lpz = self.ctc.log_softmax(hs_pad).data
            else:
                lpz = None

            wers, cers = [], []
            nbest_hyps = self.dec.recognize_beam_batch(hs_pad, torch.tensor(hlens), lpz,
                                                       self.recog_args, self.char_list,
                                                       self.rnnlm)
            # remove <sos> and <eos>
            y_hats = [nbest_hyp[0]['yseq'][1:-1] for nbest_hyp in nbest_hyps]
            for i, y_hat in enumerate(y_hats):
                y_true = ys_pad[i]

                seq_hat = [self.char_list[int(idx)] for idx in y_hat if int(idx) != -1]
                seq_true = [self.char_list[int(idx)] for idx in y_true if int(idx) != -1]
                seq_hat_text = "".join(seq_hat).replace(self.recog_args.space, ' ')
                seq_hat_text = seq_hat_text.replace(self.recog_args.blank, '')
                seq_true_text = "".join(seq_true).replace(self.recog_args.space, ' ')

                hyp_words = seq_hat_text.split()
                ref_words = seq_true_text.split()
                wers.append(editdistance.eval(hyp_words, ref_words) / len(ref_words))
                hyp_chars = seq_hat_text.replace(' ', '')
                ref_chars = seq_true_text.replace(' ', '')
                cers.append(editdistance.eval(hyp_chars, ref_chars) / len(ref_chars))

            wer = 0.0 if not self.report_wer else sum(wers) / len(wers)
            cer = 0.0 if not self.report_cer else sum(cers) / len(cers)

        return loss_ctc, loss_att, acc, cer, wer

    def recognize(self, x, recog_args, char_list, rnnlm=None):
        '''E2E beam search

        :param ndarray x: input acouctic feature (T, D)
        :param namespace recog_args: argment namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        '''
        prev = self.training
        self.eval()
        # subsample frame
        x = x[::self.subsample[0], :]
        ilen = [x.shape[0]]
        h = to_cuda(self, torch.from_numpy(
            np.array(x, dtype=np.float32)))

        # 1. encoder
        # make a utt list (1) to use the same interface for encoder
        h = h.contiguous()
        h, _ = self.enc(h.unsqueeze(0), ilen)

        # calculate log P(z_t|X) for CTC scores
        if recog_args.ctc_weight > 0.0:
            lpz = self.ctc.log_softmax(h)[0]
        else:
            lpz = None

        # 2. decoder
        # decode the first utterance
        y = self.dec.recognize_beam(h[0], lpz, recog_args, char_list, rnnlm)

        if prev:
            self.train()
        return y

    def recognize_batch(self, xs, recog_args, char_list, rnnlm=None):
        '''E2E beam search

        :param ndarray x: input acouctic feature (T, D)
        :param namespace recog_args: argment namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        '''
        prev = self.training
        self.eval()
        # subsample frame
        xs = [xx[::self.subsample[0], :] for xx in xs]
        ilens = np.fromiter((xx.shape[0] for xx in xs), dtype=np.int64)
        hs = [to_cuda(self, torch.from_numpy(np.array(xx, dtype=np.float32)))
              for xx in xs]

        # 1. encoder
        xpad = pad_list(hs, 0.0)
        hpad, hlens = self.enc(xpad, ilens)

        # calculate log P(z_t|X) for CTC scores
        if recog_args.ctc_weight > 0.0:
            lpz = self.ctc.log_softmax(hpad)
        else:
            lpz = None

        # 2. decoder
        y = self.dec.recognize_beam_batch(hpad, hlens, lpz, recog_args, char_list, rnnlm)

        if prev:
            self.train()
        return y

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad):
        '''E2E attention calculation

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :return: attention weights with the following shape,
            1) multi-head case => attention weights (B, H, Lmax, Tmax),
            2) other case => attention weights (B, Lmax, Tmax).
        :rtype: float ndarray
        '''
        with torch.no_grad():
            # encoder
            hpad, hlens = self.enc(xs_pad, ilens)

            # decoder
            att_ws = self.dec.calculate_all_attentions(hpad, hlens, ys_pad)

        return att_ws

    def bnf_enc(self, xs, args):
        '''E2E extract bottleneck features (encoder output)

        :param list of ndarray x: list of input acoustic feature (T, D)
        :param namespace args: argument namespace containing options
        :return: bottleneck features
        :rtype: ndarray
        '''
        prev = self.training
        self.eval()
        # subsample frame
        xs = [xx[::self.subsample[0], :] for xx in xs]
        ilens = np.fromiter((xx.shape[0] for xx in xs), dtype=np.int64)
        hs = [to_cuda(self, torch.from_numpy(np.array(xx, dtype=np.float32)))
              for xx in xs]

        # 1. encoder
        xpad = pad_list(hs, 0.0)

        hpad, hlens = self.enc(xpad, ilens)

        bnfs = unpad_list(hpad.cpu(), hlens.cpu())
        bnfs = [bnf.numpy() for bnf in bnfs]

        if prev:
            self.train()

        return bnfs

    def bnf_ctcpresm(self, xs, recog_args):
        '''E2E extract bottleneck features (ctc presoftmax)

        :param list of ndarray x: list of input acoustic feature list of (T, D)
        :param namespace args: argument namespace containing options
        :return: bottleneck features
        :rtype: ndarray
        '''
        prev = self.training
        self.eval()
        # subsample frame
        xs = [xx[::self.subsample[0], :] for xx in xs]
        ilens = np.fromiter((xx.shape[0] for xx in xs), dtype=np.int64)
        hs = [to_cuda(self, torch.from_numpy(np.array(xx, dtype=np.float32)))
              for xx in xs]

        # 1. encoder
        xpad = pad_list(hs, 0.0)
        hpad, hlens = self.enc(xpad, ilens)

        # calculate log P(z_t|X) for CTC scores
        lpz = self.ctc.log_softmax(hpad)

        bnfs = unpad_list(lpz.cpu(), hlens.cpu())
        bnfs = [bnf.numpy() for bnf in bnfs]

        if prev:
            self.train()

        return bnfs

    def bnf_dec(self, xs, recog_args, char_list, rnnlm=None):
        '''E2E extract bottleneck features (decoder related features)

        :param list of ndarray x: list of input acoustic feature list of (T, D)
        :param namespace args: argument namespace containing options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: bottleneck features
        :return: N-best decoding results
        :rtype: ndarray, list
        '''
        prev = self.training
        self.eval()
        # subsample frame
        xs = [xx[::self.subsample[0], :] for xx in xs]
        ilens = np.fromiter((xx.shape[0] for xx in xs), dtype=np.int64)
        hs = [to_cuda(self, torch.from_numpy(np.array(xx, dtype=np.float32)))
              for xx in xs]

        # 1. encoder
        xpad = pad_list(hs, 0.0)
        hpad, hlens = self.enc(xpad, ilens)

        # calculate log P(z_t|X) for CTC scores
        if recog_args.ctc_weight > 0.0:
            lpz = self.ctc.log_softmax(hpad)
        else:
            lpz = None

        # 2. decoder
        y = self.dec.bnf_dec(hpad, hlens, lpz, recog_args, char_list, rnnlm)

        bnfs = []

        for i in range(len(xs)):
            # get only the BEST path
            bnfs.append(y[i][0]['bnf'])

            # delete the numpy in the best paths
            for j in range(recog_args.nbest): del y[i][0]['bnf']

        if prev:
            self.train()

        return bnfs, y


class Loss_MulEnc(torch.nn.Module):
    """Multi-task learning loss module

    :param torch.nn.Module predictor: E2E_MulEnc model instance
    :param float mtlalpha: mtl coefficient value (0.0 ~ 1.0)
    """

    def __init__(self, predictor, mtlalpha, ctc_ws):
        super(Loss_MulEnc, self).__init__()
        assert 0.0 <= mtlalpha <= 1.0, "mtlalpha shoule be [0.0, 1.0]"
        self.mtlalpha = mtlalpha
        self.ctc_ws = ctc_ws
        self.loss = None
        self.accuracy = None
        self.predictor = predictor
        self.reporter = ReporterMulEnc()

    def forward(self, xs_pad, ilens, ys_pad):
        '''Multi-task learning loss forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :return: loss value
        :rtype: torch.Tensor
        '''
        self.loss = None
        loss_ctc_list, loss_att, acc, cer, wer = self.predictor(xs_pad, ilens, ys_pad)
        alpha = self.mtlalpha
        ctc_ws = self.ctc_ws

        assert len(loss_ctc_list) == len(ctc_ws)

        if alpha == 0:
            self.loss = loss_att
            loss_att_data = float(loss_att)
            loss_ctcs_data = [None] * (len(ctc_ws) + 1)
        elif alpha == 1:
            self.loss = torch.sum([item * ctc_ws[i] for i, item in enumerate(loss_ctc_list)])
            loss_att_data = None
            loss_ctcs_data = [float(self.loss)] + [float(item) for item in loss_ctc_list]
        else:
            loss_ctc = torch.sum(torch.cat([item * ctc_ws[i] for i, item in enumerate(loss_ctc_list)]))
            self.loss = alpha * loss_ctc + (1 - alpha) * loss_att
            loss_att_data = float(loss_att)
            loss_ctcs_data = [float(loss_ctc)] + [float(item) for item in loss_ctc_list]

        loss_data = float(self.loss)
        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            self.reporter.report(loss_ctcs_data, loss_att_data, acc, cer, wer, loss_data)
        else:
            logging.warning('loss (=%f) is not correct', loss_data)

        return self.loss


class E2E_MulEnc(torch.nn.Module):
    """E2E_MulEnc module

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param namespace args: argument namespace containing options
    """

    def __init__(self, idim, odim, args):
        super(E2E_MulEnc, self).__init__()
        self.verbose = args.verbose
        self.char_list = args.char_list
        self.outdir = args.outdir
        self.mtlalpha = args.mtlalpha
        self.num_enc = args.num_enc
        self.sharectc = args.sharectc

        # below means the last number becomes eos/sos ID
        # note that sos/eos IDs are identical
        self.sos = odim - 1
        self.eos = odim - 1

        # label smoothing info
        if args.lsm_type:
            logging.info("Use label smoothing with " + args.lsm_type)
            labeldist = label_smoothing_dist(odim, args.lsm_type, transcript=args.train_json)
        else:
            labeldist = None

        # read multi-encoder configuration file
        with open(args.conf_file, "rb") as f:
            logging.info('reading a multi-encoder config file from ' + args.conf_file)
            model_dict = json.load(f)
        assert self.num_enc <= model_dict['encoders']
        encs_eprojs = [model_dict['encoders'][i]['eprojs'] for i in six.moves.range(self.num_enc)]
        assert len(set(encs_eprojs)) == 1, 'eprojs needs to be the same for all encoders due to encoder-attention mechanism'
        self.subsample = []

        # share ctc
        if self.sharectc:
            encs_dropout_rate = [model_dict['encoders'][i]['dropout_rate'] for i in six.moves.range(self.num_enc)]
            # dropout_rate from each enc should be the same
            assert len(set(encs_dropout_rate))  == 1
            self.ctc = CTC(odim, encs_eprojs[0], encs_dropout_rate[0])

        # ctc_ws_trn and ctc_ws_dec
        ctc_ws_trn = np.array([float(item) for item in args.ctc_ws_trn.split("_")])
        ctc_ws_dec = np.array([float(item) for item in args.ctc_ws_dec.split("_")])
        self.ctc_ws_trn = ctc_ws_trn / np.sum(ctc_ws_trn) # normalize
        self.ctc_ws_dec = ctc_ws_dec / np.sum(ctc_ws_dec) # normalize
        self.ctc_ws_trn = [float(item) for item in ctc_ws_trn]
        self.ctc_ws_dec = [float(item) for item in ctc_ws_dec]
        logging.info('ctc weights (training during training): ' + ' '.join([str(x) for x in self.ctc_ws_trn]))
        logging.info('ctc weights (decoding during training): ' + ' '.join([str(x) for x in self.ctc_ws_dec]))

        # decoder config
        dlayers = model_dict['decoder']['dlayers']
        dunits = model_dict['decoder']['dunits']

        # encoder
        for i in six.moves.range(self.num_enc):
            # load config
            etype = model_dict['encoders'][i]['etype']
            elayers = model_dict['encoders'][i]['elayers']
            eunits = model_dict['encoders'][i]['eunits']
            eprojs = model_dict['encoders'][i]['eprojs']
            dropout_rate = model_dict['encoders'][i]['dropout_rate']
            atype = model_dict['encoders'][i]['atype']
            adim = model_dict['encoders'][i]['adim']
            awin = model_dict['encoders'][i]['awin']
            aheads = model_dict['encoders'][i]['aheads']
            aconv_chans = model_dict['encoders'][i]['aconv_chans']
            aconv_filts = model_dict['encoders'][i]['aconv_filts']

            # load subsample
            subsample = process_subsample_string(elayers, etype, model_dict['encoders'][i]['subsample'])
            self.subsample.append(subsample)

            # set up encoder
            setattr(self, "enc{}".format(i), Encoder(etype, idim, elayers, eunits, eprojs,
                           subsample, dropout_rate))

        # ctc
        for i in six.moves.range(self.num_enc):
            # load config
            eprojs = model_dict['encoders'][i]['eprojs']
            dropout_rate = model_dict['encoders'][i]['dropout_rate']

            # set up ctc
            if not self.sharectc:
                # dropout is applied to the encoder outout
                setattr(self, "ctc{}".format(i), CTC(odim, eprojs, dropout_rate))

        # attention
        self.att_list = torch.nn.ModuleList()
        for i in six.moves.range(self.num_enc):
            # load config
            eprojs = model_dict['encoders'][i]['eprojs']
            atype = model_dict['encoders'][i]['atype']
            adim = model_dict['encoders'][i]['adim']
            awin = model_dict['encoders'][i]['awin']
            aheads = model_dict['encoders'][i]['aheads']
            aconv_chans = model_dict['encoders'][i]['aconv_chans']
            aconv_filts = model_dict['encoders'][i]['aconv_filts']

            # set up attention
            self.att_list.append(create_attention_module(eprojs, dunits, atype, adim, awin, aheads, aconv_chans, aconv_filts))

        # subsapmle
        encs_subsample_1 = [self.subsample[i][0] for i in six.moves.range(self.num_enc)]
        assert len(set(encs_subsample_1)) == 1, 'first subsample factor needs to be the same for all encoders'
        # encoder attention
        atype = model_dict['encatt']['atype']
        adim = model_dict['encatt']['adim']
        awin = model_dict['encatt']['awin']
        aheads = model_dict['encatt']['aheads']
        aconv_chans = model_dict['encatt']['aconv_chans']
        aconv_filts = model_dict['encatt']['aconv_filts']
        self.encatt = create_attention_module(encs_eprojs[0], dunits, atype, adim, awin, aheads, aconv_chans, aconv_filts, store_hs=False)

        # decoder
        self.dec = Decoder_MulEnc(encs_eprojs[0], odim, dlayers, dunits,
                           self.sos, self.eos, self.encatt, self.att_list, self.verbose, self.char_list,
                           labeldist, args.lsm_weight, args.sampling_probability)

        # weight initialization
        self.init_like_chainer()

        # options for beam search
        if 'report_cer' in vars(args) and (args.report_cer or args.report_wer):
            recog_args = {'beam_size': args.beam_size, 'penalty': args.penalty,
                          'ctc_weight': args.ctc_weight, 'maxlenratio': args.maxlenratio,
                          'minlenratio': args.minlenratio, 'lm_weight': args.lm_weight,
                          'rnnlm': args.rnnlm, 'nbest': args.nbest,
                          'space': args.sym_space, 'blank': args.sym_blank, 'ctc_ws': self.ctc_ws_dec}

            self.recog_args = argparse.Namespace(**recog_args)
            self.report_cer = args.report_cer
            self.report_wer = args.report_wer
        else:
            self.report_cer = False
            self.report_wer = False
        self.rnnlm = None

        self.logzero = -10000000000.0

    def init_like_chainer(self):
        """Initialize weight like chainer

        chainer basically uses LeCun way: W ~ Normal(0, fan_in ** -0.5), b = 0
        pytorch basically uses W, b ~ Uniform(-fan_in**-0.5, fan_in**-0.5)

        however, there are two exceptions as far as I know.
        - EmbedID.W ~ Normal(0, 1)
        - LSTM.upward.b[forget_gate_range] = 1 (but not used in NStepLSTM)
        """
        def lecun_normal_init_parameters(module):
            for name, p in module.named_parameters():
                data = p.data
                if data.dim() == 1:
                    # bias
                    data.zero_()
                elif data.dim() == 2:
                    # linear weight
                    n = data.size(1)
                    stdv = 1. / math.sqrt(n)
                    data.normal_(0, stdv)
                elif data.dim() == 4:
                    # conv weight
                    n = data.size(1)
                    for k in data.size()[2:]:
                        n *= k
                    stdv = 1. / math.sqrt(n)
                    data.normal_(0, stdv)
                else:
                    raise NotImplementedError

        def set_forget_bias_to_one(bias):
            n = bias.size(0)
            start, end = n // 4, n // 2
            bias.data[start:end].fill_(1.)

        lecun_normal_init_parameters(self)
        # exceptions
        # embed weight ~ Normal(0, 1)
        self.dec.embed.weight.data.normal_(0, 1)
        # forget-bias = 1.0
        # https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745
        for l in six.moves.range(len(self.dec.decoder)):
            set_forget_bias_to_one(self.dec.decoder[l].bias_ih)

    def forward(self, xs_pad, ilens, ys_pad):
        '''E2E_MulEnc forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :return: ctc loass value
        :rtype: torch.Tensor
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy in attention decoder
        :rtype: float
        '''

        hs_pad_list = []
        hlens_list = []
        loss_ctc_list = []
        for i in six.moves.range(self.num_enc):
            # 1. encoder
            hs_pad, hlens = getattr(self, "enc{}".format(i))(xs_pad, ilens)

            # 2. CTC loss
            if self.mtlalpha == 0:
                loss_ctc_list.append(None)
            elif self.sharectc:
                loss_ctc_list.append(self.ctc(hs_pad, hlens, ys_pad))
            else:
                loss_ctc_list.append(getattr(self, "ctc{}".format(i))(hs_pad, hlens, ys_pad))

            hs_pad_list.append(hs_pad)
            hlens_list.append(hlens)

        # 3. attention loss
        if self.mtlalpha == 1:
            loss_att = None
            acc = None
        else:
            loss_att, acc = self.dec(hs_pad_list, hlens_list, ys_pad)

        # 4. compute cer/wer
        if self.training or not (self.report_cer or self.report_wer):
            cer, wer = 0.0, 0.0
            # oracle_cer, oracle_wer = 0.0, 0.0
        else:
            if self.recog_args.ctc_weight > 0.0:
                lpz_list = [getattr(self, "ctc{}".format(i)).log_softmax(hs_pad).data for i, hs_pad in enumerate(hs_pad_list)]

            else:
                lpz_list = [None for _ in hs_pad_list]

            wers, cers = [], []
            nbest_hyps = self.dec.recognize_beam_batch(hs_pad_list, hlens_list, lpz_list,
                                                       self.recog_args, self.char_list,
                                                       self.rnnlm)
            # remove <sos> and <eos>
            y_hats = [nbest_hyp[0]['yseq'][1:-1] for nbest_hyp in nbest_hyps]
            for i, y_hat in enumerate(y_hats):
                y_true = ys_pad[i]

                seq_hat = [self.char_list[int(idx)] for idx in y_hat if int(idx) != -1]
                seq_true = [self.char_list[int(idx)] for idx in y_true if int(idx) != -1]
                seq_hat_text = "".join(seq_hat).replace(self.recog_args.space, ' ')
                seq_hat_text = seq_hat_text.replace(self.recog_args.blank, '')
                seq_true_text = "".join(seq_true).replace(self.recog_args.space, ' ')

                hyp_words = seq_hat_text.split()
                ref_words = seq_true_text.split()
                wers.append(editdistance.eval(hyp_words, ref_words) / len(ref_words))
                hyp_chars = seq_hat_text.replace(' ', '')
                ref_chars = seq_true_text.replace(' ', '')
                cers.append(editdistance.eval(hyp_chars, ref_chars) / len(ref_chars))

            wer = 0.0 if not self.report_wer else sum(wers) / len(wers)
            cer = 0.0 if not self.report_cer else sum(cers) / len(cers)

        # acc is  from only decoder no ctc!! cer and wer are both ctc and decoder and rnnlm
        return loss_ctc_list, loss_att, acc, cer, wer

    def recognize(self, x, recog_args, char_list, rnnlm=None):
        '''E2E_MulEnc beam search

        :param ndarray x: input acouctic feature of (T, D)
        :param namespace recog_args: argment namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        '''
        prev = self.training
        self.eval()
        # subsample frame
        x = x[::self.subsample[0][0], :]
        ilen = [x.shape[0]]
        h = to_cuda(self, torch.from_numpy(
            np.array(x, dtype=np.float32)))

        # 1. encoder
        # make a utt list (1) to use the same interface for encoder
        h_list = [None] * self.num_enc
        h = h.contiguous()
        for idx in range(self.num_enc):
            h_list[idx], _ = getattr(self, "enc{}".format(idx))(h.unsqueeze(0), ilen)

        # calculate log P(z_t|X) for CTC scores
        if recog_args.ctc_weight > 0.0:
            lpz_list = [getattr(self, "ctc{}".format(idx)).log_softmax(h) for idx, h in enumerate(h_list)]
        else:
            lpz_list = [None] * self.num_enc

        # 2. decoder
        # decode the first utterance
        h_list = [h[0] for h in h_list]
        lpz_list = [lpz[0] for lpz in lpz_list]
        y = self.dec.recognize_beam(h_list, lpz_list, recog_args, char_list, rnnlm)

        if prev:
            self.train()
        return y

    def recognize_batch(self, xs, recog_args, char_list, rnnlm=None):
        '''E2E_MulEnc beam search

        :param ndarray x: input acouctic feature (T, D)
        :param namespace recog_args: argment namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        '''
        prev = self.training
        self.eval()
        # subsample frame
        xs = [xx[::self.subsample[0][0], :] for xx in xs]
        ilens = np.fromiter((xx.shape[0] for xx in xs), dtype=np.int64)
        hs = [to_cuda(self, torch.from_numpy(np.array(xx, dtype=np.float32)))
              for xx in xs]

        # 1. encoder
        xpad = pad_list(hs, 0.0)
        hpad_list = [None] * self.num_enc
        hlens_list = [None] * self.num_enc
        for idx in range(self.num_enc):
            hpad_list[idx], hlens_list[idx] = getattr(self, "enc{}".format(idx))(xpad, ilens)

        # calculate log P(z_t|X) for CTC scores
        if recog_args.ctc_weight > 0.0:
            lpz_list = [getattr(self, "ctc{}".format(idx)).log_softmax(hpad) for idx, hpad in enumerate(hpad_list)]
        else:
            lpz_list = [None] * self.num_enc

        # 2. decoder
        y = self.dec.recognize_beam_batch(hpad_list, hlens_list, lpz_list, recog_args, char_list, rnnlm)

        if prev:
            self.train()
        return y

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad):
        '''E2E attention calculation
        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :return: attention weights with the following shape,
            1) multi-head case => attention weights (B, H, Lmax, Tmax),
            2) multi-encoder case => [encatt(B, Lmax, NumEnc), [att1(B, Lmax, Tmax1), att2(B, Lmax, Tmax2), ...], ilens_list]
            3) other case => attention weights (B, Lmax, Tmax).
        :rtype: float ndarray or list(multi-encoder case)
        '''
        with torch.no_grad():
            # encoder
            hpad_list = [None] * self.num_enc
            hlens_list = [None] * self.num_enc
            for idx in range(self.num_enc):
                hpad_list[idx], hlens_list[idx] = getattr(self,"enc{}".format(idx))(xs_pad, ilens)

            # decoder
            encatt_ws, att_ws_list, ylens = self.dec.calculate_all_attentions(hpad_list, hlens_list, ys_pad)
        return encatt_ws, att_ws_list, hlens_list, ylens

    # todo add extract dec state, dec presoftmax and ctc presodtmax using batch recog

# ------------- CTC Network --------------------------------------------------------------------------------------------
class CTC(torch.nn.Module):
    """CTC module

    :param int odim: dimension of outputs
    :param int eprojs: number of encoder projection units
    :param float dropout_rate: dropout rate (0.0 ~ 1.0)
    """

    def __init__(self, odim, eprojs, dropout_rate):
        super(CTC, self).__init__()
        self.dropout_rate = dropout_rate
        self.loss = None
        self.ctc_lo = torch.nn.Linear(eprojs, odim)
        self.loss_fn = warp_ctc.CTCLoss(size_average=True)
        self.ignore_id = -1

    def forward(self, hs_pad, hlens, ys_pad):
        '''CTC forward

        :param torch.Tensor hs_pad: batch of padded hidden state sequences (B, Tmax, D)
        :param torch.Tensor hlens: batch of lengths of hidden state sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :return: ctc loss value
        :rtype: torch.Tensor
        '''
        # TODO(kan-bayashi): need to make more smart way
        ys = [y[y != self.ignore_id] for y in ys_pad]  # parse padded ys

        self.loss = None
        hlens = torch.from_numpy(np.fromiter(hlens, dtype=np.int32))
        olens = torch.from_numpy(np.fromiter(
            (x.size(0) for x in ys), dtype=np.int32))

        # zero padding for hs
        ys_hat = self.ctc_lo(F.dropout(hs_pad, p=self.dropout_rate))

        # zero padding for ys
        ys_true = torch.cat(ys).cpu().int()  # batch x olen

        # get length info
        logging.info(self.__class__.__name__ + ' input lengths:  ' + ''.join(str(hlens).split('\n')))
        logging.info(self.__class__.__name__ + ' output lengths: ' + ''.join(str(olens).split('\n')))

        # get ctc loss
        # expected shape of seqLength x batchSize x alphabet_size
        ys_hat = ys_hat.transpose(0, 1)
        self.loss = to_cuda(self, self.loss_fn(ys_hat, ys_true, hlens, olens))
        logging.info('ctc loss:' + str(float(self.loss)))

        return self.loss

    def log_softmax(self, hs_pad):
        '''log_softmax of frame activations

        :param torch.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        :return: log softmax applied 3d tensor (B, Tmax, odim)
        :rtype: torch.Tensor
        '''
        return F.log_softmax(self.ctc_lo(hs_pad), dim=2)


# ------------- Attention Network --------------------------------------------------------------------------------------
class NoAtt(torch.nn.Module):
    '''No attention'''
    # TODO no attention for enc attention actually it is fixed attention
    def __init__(self):
        super(NoAtt, self).__init__()
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.c = None

    def reset(self):
        '''reset states'''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.c = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev):
        '''NoAtt forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B, T_max, D_enc)
        :param list enc_h_len: padded encoder hidden state lenght (B)
        :param torch.Tensor dec_z: dummy (does not use)
        :param torch.Tensor att_prev: dummy (does not use)
        :return: attentioin weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: previous attentioin weights
        :rtype: torch.Tensor
        '''
        batch = len(enc_hs_pad)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)

        # initialize attention weight with uniform dist.
        if att_prev is None:
            # if no bias, 0 0-pad goes 0
            mask = 1. - make_pad_mask(enc_hs_len).float()
            att_prev = mask / mask.new(enc_hs_len).unsqueeze(-1)
            att_prev = att_prev.to(self.enc_h)
            self.c = torch.sum(self.enc_h * att_prev.view(batch, self.h_length, 1), dim=1)

        return self.c, att_prev
# TODO implent fix att

class AttDot(torch.nn.Module):
    '''Dot product attention

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    '''

    def __init__(self, eprojs, dunits, att_dim):
        super(AttDot, self).__init__()
        self.mlp_enc = torch.nn.Linear(eprojs, att_dim)
        self.mlp_dec = torch.nn.Linear(dunits, att_dim)

        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def reset(self):
        '''reset states'''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev, scaling=2.0):
        '''AttDot forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_h_len: padded encoder hidden state lenght (B)
        :param torch.Tensor dec_z: dummy (does not use)
        :param torch.Tensor att_prev: dummy (does not use)
        :param float scaling: scaling parameter before applying softmax
        :return: attentioin weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: previous attentioin weight (B x T_max)
        :rtype: torch.Tensor
        '''

        batch = enc_hs_pad.size(0)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = torch.tanh(self.mlp_enc(self.enc_h))

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        e = torch.sum(self.pre_compute_enc_h * torch.tanh(self.mlp_dec(dec_z)).view(batch, 1, self.att_dim),
                      dim=2)  # utt x frame

        # NOTE consider zero padding when compute w.
        if self.mask is None:
            self.mask = to_cuda(self, make_pad_mask(enc_hs_len))
        e.masked_fill_(self.mask, -float('inf'))
        w = F.softmax(scaling * e, dim=1)

        # weighted sum over flames
        # utt x hdim
        # NOTE use bmm instead of sum(*)
        c = torch.sum(self.enc_h * w.view(batch, self.h_length, 1), dim=1)
        return c, w


class AttAdd(torch.nn.Module):
    '''Additive attention
    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    '''

    def __init__(self, eprojs, dunits, att_dim, store_hs=True):
        super(AttAdd, self).__init__()
        self.mlp_enc = torch.nn.Linear(eprojs, att_dim)
        self.mlp_dec = torch.nn.Linear(dunits, att_dim, bias=False)
        self.gvec = torch.nn.Linear(att_dim, 1)
        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None
        self.store_hs = store_hs

    def reset(self):
        '''reset states'''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev, scaling=2.0):
        '''AttLoc forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_h_len: padded encoder hidden state lenght (B)
        :param torch.Tensor dec_z: docoder hidden state (B x D_dec)
        :param torch.Tensor att_prev: dummy (does not use)
        :param float scaling: scaling parameter before applying softmax
        :param boolean store_hs: store pre-compute hs (switch off when dealing with encoder-level attention)
        :return: attentioin weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: previous attentioin weights (B x T_max)
        :rtype: torch.Tensor
        '''

        batch = len(enc_hs_pad)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None or not self.store_hs:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = self.mlp_enc(self.enc_h)

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.mlp_dec(dec_z).view(batch, 1, self.att_dim)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        e = self.gvec(torch.tanh(self.pre_compute_enc_h + dec_z_tiled)).squeeze(2)

        # NOTE consider zero padding when compute w.
        if self.mask is None:
            self.mask = to_cuda(self, make_pad_mask(enc_hs_len))
        e.masked_fill_(self.mask, -float('inf'))
        w = F.softmax(scaling * e, dim=1)

        # weighted sum over flames
        # utt x hdim
        # NOTE use bmm instead of sum(*)
        # c = torch.sum(self.enc_h * w.view(batch, self.h_length, 1), dim=1)
        c = torch.matmul(w.unsqueeze(1), self.enc_h).squeeze(1)

        return c, w


class AttLoc(torch.nn.Module):
    '''location-aware attention

    Reference: Attention-Based Models for Speech Recognition
        (https://arxiv.org/pdf/1506.07503.pdf)

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    :param int aconv_chans: # channels of attention convolution
    :param int aconv_filts: filter size of attention convolution
    '''

    def __init__(self, eprojs, dunits, att_dim, aconv_chans, aconv_filts):
        super(AttLoc, self).__init__()
        self.mlp_enc = torch.nn.Linear(eprojs, att_dim)
        self.mlp_dec = torch.nn.Linear(dunits, att_dim, bias=False)
        self.mlp_att = torch.nn.Linear(aconv_chans, att_dim, bias=False)
        self.loc_conv = torch.nn.Conv2d(
            1, aconv_chans, (1, 2 * aconv_filts + 1), padding=(0, aconv_filts), bias=False)
        self.gvec = torch.nn.Linear(att_dim, 1)

        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def reset(self):
        '''reset states'''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev, scaling=2.0):
        '''AttLoc forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_h_len: padded encoder hidden state lenght (B)
        :param torch.Tensor dec_z: docoder hidden state (B x D_dec)
        :param torch.Tensor att_prev: previous attetion weight (B x T_max)
        :param float scaling: scaling parameter before applying softmax
        :return: attentioin weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: previous attentioin weights (B x T_max)
        :rtype: torch.Tensor
        '''
        batch = len(enc_hs_pad)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = self.mlp_enc(self.enc_h)

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        # initialize attention weight with uniform dist.
        if att_prev is None:
            # if no bias, 0 0-pad goes 0
            att_prev = to_cuda(self, (1. - make_pad_mask(enc_hs_len).float()))
            att_prev = att_prev / att_prev.new(enc_hs_len).unsqueeze(-1)

        # att_prev: utt x frame -> utt x 1 x 1 x frame -> utt x att_conv_chans x 1 x frame
        att_conv = self.loc_conv(att_prev.view(batch, 1, 1, self.h_length))
        # att_conv: utt x att_conv_chans x 1 x frame -> utt x frame x att_conv_chans
        att_conv = att_conv.squeeze(2).transpose(1, 2)
        # att_conv: utt x frame x att_conv_chans -> utt x frame x att_dim
        att_conv = self.mlp_att(att_conv)

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.mlp_dec(dec_z).view(batch, 1, self.att_dim)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        e = self.gvec(torch.tanh(att_conv + self.pre_compute_enc_h + dec_z_tiled)).squeeze(2)

        # NOTE consider zero padding when compute w.
        if self.mask is None:
            self.mask = to_cuda(self, make_pad_mask(enc_hs_len))
        e.masked_fill_(self.mask, -float('inf'))
        w = F.softmax(scaling * e, dim=1)

        # weighted sum over flames
        # utt x hdim
        c = torch.sum(self.enc_h * w.view(batch, self.h_length, 1), dim=1)

        return c, w


class AttCov(torch.nn.Module):
    '''Coverage mechanism attention

    Reference: Get To The Point: Summarization with Pointer-Generator Network
       (https://arxiv.org/abs/1704.04368)

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    '''

    def __init__(self, eprojs, dunits, att_dim):
        super(AttCov, self).__init__()
        self.mlp_enc = torch.nn.Linear(eprojs, att_dim)
        self.mlp_dec = torch.nn.Linear(dunits, att_dim, bias=False)
        self.wvec = torch.nn.Linear(1, att_dim)
        self.gvec = torch.nn.Linear(att_dim, 1)

        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def reset(self):
        '''reset states'''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev_list, scaling=2.0):
        '''AttCov forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_h_len: padded encoder hidden state lenght (B)
        :param torch.Tensor dec_z: docoder hidden state (B x D_dec)
        :param list att_prev_list: list of previous attetion weight
        :param float scaling: scaling parameter before applying softmax
        :return: attentioin weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: list of previous attentioin weights
        :rtype: list
        '''

        batch = len(enc_hs_pad)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = self.mlp_enc(self.enc_h)

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        # initialize attention weight with uniform dist.
        if att_prev_list is None:
            # if no bias, 0 0-pad goes 0
            att_prev_list = to_cuda(self, (1. - make_pad_mask(enc_hs_len).float()))
            att_prev_list = [att_prev_list / att_prev_list.new(enc_hs_len).unsqueeze(-1)]

        # att_prev_list: L' * [B x T] => cov_vec B x T
        cov_vec = sum(att_prev_list)
        # cov_vec: B x T => B x T x 1 => B x T x att_dim
        cov_vec = self.wvec(cov_vec.unsqueeze(-1))

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.mlp_dec(dec_z).view(batch, 1, self.att_dim)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        e = self.gvec(torch.tanh(cov_vec + self.pre_compute_enc_h + dec_z_tiled)).squeeze(2)

        # NOTE consider zero padding when compute w.
        if self.mask is None:
            self.mask = to_cuda(self, make_pad_mask(enc_hs_len))
        e.masked_fill_(self.mask, -float('inf'))
        w = F.softmax(scaling * e, dim=1)
        att_prev_list += [w]

        # weighted sum over flames
        # utt x hdim
        # NOTE use bmm instead of sum(*)
        c = torch.sum(self.enc_h * w.view(batch, self.h_length, 1), dim=1)

        return c, att_prev_list


class AttLoc2D(torch.nn.Module):
    '''2D location-aware attention

    This attention is an extended version of location aware attention.
    It take not only one frame before attention weights, but also earlier frames into account.

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    :param int aconv_chans: # channels of attention convolution
    :param int aconv_filts: filter size of attention convolution
    :param int att_win: attention window size (default=5)
    '''

    def __init__(self, eprojs, dunits, att_dim, att_win, aconv_chans, aconv_filts):
        super(AttLoc2D, self).__init__()
        self.mlp_enc = torch.nn.Linear(eprojs, att_dim)
        self.mlp_dec = torch.nn.Linear(dunits, att_dim, bias=False)
        self.mlp_att = torch.nn.Linear(aconv_chans, att_dim, bias=False)
        self.loc_conv = torch.nn.Conv2d(
            1, aconv_chans, (att_win, 2 * aconv_filts + 1), padding=(0, aconv_filts), bias=False)
        self.gvec = torch.nn.Linear(att_dim, 1)

        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.aconv_chans = aconv_chans
        self.att_win = att_win
        self.mask = None

    def reset(self):
        '''reset states'''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev, scaling=2.0):
        '''AttLoc2D forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_h_len: padded encoder hidden state lenght (B)
        :param torch.Tensor dec_z: docoder hidden state (B x D_dec)
        :param torch.Tensor att_prev: previous attetion weight (B x att_win x T_max)
        :param float scaling: scaling parameter before applying softmax
        :return: attentioin weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: previous attentioin weights (B x att_win x T_max)
        :rtype: torch.Tensor
        '''

        batch = len(enc_hs_pad)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = self.mlp_enc(self.enc_h)

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        # initialize attention weight with uniform dist.
        if att_prev is None:
            # B * [Li x att_win]
            # if no bias, 0 0-pad goes 0
            att_prev = to_cuda(self, (1. - make_pad_mask(enc_hs_len).float()))
            att_prev = att_prev / att_prev.new(enc_hs_len).unsqueeze(-1)
            att_prev = att_prev.unsqueeze(1).expand(-1, self.att_win, -1)

        # att_prev: B x att_win x Tmax -> B x 1 x att_win x Tmax -> B x C x 1 x Tmax
        att_conv = self.loc_conv(att_prev.unsqueeze(1))
        # att_conv: B x C x 1 x Tmax -> B x Tmax x C
        att_conv = att_conv.squeeze(2).transpose(1, 2)
        # att_conv: utt x frame x att_conv_chans -> utt x frame x att_dim
        att_conv = self.mlp_att(att_conv)

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.mlp_dec(dec_z).view(batch, 1, self.att_dim)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        e = self.gvec(torch.tanh(att_conv + self.pre_compute_enc_h + dec_z_tiled)).squeeze(2)

        # NOTE consider zero padding when compute w.
        if self.mask is None:
            self.mask = to_cuda(self, make_pad_mask(enc_hs_len))
        e.masked_fill_(self.mask, -float('inf'))
        w = F.softmax(scaling * e, dim=1)

        # weighted sum over flames
        # utt x hdim
        # NOTE use bmm instead of sum(*)
        c = torch.sum(self.enc_h * w.view(batch, self.h_length, 1), dim=1)

        # update att_prev: B x att_win x Tmax -> B x att_win+1 x Tmax -> B x att_win x Tmax
        att_prev = torch.cat([att_prev, w.unsqueeze(1)], dim=1)
        att_prev = att_prev[:, 1:]

        return c, att_prev


class AttLocRec(torch.nn.Module):
    '''location-aware recurrent attention

    This attention is an extended version of location aware attention.
    With the use of RNN, it take the effect of the history of attention weights into account.

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    :param int aconv_chans: # channels of attention convolution
    :param int aconv_filts: filter size of attention convolution
    '''

    def __init__(self, eprojs, dunits, att_dim, aconv_chans, aconv_filts):
        super(AttLocRec, self).__init__()
        self.mlp_enc = torch.nn.Linear(eprojs, att_dim)
        self.mlp_dec = torch.nn.Linear(dunits, att_dim, bias=False)
        self.loc_conv = torch.nn.Conv2d(
            1, aconv_chans, (1, 2 * aconv_filts + 1), padding=(0, aconv_filts), bias=False)
        self.att_lstm = torch.nn.LSTMCell(aconv_chans, att_dim, bias=False)
        self.gvec = torch.nn.Linear(att_dim, 1)

        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def reset(self):
        '''reset states'''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev_states, scaling=2.0):
        '''AttLocRec forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_h_len: padded encoder hidden state lenght (B)
        :param torch.Tensor dec_z: docoder hidden state (B x D_dec)
        :param tuple att_prev_states: previous attetion weight and lstm states
                                      ((B, T_max), ((B, att_dim), (B, att_dim)))
        :param float scaling: scaling parameter before applying softmax
        :return: attentioin weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: previous attention weights and lstm states (w, (hx, cx))
                 ((B, T_max), ((B, att_dim), (B, att_dim)))
        :rtype: tuple
        '''

        batch = len(enc_hs_pad)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = self.mlp_enc(self.enc_h)

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        if att_prev_states is None:
            # initialize attention weight with uniform dist.
            # if no bias, 0 0-pad goes 0
            att_prev = to_cuda(self, (1. - make_pad_mask(enc_hs_len).float()))
            att_prev = att_prev / att_prev.new(enc_hs_len).unsqueeze(-1)

            # initialize lstm states
            att_h = enc_hs_pad.new_zeros(batch, self.att_dim)
            att_c = enc_hs_pad.new_zeros(batch, self.att_dim)
            att_states = (att_h, att_c)
        else:
            att_prev = att_prev_states[0]
            att_states = att_prev_states[1]

        # B x 1 x 1 x T -> B x C x 1 x T
        att_conv = self.loc_conv(att_prev.view(batch, 1, 1, self.h_length))
        # apply non-linear
        att_conv = F.relu(att_conv)
        # B x C x 1 x T -> B x C x 1 x 1 -> B x C
        att_conv = F.max_pool2d(att_conv, (1, att_conv.size(3))).view(batch, -1)

        att_h, att_c = self.att_lstm(att_conv, att_states)

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.mlp_dec(dec_z).view(batch, 1, self.att_dim)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        e = self.gvec(torch.tanh(att_h.unsqueeze(1) + self.pre_compute_enc_h + dec_z_tiled)).squeeze(2)

        # NOTE consider zero padding when compute w.
        if self.mask is None:
            self.mask = to_cuda(self, make_pad_mask(enc_hs_len))
        e.masked_fill_(self.mask, -float('inf'))
        w = F.softmax(scaling * e, dim=1)

        # weighted sum over flames
        # utt x hdim
        # NOTE use bmm instead of sum(*)
        c = torch.sum(self.enc_h * w.view(batch, self.h_length, 1), dim=1)

        return c, (w, (att_h, att_c))


class AttCovLoc(torch.nn.Module):
    '''Coverage mechanism location aware attention

    This attention is a combination of coverage and location-aware attentions.

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    :param int aconv_chans: # channels of attention convolution
    :param int aconv_filts: filter size of attention convolution
    '''

    def __init__(self, eprojs, dunits, att_dim, aconv_chans, aconv_filts):
        super(AttCovLoc, self).__init__()
        self.mlp_enc = torch.nn.Linear(eprojs, att_dim)
        self.mlp_dec = torch.nn.Linear(dunits, att_dim, bias=False)
        self.mlp_att = torch.nn.Linear(aconv_chans, att_dim, bias=False)
        self.loc_conv = torch.nn.Conv2d(
            1, aconv_chans, (1, 2 * aconv_filts + 1), padding=(0, aconv_filts), bias=False)
        self.gvec = torch.nn.Linear(att_dim, 1)

        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.aconv_chans = aconv_chans
        self.mask = None

    def reset(self):
        '''reset states'''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev_list, scaling=2.0):
        '''AttCovLoc forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_h_len: padded encoder hidden state lenght (B)
        :param torch.Tensor dec_z: docoder hidden state (B x D_dec)
        :param list att_prev_list: list of previous attetion weight
        :param float scaling: scaling parameter before applying softmax
        :return: attentioin weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: list of previous attentioin weights
        :rtype: list
        '''

        batch = len(enc_hs_pad)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = self.mlp_enc(self.enc_h)

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        # initialize attention weight with uniform dist.
        if att_prev_list is None:
            # if no bias, 0 0-pad goes 0
            mask = 1. - make_pad_mask(enc_hs_len).float()
            att_prev_list = [to_cuda(self, mask / mask.new(enc_hs_len).unsqueeze(-1))]

        # att_prev_list: L' * [B x T] => cov_vec B x T
        cov_vec = sum(att_prev_list)

        # cov_vec: B x T -> B x 1 x 1 x T -> B x C x 1 x T
        att_conv = self.loc_conv(cov_vec.view(batch, 1, 1, self.h_length))
        # att_conv: utt x att_conv_chans x 1 x frame -> utt x frame x att_conv_chans
        att_conv = att_conv.squeeze(2).transpose(1, 2)
        # att_conv: utt x frame x att_conv_chans -> utt x frame x att_dim
        att_conv = self.mlp_att(att_conv)

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.mlp_dec(dec_z).view(batch, 1, self.att_dim)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        e = self.gvec(torch.tanh(att_conv + self.pre_compute_enc_h + dec_z_tiled)).squeeze(2)

        # NOTE consider zero padding when compute w.
        if self.mask is None:
            self.mask = to_cuda(self, make_pad_mask(enc_hs_len))
        e.masked_fill_(self.mask, -float('inf'))
        w = F.softmax(scaling * e, dim=1)
        att_prev_list += [w]

        # weighted sum over flames
        # utt x hdim
        # NOTE use bmm instead of sum(*)
        c = torch.sum(self.enc_h * w.view(batch, self.h_length, 1), dim=1)

        return c, att_prev_list


class AttMultiHeadDot(torch.nn.Module):
    '''Multi head dot product attention

    Reference: Attention is all you need
        (https://arxiv.org/abs/1706.03762)

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int ahead: # heads of multi head attention
    :param int att_dim_k: dimension k in multi head attention
    :param int att_dim_v: dimension v in multi head attention
    '''

    def __init__(self, eprojs, dunits, aheads, att_dim_k, att_dim_v):
        super(AttMultiHeadDot, self).__init__()
        self.mlp_q = torch.nn.ModuleList()
        self.mlp_k = torch.nn.ModuleList()
        self.mlp_v = torch.nn.ModuleList()
        for h in six.moves.range(aheads):
            self.mlp_q += [torch.nn.Linear(dunits, att_dim_k)]
            self.mlp_k += [torch.nn.Linear(eprojs, att_dim_k, bias=False)]
            self.mlp_v += [torch.nn.Linear(eprojs, att_dim_v, bias=False)]
        self.mlp_o = torch.nn.Linear(aheads * att_dim_v, eprojs, bias=False)
        self.dunits = dunits
        self.eprojs = eprojs
        self.aheads = aheads
        self.att_dim_k = att_dim_k
        self.att_dim_v = att_dim_v
        self.scaling = 1.0 / math.sqrt(att_dim_k)
        self.h_length = None
        self.enc_h = None
        self.pre_compute_k = None
        self.pre_compute_v = None
        self.mask = None

    def reset(self):
        '''reset states'''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_k = None
        self.pre_compute_v = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev):
        '''AttMultiHeadDot forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_h_len: padded encoder hidden state lenght (B)
        :param torch.Tensor dec_z: decoder hidden state (B x D_dec)
        :param torch.Tensor att_prev: dummy (does not use)
        :param float scaling: scaling parameter before applying softmax
        :return: attentioin weighted encoder state (B x D_enc)
        :rtype: torch.Tensor
        :return: list of previous attentioin weight (B x T_max) * aheads
        :rtype: list
        '''

        batch = enc_hs_pad.size(0)
        # pre-compute all k and v outside the decoder loop
        if self.pre_compute_k is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_k = [
                torch.tanh(self.mlp_k[h](self.enc_h)) for h in six.moves.range(self.aheads)]

        if self.pre_compute_v is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_v = [
                self.mlp_v[h](self.enc_h) for h in six.moves.range(self.aheads)]

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        c = []
        w = []
        for h in six.moves.range(self.aheads):
            e = torch.sum(self.pre_compute_k[h] * torch.tanh(self.mlp_q[h](dec_z)).view(
                batch, 1, self.att_dim_k), dim=2)  # utt x frame

            # NOTE consider zero padding when compute w.
            if self.mask is None:
                self.mask = to_cuda(self, make_pad_mask(enc_hs_len))
            e.masked_fill_(self.mask, -float('inf'))
            w += [F.softmax(self.scaling * e, dim=1)]

            # weighted sum over flames
            # utt x hdim
            # NOTE use bmm instead of sum(*)
            c += [torch.sum(self.pre_compute_v[h] * w[h].view(batch, self.h_length, 1), dim=1)]

        # concat all of c
        c = self.mlp_o(torch.cat(c, dim=1))

        return c, w


class AttMultiHeadAdd(torch.nn.Module):
    '''Multi head additive attention

    Reference: Attention is all you need
        (https://arxiv.org/abs/1706.03762)

    This attention is multi head attention using additive attention for each head.

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int ahead: # heads of multi head attention
    :param int att_dim_k: dimension k in multi head attention
    :param int att_dim_v: dimension v in multi head attention
    '''

    def __init__(self, eprojs, dunits, aheads, att_dim_k, att_dim_v):
        super(AttMultiHeadAdd, self).__init__()
        self.mlp_q = torch.nn.ModuleList()
        self.mlp_k = torch.nn.ModuleList()
        self.mlp_v = torch.nn.ModuleList()
        self.gvec = torch.nn.ModuleList()
        for h in six.moves.range(aheads):
            self.mlp_q += [torch.nn.Linear(dunits, att_dim_k)]
            self.mlp_k += [torch.nn.Linear(eprojs, att_dim_k, bias=False)]
            self.mlp_v += [torch.nn.Linear(eprojs, att_dim_v, bias=False)]
            self.gvec += [torch.nn.Linear(att_dim_k, 1)]
        self.mlp_o = torch.nn.Linear(aheads * att_dim_v, eprojs, bias=False)
        self.dunits = dunits
        self.eprojs = eprojs
        self.aheads = aheads
        self.att_dim_k = att_dim_k
        self.att_dim_v = att_dim_v
        self.scaling = 1.0 / math.sqrt(att_dim_k)
        self.h_length = None
        self.enc_h = None
        self.pre_compute_k = None
        self.pre_compute_v = None
        self.mask = None

    def reset(self):
        '''reset states'''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_k = None
        self.pre_compute_v = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev):
        '''AttMultiHeadAdd forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_h_len: padded encoder hidden state lenght (B)
        :param torch.Tensor dec_z: decoder hidden state (B x D_dec)
        :param torch.Tensor att_prev: dummy (does not use)
        :param float scaling: scaling parameter before applying softmax
        :return: attentioin weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: list of previous attentioin weight (B x T_max) * aheads
        :rtype: list
        '''

        batch = enc_hs_pad.size(0)
        # pre-compute all k and v outside the decoder loop
        if self.pre_compute_k is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_k = [
                self.mlp_k[h](self.enc_h) for h in six.moves.range(self.aheads)]

        if self.pre_compute_v is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_v = [
                self.mlp_v[h](self.enc_h) for h in six.moves.range(self.aheads)]

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        c = []
        w = []
        for h in six.moves.range(self.aheads):
            e = self.gvec[h](torch.tanh(
                self.pre_compute_k[h] + self.mlp_q[h](dec_z).view(batch, 1, self.att_dim_k))).squeeze(2)

            # NOTE consider zero padding when compute w.
            if self.mask is None:
                self.mask = to_cuda(self, make_pad_mask(enc_hs_len))
            e.masked_fill_(self.mask, -float('inf'))
            w += [F.softmax(self.scaling * e, dim=1)]

            # weighted sum over flames
            # utt x hdim
            # NOTE use bmm instead of sum(*)
            c += [torch.sum(self.pre_compute_v[h] * w[h].view(batch, self.h_length, 1), dim=1)]

        # concat all of c
        c = self.mlp_o(torch.cat(c, dim=1))

        return c, w


class AttMultiHeadLoc(torch.nn.Module):
    '''Multi head location based attention

    Reference: Attention is all you need
        (https://arxiv.org/abs/1706.03762)

    This attention is multi head attention using location-aware attention for each head.

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int aheads: # heads of multi head attention
    :param int att_dim_k: dimension k in multi head attention
    :param int att_dim_v: dimension v in multi head attention
    :param int aconv_chans: # channels of attention convolution
    :param int aconv_filts: filter size of attention convolution
    '''

    def __init__(self, eprojs, dunits, aheads, att_dim_k, att_dim_v, aconv_chans, aconv_filts):
        super(AttMultiHeadLoc, self).__init__()
        self.mlp_q = torch.nn.ModuleList()
        self.mlp_k = torch.nn.ModuleList()
        self.mlp_v = torch.nn.ModuleList()
        self.gvec = torch.nn.ModuleList()
        self.loc_conv = torch.nn.ModuleList()
        self.mlp_att = torch.nn.ModuleList()
        for h in six.moves.range(aheads):
            self.mlp_q += [torch.nn.Linear(dunits, att_dim_k)]
            self.mlp_k += [torch.nn.Linear(eprojs, att_dim_k, bias=False)]
            self.mlp_v += [torch.nn.Linear(eprojs, att_dim_v, bias=False)]
            self.gvec += [torch.nn.Linear(att_dim_k, 1)]
            self.loc_conv += [torch.nn.Conv2d(
                1, aconv_chans, (1, 2 * aconv_filts + 1), padding=(0, aconv_filts), bias=False)]
            self.mlp_att += [torch.nn.Linear(aconv_chans, att_dim_k, bias=False)]
        self.mlp_o = torch.nn.Linear(aheads * att_dim_v, eprojs, bias=False)
        self.dunits = dunits
        self.eprojs = eprojs
        self.aheads = aheads
        self.att_dim_k = att_dim_k
        self.att_dim_v = att_dim_v
        self.scaling = 1.0 / math.sqrt(att_dim_k)
        self.h_length = None
        self.enc_h = None
        self.pre_compute_k = None
        self.pre_compute_v = None
        self.mask = None

    def reset(self):
        '''reset states'''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_k = None
        self.pre_compute_v = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev, scaling=2.0):
        '''AttMultiHeadLoc forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_h_len: padded encoder hidden state lenght (B)
        :param torch.Tensor dec_z: decoder hidden state (B x D_dec)
        :param torch.Tensor att_prev: list of previous attentioin weight (B x T_max) * aheads
        :param float scaling: scaling parameter before applying softmax
        :return: attentioin weighted encoder state (B x D_enc)
        :rtype: torch.Tensor
        :return: list of previous attentioin weight (B x T_max) * aheads
        :rtype: list
        '''

        batch = enc_hs_pad.size(0)
        # pre-compute all k and v outside the decoder loop
        if self.pre_compute_k is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_k = [
                self.mlp_k[h](self.enc_h) for h in six.moves.range(self.aheads)]

        if self.pre_compute_v is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_v = [
                self.mlp_v[h](self.enc_h) for h in six.moves.range(self.aheads)]

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        if att_prev is None:
            att_prev = []
            for h in six.moves.range(self.aheads):
                # if no bias, 0 0-pad goes 0
                mask = 1. - make_pad_mask(enc_hs_len).float()
                att_prev += [to_cuda(self, mask / mask.new(enc_hs_len).unsqueeze(-1))]

        c = []
        w = []
        for h in six.moves.range(self.aheads):
            att_conv = self.loc_conv[h](att_prev[h].view(batch, 1, 1, self.h_length))
            att_conv = att_conv.squeeze(2).transpose(1, 2)
            att_conv = self.mlp_att[h](att_conv)

            e = self.gvec[h](torch.tanh(
                self.pre_compute_k[h] + att_conv + self.mlp_q[h](dec_z).view(
                    batch, 1, self.att_dim_k))).squeeze(2)

            # NOTE consider zero padding when compute w.
            if self.mask is None:
                self.mask = to_cuda(self, make_pad_mask(enc_hs_len))
            e.masked_fill_(self.mask, -float('inf'))
            w += [F.softmax(scaling * e, dim=1)]

            # weighted sum over flames
            # utt x hdim
            # NOTE use bmm instead of sum(*)
            c += [torch.sum(self.pre_compute_v[h] * w[h].view(batch, self.h_length, 1), dim=1)]

        # concat all of c
        c = self.mlp_o(torch.cat(c, dim=1))

        return c, w


class AttMultiHeadMultiResLoc(torch.nn.Module):
    '''Multi head multi resolution location based attention

    Reference: Attention is all you need
        (https://arxiv.org/abs/1706.03762)

    This attention is multi head attention using location-aware attention for each head.
    Furthermore, it uses different filter size for each head.

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int aheads: # heads of multi head attention
    :param int att_dim_k: dimension k in multi head attention
    :param int att_dim_v: dimension v in multi head attention
    :param int aconv_chans: maximum # channels of attention convolution
        each head use #ch = aconv_chans * (head + 1) / aheads
        e.g. aheads=4, aconv_chans=100 => filter size = 25, 50, 75, 100
    :param int aconv_filts: filter size of attention convolution
    '''

    def __init__(self, eprojs, dunits, aheads, att_dim_k, att_dim_v, aconv_chans, aconv_filts):
        super(AttMultiHeadMultiResLoc, self).__init__()
        self.mlp_q = torch.nn.ModuleList()
        self.mlp_k = torch.nn.ModuleList()
        self.mlp_v = torch.nn.ModuleList()
        self.gvec = torch.nn.ModuleList()
        self.loc_conv = torch.nn.ModuleList()
        self.mlp_att = torch.nn.ModuleList()
        for h in six.moves.range(aheads):
            self.mlp_q += [torch.nn.Linear(dunits, att_dim_k)]
            self.mlp_k += [torch.nn.Linear(eprojs, att_dim_k, bias=False)]
            self.mlp_v += [torch.nn.Linear(eprojs, att_dim_v, bias=False)]
            self.gvec += [torch.nn.Linear(att_dim_k, 1)]
            afilts = aconv_filts * (h + 1) // aheads
            self.loc_conv += [torch.nn.Conv2d(
                1, aconv_chans, (1, 2 * afilts + 1), padding=(0, afilts), bias=False)]
            self.mlp_att += [torch.nn.Linear(aconv_chans, att_dim_k, bias=False)]
        self.mlp_o = torch.nn.Linear(aheads * att_dim_v, eprojs, bias=False)
        self.dunits = dunits
        self.eprojs = eprojs
        self.aheads = aheads
        self.att_dim_k = att_dim_k
        self.att_dim_v = att_dim_v
        self.scaling = 1.0 / math.sqrt(att_dim_k)
        self.h_length = None
        self.enc_h = None
        self.pre_compute_k = None
        self.pre_compute_v = None
        self.mask = None

    def reset(self):
        '''reset states'''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_k = None
        self.pre_compute_v = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev):
        '''AttMultiHeadMultiResLoc forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_h_len: padded encoder hidden state lenght (B)
        :param torch.Tensor dec_z: decoder hidden state (B x D_dec)
        :param torch.Tensor att_prev: list of previous attentioin weight (B x T_max) * aheads
        :param float scaling: scaling parameter before applying softmax
        :return: attentioin weighted encoder state (B x D_enc)
        :rtype: torch.Tensor
        :return: list of previous attentioin weight (B x T_max) * aheads
        :rtype: list
        '''

        batch = enc_hs_pad.size(0)
        # pre-compute all k and v outside the decoder loop
        if self.pre_compute_k is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_k = [
                self.mlp_k[h](self.enc_h) for h in six.moves.range(self.aheads)]

        if self.pre_compute_v is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_v = [
                self.mlp_v[h](self.enc_h) for h in six.moves.range(self.aheads)]

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        if att_prev is None:
            att_prev = []
            for h in six.moves.range(self.aheads):
                # if no bias, 0 0-pad goes 0
                mask = 1. - make_pad_mask(enc_hs_len).float()
                att_prev += [to_cuda(self, mask / mask.new(enc_hs_len).unsqueeze(-1))]

        c = []
        w = []
        for h in six.moves.range(self.aheads):
            att_conv = self.loc_conv[h](att_prev[h].view(batch, 1, 1, self.h_length))
            att_conv = att_conv.squeeze(2).transpose(1, 2)
            att_conv = self.mlp_att[h](att_conv)

            e = self.gvec[h](torch.tanh(
                self.pre_compute_k[h] + att_conv + self.mlp_q[h](dec_z).view(
                    batch, 1, self.att_dim_k))).squeeze(2)

            # NOTE consider zero padding when compute w.
            if self.mask is None:
                self.mask = to_cuda(self, make_pad_mask(enc_hs_len))
            e.masked_fill_(self.mask, -float('inf'))
            w += [F.softmax(self.scaling * e, dim=1)]

            # weighted sum over flames
            # utt x hdim
            # NOTE use bmm instead of sum(*)
            c += [torch.sum(self.pre_compute_v[h] * w[h].view(batch, self.h_length, 1), dim=1)]

        # concat all of c
        c = self.mlp_o(torch.cat(c, dim=1))

        return c, w


class AttForward(torch.nn.Module):
    '''Forward attention

    Reference: Forward attention in sequence-to-sequence acoustic modeling for speech synthesis
        (https://arxiv.org/pdf/1807.06736.pdf)

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    :param int aconv_chans: # channels of attention convolution
    :param int aconv_filts: filter size of attention convolution
    '''

    def __init__(self, eprojs, dunits, att_dim, aconv_chans, aconv_filts):
        super(AttForward, self).__init__()
        self.mlp_enc = torch.nn.Linear(eprojs, att_dim)
        self.mlp_dec = torch.nn.Linear(dunits, att_dim, bias=False)
        self.mlp_att = torch.nn.Linear(aconv_chans, att_dim, bias=False)
        self.loc_conv = torch.nn.Conv2d(
            1, aconv_chans, (1, 2 * aconv_filts + 1), padding=(0, aconv_filts), bias=False)
        self.gvec = torch.nn.Linear(att_dim, 1)
        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def reset(self):
        '''reset states'''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev, scaling=1.0):
        '''AttForward forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_hs_len: padded encoder hidden state lenght (B)
        :param torch.Tensor dec_z: docoder hidden state (B x D_dec)
        :param torch.Tensor att_prev: attention weights of previous step
        :param float scaling: scaling parameter before applying softmax
        :return: attentioin weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: previous attentioin weights (B x T_max)
        :rtype: torch.Tensor
        '''
        batch = len(enc_hs_pad)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = self.mlp_enc(self.enc_h)

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        if att_prev is None:
            # initial attention will be [1, 0, 0, ...]
            att_prev = enc_hs_pad.new_zeros(*enc_hs_pad.size()[:2])
            att_prev[:, 0] = 1.0

        # att_prev: utt x frame -> utt x 1 x 1 x frame -> utt x att_conv_chans x 1 x frame
        att_conv = self.loc_conv(att_prev.view(batch, 1, 1, self.h_length))
        # att_conv: utt x att_conv_chans x 1 x frame -> utt x frame x att_conv_chans
        att_conv = att_conv.squeeze(2).transpose(1, 2)
        # att_conv: utt x frame x att_conv_chans -> utt x frame x att_dim
        att_conv = self.mlp_att(att_conv)

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.mlp_dec(dec_z).unsqueeze(1)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        e = self.gvec(torch.tanh(self.pre_compute_enc_h + dec_z_tiled + att_conv)).squeeze(2)

        # NOTE consider zero padding when compute w.
        if self.mask is None:
            self.mask = to_cuda(self, make_pad_mask(enc_hs_len))
        e.masked_fill_(self.mask, -float('inf'))
        w = F.softmax(scaling * e, dim=1)

        # forward attention
        att_prev_shift = F.pad(att_prev, (1, 0))[:, :-1]
        w = (att_prev + att_prev_shift) * w
        # NOTE: clamp is needed to avoid nan gradient
        w = F.normalize(torch.clamp(w, 1e-6), p=1, dim=1)

        # weighted sum over flames
        # utt x hdim
        # NOTE use bmm instead of sum(*)
        c = torch.sum(self.enc_h * w.unsqueeze(-1), dim=1)

        return c, w


class AttForwardTA(torch.nn.Module):
    '''Forward attention with transition agent

    Reference: Forward attention in sequence-to-sequence acoustic modeling for speech synthesis
        (https://arxiv.org/pdf/1807.06736.pdf)

    :param int eunits: # units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    :param int aconv_chans: # channels of attention convolution
    :param int aconv_filts: filter size of attention convolution
    :param int odim: output dimension
    '''

    def __init__(self, eunits, dunits, att_dim, aconv_chans, aconv_filts, odim):
        super(AttForwardTA, self).__init__()
        self.mlp_enc = torch.nn.Linear(eunits, att_dim)
        self.mlp_dec = torch.nn.Linear(dunits, att_dim, bias=False)
        self.mlp_ta = torch.nn.Linear(eunits + dunits + odim, 1)
        self.mlp_att = torch.nn.Linear(aconv_chans, att_dim, bias=False)
        self.loc_conv = torch.nn.Conv2d(
            1, aconv_chans, (1, 2 * aconv_filts + 1), padding=(0, aconv_filts), bias=False)
        self.gvec = torch.nn.Linear(att_dim, 1)
        self.dunits = dunits
        self.eunits = eunits
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None
        self.trans_agent_prob = 0.5

    def reset(self):
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None
        self.trans_agent_prob = 0.5

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev, out_prev, scaling=1.0):
        '''AttForwardTA forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B, Tmax, eunits)
        :param list enc_hs_len: padded encoder hidden state lenght (B)
        :param torch.Tensor dec_z: docoder hidden state (B, dunits)
        :param torch.Tensor att_prev: attention weights of previous step
        :param torch.Tensor prev_out: decoder outputs of previous step (B, odim)
        :param float scaling: scaling parameter before applying softmax
        :return: attentioin weighted encoder state (B, dunits)
        :rtype: torch.Tensor
        :return: previous attentioin weights (B, Tmax)
        :rtype: torch.Tensor
        '''
        batch = len(enc_hs_pad)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = self.mlp_enc(self.enc_h)

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        if att_prev is None:
            # initial attention will be [1, 0, 0, ...]
            att_prev = enc_hs_pad.new_zeros(*enc_hs_pad.size()[:2])
            att_prev[:, 0] = 1.0

        # att_prev: utt x frame -> utt x 1 x 1 x frame -> utt x att_conv_chans x 1 x frame
        att_conv = self.loc_conv(att_prev.view(batch, 1, 1, self.h_length))
        # att_conv: utt x att_conv_chans x 1 x frame -> utt x frame x att_conv_chans
        att_conv = att_conv.squeeze(2).transpose(1, 2)
        # att_conv: utt x frame x att_conv_chans -> utt x frame x att_dim
        att_conv = self.mlp_att(att_conv)

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.mlp_dec(dec_z).view(batch, 1, self.att_dim)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        e = self.gvec(torch.tanh(att_conv + self.pre_compute_enc_h + dec_z_tiled)).squeeze(2)

        # NOTE consider zero padding when compute w.
        if self.mask is None:
            self.mask = to_cuda(self, make_pad_mask(enc_hs_len))
        e.masked_fill_(self.mask, -float('inf'))
        w = F.softmax(scaling * e, dim=1)

        # forward attention
        att_prev_shift = F.pad(att_prev, (1, 0))[:, :-1]
        w = (self.trans_agent_prob * att_prev + (1 - self.trans_agent_prob) * att_prev_shift) * w
        # NOTE: clamp is needed to avoid nan gradient
        w = F.normalize(torch.clamp(w, 1e-6), p=1, dim=1)

        # weighted sum over flames
        # utt x hdim
        # NOTE use bmm instead of sum(*)
        c = torch.sum(self.enc_h * w.view(batch, self.h_length, 1), dim=1)

        # update transition agent prob
        self.trans_agent_prob = torch.sigmoid(
            self.mlp_ta(torch.cat([c, out_prev, dec_z], dim=1)))

        return c, w


# ------------- Decoder Network ----------------------------------------------------------------------------------------
class Decoder(torch.nn.Module):
    """Decoder module

    :param int eprojs: # encoder projection units
    :param int odim: dimension of outputs
    :param int dlayers: # decoder layers
    :param int dunits: # decoder units
    :param int sos: start of sequence symbol id
    :param int eos: end of sequence symbol id
    :param torch.nn.Module att: attention module
    :param int verbose: verbose level
    :param list char_list: list of character strings
    :param ndarray labeldist: distribution of label smoothing
    :param float lsm_weight: label smoothing weight
    """

    def __init__(self, eprojs, odim, dlayers, dunits, sos, eos, att, verbose=0,
                 char_list=None, labeldist=None, lsm_weight=0., sampling_probability=0.0):
        super(Decoder, self).__init__()
        self.dunits = dunits
        self.dlayers = dlayers
        self.embed = torch.nn.Embedding(odim, dunits)
        self.decoder = torch.nn.ModuleList()
        self.decoder += [torch.nn.LSTMCell(dunits + eprojs, dunits)]
        for l in six.moves.range(1, self.dlayers):
            self.decoder += [torch.nn.LSTMCell(dunits, dunits)]
        self.ignore_id = -1
        self.output = torch.nn.Linear(dunits, odim)

        self.loss = None
        self.att = att
        self.dunits = dunits
        self.sos = sos
        self.eos = eos
        self.odim = odim
        self.verbose = verbose
        self.char_list = char_list
        # for label smoothing
        self.labeldist = labeldist
        self.vlabeldist = None
        self.lsm_weight = lsm_weight
        self.sampling_probability = sampling_probability

        self.logzero = -10000000000.0

    def zero_state(self, hs_pad):
        return hs_pad.new_zeros(hs_pad.size(0), self.dunits)

    def forward(self, hs_pad, hlens, ys_pad):
        '''Decoder forward

        :param torch.Tensor hs_pad: batch of padded hidden state sequences (B, Tmax, D)
        :param torch.Tensor hlens: batch of lengths of hidden state sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy
        :rtype: float
        '''
        # TODO(kan-bayashi): need to make more smart way
        ys = [y[y != self.ignore_id] for y in ys_pad]  # parse padded ys

        # hlen should be list of integer
        hlens = list(map(int, hlens))

        self.loss = None
        # prepare input and output word sequences with sos/eos IDs
        eos = ys[0].new([self.eos])
        sos = ys[0].new([self.sos])
        ys_in = [torch.cat([sos, y], dim=0) for y in ys]
        ys_out = [torch.cat([y, eos], dim=0) for y in ys]

        # padding for ys with -1
        # pys: utt x olen
        ys_in_pad = pad_list(ys_in, self.eos)
        ys_out_pad = pad_list(ys_out, self.ignore_id)

        # get dim, length info
        batch = ys_out_pad.size(0)
        olength = ys_out_pad.size(1)
        logging.info(self.__class__.__name__ + ' input lengths:  ' + str(hlens))
        logging.info(self.__class__.__name__ + ' output lengths: ' + str([y.size(0) for y in ys_out]))

        # initialization
        c_list = [self.zero_state(hs_pad)]
        z_list = [self.zero_state(hs_pad)]
        for l in six.moves.range(1, self.dlayers):
            c_list.append(self.zero_state(hs_pad))
            z_list.append(self.zero_state(hs_pad))
        att_w = None
        z_all = []
        self.att.reset()  # reset pre-computation of h

        # pre-computation of embedding
        eys = self.embed(ys_in_pad)  # utt x olen x zdim

        # loop for an output sequence
        for i in six.moves.range(olength):
            att_c, att_w = self.att(hs_pad, hlens, z_list[0], att_w)
            if i > 0 and random.random() < self.sampling_probability:
                logging.info(' scheduled sampling ')
                z_out = self.output(z_all[-1])
                z_out = np.argmax(z_out.detach(), axis=1)
                z_out = self.embed(z_out.cuda())
                ey = torch.cat((z_out, att_c), dim=1)  # utt x (zdim + hdim)
            else:
                ey = torch.cat((eys[:, i, :], att_c), dim=1)  # utt x (zdim + hdim)
            z_list[0], c_list[0] = self.decoder[0](ey, (z_list[0], c_list[0]))
            for l in six.moves.range(1, self.dlayers):
                z_list[l], c_list[l] = self.decoder[l](
                    z_list[l - 1], (z_list[l], c_list[l]))
            z_all.append(z_list[-1])

        z_all = torch.stack(z_all, dim=1).view(batch * olength, self.dunits)
        # compute loss
        y_all = self.output(z_all)
        self.loss = F.cross_entropy(y_all, ys_out_pad.view(-1),
                                    ignore_index=self.ignore_id,
                                    size_average=True)
        # -1: eos, which is removed in the loss computation
        self.loss *= (np.mean([len(x) for x in ys_in]) - 1)
        acc = th_accuracy(y_all, ys_out_pad, ignore_label=self.ignore_id)
        logging.info('att loss:' + ''.join(str(self.loss.item()).split('\n')))

        # show predicted character sequence for debug
        if self.verbose > 0 and self.char_list is not None:
            ys_hat = y_all.view(batch, olength, -1)
            ys_true = ys_out_pad
            for (i, y_hat), y_true in zip(enumerate(ys_hat.detach().cpu().numpy()),
                                          ys_true.detach().cpu().numpy()):
                if i == MAX_DECODER_OUTPUT:
                    break
                idx_hat = np.argmax(y_hat[y_true != self.ignore_id], axis=1)
                idx_true = y_true[y_true != self.ignore_id]
                seq_hat = [self.char_list[int(idx)] for idx in idx_hat]
                seq_true = [self.char_list[int(idx)] for idx in idx_true]
                seq_hat = "".join(seq_hat)
                seq_true = "".join(seq_true)
                logging.info("groundtruth[%d]: " % i + seq_true)
                logging.info("prediction [%d]: " % i + seq_hat)

        if self.labeldist is not None:
            if self.vlabeldist is None:
                self.vlabeldist = to_cuda(self, torch.from_numpy(self.labeldist))
            loss_reg = - torch.sum((F.log_softmax(y_all, dim=1) * self.vlabeldist).view(-1), dim=0) / len(ys_in)
            self.loss = (1. - self.lsm_weight) * self.loss + self.lsm_weight * loss_reg

        return self.loss, acc

    def recognize_beam(self, h, lpz, recog_args, char_list, rnnlm=None):
        '''beam search implementation

        :param torch.Tensor h: encoder hidden state (T, eprojs)
        :param torch.Tensor lpz: ctc log softmax output (T, odim)
        :param Namespace recog_args: argument namespace contraining options
        :param char_list: list of character strings
        :param torch.nn.Module rnnlm: language module
        :return: N-best decoding results
        :rtype: list of dicts
        '''
        logging.info('input lengths: ' + str(h.size(0)))
        # initialization
        c_list = [self.zero_state(h.unsqueeze(0))]
        z_list = [self.zero_state(h.unsqueeze(0))]
        for l in six.moves.range(1, self.dlayers):
            c_list.append(self.zero_state(h.unsqueeze(0)))
            z_list.append(self.zero_state(h.unsqueeze(0)))
        a = None
        self.att.reset()  # reset pre-computation of h

        # search parms
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = recog_args.ctc_weight

        # preprate sos
        y = self.sos
        vy = h.new_zeros(1).long()

        if recog_args.maxlenratio == 0:
            maxlen = h.shape[0]
        else:
            # maxlen >= 1
            maxlen = max(1, int(recog_args.maxlenratio * h.size(0)))
        minlen = int(recog_args.minlenratio * h.size(0))
        logging.info('max output length: ' + str(maxlen))
        logging.info('min output length: ' + str(minlen))

        # initialize hypothesis
        if rnnlm:
            hyp = {'score': 0.0, 'yseq': [y], 'c_prev': c_list,
                   'z_prev': z_list, 'a_prev': a, 'rnnlm_prev': None}
        else:
            hyp = {'score': 0.0, 'yseq': [y], 'c_prev': c_list, 'z_prev': z_list, 'a_prev': a}
        if lpz is not None:
            ctc_prefix_score = CTCPrefixScore(lpz.detach().numpy(), 0, self.eos, np)
            hyp['ctc_state_prev'] = ctc_prefix_score.initial_state()
            hyp['ctc_score_prev'] = 0.0
            if ctc_weight != 1.0:
                # pre-pruning based on attention scores
                ctc_beam = min(lpz.shape[-1], int(beam * CTC_SCORING_RATIO))
            else:
                ctc_beam = lpz.shape[-1]
        hyps = [hyp]
        ended_hyps = []

        for i in six.moves.range(maxlen):
            logging.debug('position ' + str(i))

            hyps_best_kept = []
            for hyp in hyps:
                vy.unsqueeze(1)
                vy[0] = hyp['yseq'][i]
                ey = self.embed(vy)           # utt list (1) x zdim
                ey.unsqueeze(0)
                att_c, att_w = self.att(h.unsqueeze(0), [h.size(0)], hyp['z_prev'][0], hyp['a_prev'])
                ey = torch.cat((ey, att_c), dim=1)   # utt(1) x (zdim + hdim)
                z_list[0], c_list[0] = self.decoder[0](ey, (hyp['z_prev'][0], hyp['c_prev'][0]))
                for l in six.moves.range(1, self.dlayers):
                    z_list[l], c_list[l] = self.decoder[l](
                        z_list[l - 1], (hyp['z_prev'][l], hyp['c_prev'][l]))

                # get nbest local scores and their ids
                local_att_scores = F.log_softmax(self.output(z_list[-1]), dim=1)
                if rnnlm:
                    rnnlm_state, local_lm_scores = rnnlm.predict(hyp['rnnlm_prev'], vy)
                    local_scores = local_att_scores + recog_args.lm_weight * local_lm_scores
                else:
                    local_scores = local_att_scores

                if lpz is not None:
                    local_best_scores, local_best_ids = torch.topk(
                        local_att_scores, ctc_beam, dim=1)
                    ctc_scores, ctc_states = ctc_prefix_score(
                        hyp['yseq'], local_best_ids[0], hyp['ctc_state_prev'])
                    local_scores = \
                        (1.0 - ctc_weight) * local_att_scores[:, local_best_ids[0]] \
                        + ctc_weight * torch.from_numpy(ctc_scores - hyp['ctc_score_prev'])
                    if rnnlm:
                        local_scores += recog_args.lm_weight * local_lm_scores[:, local_best_ids[0]]
                    local_best_scores, joint_best_ids = torch.topk(local_scores, beam, dim=1)
                    local_best_ids = local_best_ids[:, joint_best_ids[0]]
                else:
                    local_best_scores, local_best_ids = torch.topk(local_scores, beam, dim=1)

                for j in six.moves.range(beam):
                    new_hyp = {}
                    # [:] is needed!
                    new_hyp['z_prev'] = z_list[:]
                    new_hyp['c_prev'] = c_list[:]
                    new_hyp['a_prev'] = att_w[:]
                    new_hyp['score'] = hyp['score'] + local_best_scores[0, j]
                    new_hyp['yseq'] = [0] * (1 + len(hyp['yseq']))
                    new_hyp['yseq'][:len(hyp['yseq'])] = hyp['yseq']
                    new_hyp['yseq'][len(hyp['yseq'])] = int(local_best_ids[0, j])
                    if rnnlm:
                        new_hyp['rnnlm_prev'] = rnnlm_state
                    if lpz is not None:
                        new_hyp['ctc_state_prev'] = ctc_states[joint_best_ids[0, j]]
                        new_hyp['ctc_score_prev'] = ctc_scores[joint_best_ids[0, j]]
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(
                    hyps_best_kept, key=lambda x: x['score'], reverse=True)[:beam]

            # sort and get nbest
            hyps = hyps_best_kept
            logging.debug('number of pruned hypothes: ' + str(len(hyps)))
            logging.debug(
                'best hypo: ' + ''.join([char_list[int(x)] for x in hyps[0]['yseq'][1:]]))

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                logging.info('adding <eos> in the last postion in the loop')
                for hyp in hyps:
                    hyp['yseq'].append(self.eos)

            # add ended hypothes to a final list, and removed them from current hypothes
            # (this will be a probmlem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp['yseq'][-1] == self.eos:
                    # only store the sequence that has more than minlen outputs
                    # also add penalty
                    if len(hyp['yseq']) > minlen:
                        hyp['score'] += (i + 1) * penalty
                        if rnnlm:  # Word LM needs to add final <eos> score
                            hyp['score'] += recog_args.lm_weight * rnnlm.final(
                                hyp['rnnlm_prev'])
                        ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            # end detection
            if end_detect(ended_hyps, i) and recog_args.maxlenratio == 0.0:
                logging.info('end detected at %d', i)
                break

            hyps = remained_hyps
            if len(hyps) > 0:
                logging.debug('remeined hypothes: ' + str(len(hyps)))
            else:
                logging.info('no hypothesis. Finish decoding.')
                break

            for hyp in hyps:
                logging.debug(
                    'hypo: ' + ''.join([char_list[int(x)] for x in hyp['yseq'][1:]]))

            logging.debug('number of ended hypothes: ' + str(len(ended_hyps)))

        nbest_hyps = sorted(
            ended_hyps, key=lambda x: x['score'], reverse=True)[:min(len(ended_hyps), recog_args.nbest)]

        # check number of hypotheis
        if len(nbest_hyps) == 0:
            logging.warn('there is no N-best results, perform recognition again with smaller minlenratio.')
            # should copy becasuse Namespace will be overwritten globally
            recog_args = Namespace(**vars(recog_args))
            recog_args.minlenratio = max(0.0, recog_args.minlenratio - 0.1)
            return self.recognize_beam(h, lpz, recog_args, char_list, rnnlm)

        logging.info('total log probability: ' + str(nbest_hyps[0]['score']))
        logging.info('normalized log probability: ' + str(nbest_hyps[0]['score'] / len(nbest_hyps[0]['yseq'])))

        # remove sos
        return nbest_hyps

    def recognize_beam_batch(self, h, hlens, lpz, recog_args, char_list, rnnlm=None,
                             normalize_score=True):
        logging.info('input lengths: ' + str(h.size(1)))
        h = mask_by_length(h, hlens, 0.0)

        # search params
        batch = len(hlens)
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = recog_args.ctc_weight
        att_weight = 1.0 - ctc_weight

        n_bb = batch * beam
        n_bo = beam * self.odim
        n_bbo = n_bb * self.odim
        pad_b = to_cuda(self, torch.LongTensor([i * beam for i in six.moves.range(batch)]).view(-1, 1))
        pad_bo = to_cuda(self, torch.LongTensor([i * n_bo for i in six.moves.range(batch)]).view(-1, 1))
        pad_o = to_cuda(self, torch.LongTensor([i * self.odim for i in six.moves.range(n_bb)]).view(-1, 1))

        max_hlen = int(max(hlens))
        if recog_args.maxlenratio == 0:
            maxlen = max_hlen
        else:
            maxlen = max(1, int(recog_args.maxlenratio * max_hlen))
        minlen = int(recog_args.minlenratio * max_hlen)
        logging.info('max output length: ' + str(maxlen))
        logging.info('min output length: ' + str(minlen))

        # initialization
        c_prev = [to_cuda(self, torch.zeros(n_bb, self.dunits)) for _ in range(self.dlayers)]
        z_prev = [to_cuda(self, torch.zeros(n_bb, self.dunits)) for _ in range(self.dlayers)]
        c_list = [to_cuda(self, torch.zeros(n_bb, self.dunits)) for _ in range(self.dlayers)]
        z_list = [to_cuda(self, torch.zeros(n_bb, self.dunits)) for _ in range(self.dlayers)]
        vscores = to_cuda(self, torch.zeros(batch, beam))

        a_prev = None
        rnnlm_prev = None

        self.att.reset()  # reset pre-computation of h

        yseq = [[self.sos] for _ in six.moves.range(n_bb)]
        accum_odim_ids = [self.sos for _ in six.moves.range(n_bb)]
        stop_search = [False for _ in six.moves.range(batch)]
        nbest_hyps = [[] for _ in six.moves.range(batch)]
        ended_hyps = [[] for _ in range(batch)]

        exp_hlens = hlens.repeat(beam).view(beam, batch).transpose(0, 1).contiguous()
        exp_hlens = exp_hlens.view(-1).tolist()
        exp_h = h.unsqueeze(1).repeat(1, beam, 1, 1).contiguous()
        exp_h = exp_h.view(n_bb, h.size()[1], h.size()[2])

        if lpz is not None:
            device_id = torch.cuda.device_of(next(self.parameters()).data).idx
            ctc_prefix_score = CTCPrefixScoreTH(lpz, 0, self.eos, beam, exp_hlens, device_id)
            ctc_states_prev = ctc_prefix_score.initial_state()
            ctc_scores_prev = to_cuda(self, torch.zeros(batch, n_bo))

        for i in six.moves.range(maxlen):
            logging.debug('position ' + str(i))

            vy = to_cuda(self, torch.LongTensor(get_last_yseq(yseq)))
            ey = self.embed(vy)
            att_c, att_w = self.att(exp_h, exp_hlens, z_prev[0], a_prev)
            ey = torch.cat((ey, att_c), dim=1)

            # attention decoder
            z_list[0], c_list[0] = self.decoder[0](ey, (z_prev[0], c_prev[0]))
            for l in six.moves.range(1, self.dlayers):
                z_list[l], c_list[l] = self.decoder[l](z_list[l - 1], (z_prev[l], c_prev[l]))
            local_scores = att_weight * F.log_softmax(self.output(z_list[-1]), dim=1)

            # rnnlm
            if rnnlm:
                rnnlm_state, local_lm_scores = rnnlm.buff_predict(rnnlm_prev, vy, n_bb)
                local_scores = local_scores + recog_args.lm_weight * local_lm_scores
            local_scores = local_scores.view(batch, n_bo)

            # ctc
            if lpz is not None:
                ctc_scores, ctc_states = ctc_prefix_score(yseq, ctc_states_prev, accum_odim_ids)
                ctc_scores = ctc_scores.view(batch, n_bo)
                local_scores = local_scores + ctc_weight * (ctc_scores - ctc_scores_prev)
            local_scores = local_scores.view(batch, beam, self.odim)

            if i == 0:
                local_scores[:, 1:, :] = self.logzero
            local_best_scores, local_best_odims = torch.topk(local_scores.view(batch, beam, self.odim),
                                                             beam, 2)
            # local pruning (via xp)
            local_scores = np.full((n_bbo,), self.logzero)
            _best_odims = local_best_odims.view(n_bb, beam) + pad_o
            _best_odims = _best_odims.view(-1).cpu().numpy()
            _best_score = local_best_scores.view(-1).cpu().detach().numpy()
            local_scores[_best_odims] = _best_score
            local_scores = to_cuda(self, torch.from_numpy(local_scores).float()).view(batch, beam, self.odim)

            # (or indexing)
            # local_scores = to_cuda(self, torch.full((batch, beam, self.odim), self.logzero))
            # _best_odims = local_best_odims
            # _best_score = local_best_scores
            # for si in six.moves.range(batch):
            # for bj in six.moves.range(beam):
            # for bk in six.moves.range(beam):
            # local_scores[si, bj, _best_odims[si, bj, bk]] = _best_score[si, bj, bk]

            eos_vscores = local_scores[:, :, self.eos] + vscores
            vscores = vscores.view(batch, beam, 1).repeat(1, 1, self.odim)
            vscores[:, :, self.eos] = self.logzero
            vscores = (vscores + local_scores).view(batch, n_bo)

            # global pruning
            accum_best_scores, accum_best_ids = torch.topk(vscores, beam, 1)
            accum_odim_ids = torch.fmod(accum_best_ids, self.odim).view(-1).data.cpu().tolist()
            accum_padded_odim_ids = (torch.fmod(accum_best_ids, n_bo) + pad_bo).view(-1).data.cpu().tolist()
            accum_padded_beam_ids = (torch.div(accum_best_ids, self.odim) + pad_b).view(-1).data.cpu().tolist()

            y_prev = yseq[:][:]
            yseq = index_select_list(yseq, accum_padded_beam_ids)
            yseq = append_ids(yseq, accum_odim_ids)
            vscores = accum_best_scores
            vidx = to_cuda(self, torch.LongTensor(accum_padded_beam_ids))

            a_prev = torch.index_select(att_w.view(n_bb, -1), 0, vidx)
            z_prev = [torch.index_select(z_list[li].view(n_bb, -1), 0, vidx) for li in range(self.dlayers)]
            c_prev = [torch.index_select(c_list[li].view(n_bb, -1), 0, vidx) for li in range(self.dlayers)]

            if rnnlm:
                rnnlm_prev = index_select_lm_state(rnnlm_state, 0, vidx)
            if lpz is not None:
                ctc_vidx = to_cuda(self, torch.LongTensor(accum_padded_odim_ids))
                ctc_scores_prev = torch.index_select(ctc_scores.view(-1), 0, ctc_vidx)
                ctc_scores_prev = ctc_scores_prev.view(-1, 1).repeat(1, self.odim).view(batch, n_bo)

                ctc_states = torch.transpose(ctc_states, 1, 3).contiguous()
                ctc_states = ctc_states.view(n_bbo, 2, -1)
                ctc_states_prev = torch.index_select(ctc_states, 0, ctc_vidx).view(n_bb, 2, -1)
                ctc_states_prev = torch.transpose(ctc_states_prev, 1, 2)

            # pick ended hyps
            if i > minlen:
                k = 0
                penalty_i = (i + 1) * penalty
                thr = accum_best_scores[:, -1]
                for samp_i in six.moves.range(batch):
                    if stop_search[samp_i]:
                        k = k + beam
                        continue
                    for beam_j in six.moves.range(beam):
                        if eos_vscores[samp_i, beam_j] > thr[samp_i]:
                            yk = y_prev[k][:]
                            yk.append(self.eos)
                            if len(yk) < hlens[samp_i]:
                                _vscore = eos_vscores[samp_i][beam_j] + penalty_i
                                if normalize_score:
                                    _vscore = _vscore / len(yk)
                                _score = _vscore.data.cpu().numpy()
                                ended_hyps[samp_i].append({'yseq': yk, 'vscore': _vscore, 'score': _score})
                        k = k + 1

            # end detection
            stop_search = [stop_search[samp_i] or end_detect(ended_hyps[samp_i], i)
                           for samp_i in six.moves.range(batch)]
            stop_search_summary = list(set(stop_search))
            if len(stop_search_summary) == 1 and stop_search_summary[0]:
                break

            torch.cuda.empty_cache()

        dummy_hyps = [{'yseq': [self.sos, self.eos], 'score':np.array([-float('inf')])}]
        ended_hyps = [ended_hyps[samp_i] if len(ended_hyps[samp_i]) != 0 else dummy_hyps
                      for samp_i in six.moves.range(batch)]
        nbest_hyps = [sorted(ended_hyps[samp_i], key=lambda x: x['score'],
                             reverse=True)[:min(len(ended_hyps[samp_i]), recog_args.nbest)]
                      for samp_i in six.moves.range(batch)]

        return nbest_hyps

    def bnf_dec(self, h, hlens, lpz, recog_args, char_list, rnnlm=None,
                             normalize_score=True):
        logging.info('input lengths: ' + str(h.size(1)))
        h = mask_by_length(h, hlens, 0.0)

        # search params
        batch = len(hlens)
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = recog_args.ctc_weight
        att_weight = 1.0 - ctc_weight
        bnf_component = recog_args.bnf_component

        n_bb = batch * beam
        n_bo = beam * self.odim
        n_bbo = n_bb * self.odim
        pad_b = to_cuda(self, torch.LongTensor([i * beam for i in six.moves.range(batch)]).view(-1, 1))
        pad_bo = to_cuda(self, torch.LongTensor([i * n_bo for i in six.moves.range(batch)]).view(-1, 1))
        pad_o = to_cuda(self, torch.LongTensor([i * self.odim for i in six.moves.range(n_bb)]).view(-1, 1))

        max_hlen = int(max(hlens))
        if recog_args.maxlenratio == 0:
            maxlen = max_hlen
        else:
            maxlen = max(1, int(recog_args.maxlenratio * max_hlen))
        minlen = int(recog_args.minlenratio * max_hlen)
        logging.info('max output length: ' + str(maxlen))
        logging.info('min output length: ' + str(minlen))

        # initialization
        c_prev = [to_cuda(self, torch.zeros(n_bb, self.dunits)) for _ in range(self.dlayers)]
        z_prev = [to_cuda(self, torch.zeros(n_bb, self.dunits)) for _ in range(self.dlayers)]
        c_list = [to_cuda(self, torch.zeros(n_bb, self.dunits)) for _ in range(self.dlayers)]
        z_list = [to_cuda(self, torch.zeros(n_bb, self.dunits)) for _ in range(self.dlayers)]
        vscores = to_cuda(self, torch.zeros(batch, beam))

        a_prev = None
        rnnlm_prev = None

        self.att.reset()  # reset pre-computation of h

        yseq = [[self.sos] for _ in six.moves.range(n_bb)]
        bnf = [[] for _ in six.moves.range(n_bb)]
        accum_odim_ids = [self.sos for _ in six.moves.range(n_bb)]
        stop_search = [False for _ in six.moves.range(batch)]
        nbest_hyps = [[] for _ in six.moves.range(batch)]
        ended_hyps = [[] for _ in range(batch)]

        exp_hlens = hlens.repeat(beam).view(beam, batch).transpose(0, 1).contiguous()
        exp_hlens = exp_hlens.view(-1).tolist()
        exp_h = h.unsqueeze(1).repeat(1, beam, 1, 1).contiguous()
        exp_h = exp_h.view(n_bb, h.size()[1], h.size()[2])

        if lpz is not None:
            device_id = torch.cuda.device_of(next(self.parameters()).data).idx
            ctc_prefix_score = CTCPrefixScoreTH(lpz, 0, self.eos, beam, exp_hlens, device_id)
            ctc_states_prev = ctc_prefix_score.initial_state()
            ctc_scores_prev = to_cuda(self, torch.zeros(batch, n_bo))

        for i in six.moves.range(maxlen):
            logging.debug('position ' + str(i))

            vy = to_cuda(self, torch.LongTensor(get_last_yseq(yseq)))
            ey = self.embed(vy)
            att_c, att_w = self.att(exp_h, exp_hlens, z_prev[0], a_prev)
            ey = torch.cat((ey, att_c), dim=1)

            # attention decoder
            z_list[0], c_list[0] = self.decoder[0](ey, (z_prev[0], c_prev[0]))
            for l in six.moves.range(1, self.dlayers):
                z_list[l], c_list[l] = self.decoder[l](z_list[l - 1], (z_prev[l], c_prev[l]))
            presm = self.output(z_list[-1]) # nbb x odim
            local_scores = att_weight * F.log_softmax(presm, dim=1)

            # rnnlm
            if rnnlm:
                rnnlm_state, local_lm_scores = rnnlm.buff_predict(rnnlm_prev, vy, n_bb)
                local_scores = local_scores + recog_args.lm_weight * local_lm_scores
            local_scores = local_scores.view(batch, n_bo)

            # ctc
            if lpz is not None:
                ctc_scores, ctc_states = ctc_prefix_score(yseq, ctc_states_prev, accum_odim_ids)
                ctc_scores = ctc_scores.view(batch, n_bo)
                local_scores = local_scores + ctc_weight * (ctc_scores - ctc_scores_prev)
            local_scores = local_scores.view(batch, beam, self.odim)

            if i == 0:
                local_scores[:, 1:, :] = self.logzero
            local_best_scores, local_best_odims = torch.topk(local_scores.view(batch, beam, self.odim),
                                                             beam, 2)
            # local pruning (via xp)
            local_scores = np.full((n_bbo,), self.logzero)
            _best_odims = local_best_odims.view(n_bb, beam) + pad_o
            _best_odims = _best_odims.view(-1).cpu().numpy()
            _best_score = local_best_scores.view(-1).cpu().detach().numpy()
            local_scores[_best_odims] = _best_score
            local_scores = to_cuda(self, torch.from_numpy(local_scores).float()).view(batch, beam, self.odim)

            # (or indexing)
            # local_scores = to_cuda(self, torch.full((batch, beam, self.odim), self.logzero))
            # _best_odims = local_best_odims
            # _best_score = local_best_scores
            # for si in six.moves.range(batch):
            # for bj in six.moves.range(beam):
            # for bk in six.moves.range(beam):
            # local_scores[si, bj, _best_odims[si, bj, bk]] = _best_score[si, bj, bk]

            eos_vscores = local_scores[:, :, self.eos] + vscores
            vscores = vscores.view(batch, beam, 1).repeat(1, 1, self.odim)
            vscores[:, :, self.eos] = self.logzero
            vscores = (vscores + local_scores).view(batch, n_bo)

            # global pruning
            accum_best_scores, accum_best_ids = torch.topk(vscores, beam, 1)
            accum_odim_ids = torch.fmod(accum_best_ids, self.odim).view(-1).data.cpu().tolist()
            accum_padded_odim_ids = (torch.fmod(accum_best_ids, n_bo) + pad_bo).view(-1).data.cpu().tolist()
            accum_padded_beam_ids = (torch.div(accum_best_ids, self.odim) + pad_b).view(-1).data.cpu().tolist()

            y_prev = yseq[:][:]
            bnf_prev = bnf[:][:]
            yseq = index_select_list(yseq, accum_padded_beam_ids)
            bnf = index_select_list(bnf, accum_padded_beam_ids)
            yseq = append_ids(yseq, accum_odim_ids)
            if bnf_component == 'ctxenc':
                bnf = append_states(bnf, att_c[accum_padded_beam_ids]) # att_c: bb x hdim
            elif bnf_component == 'decstate':
                bnf = append_states(bnf, z_list[0][accum_padded_beam_ids]) # att_c: bb x dunit
            elif bnf_component == 'decpresm':
                bnf = append_states(bnf, presm[accum_padded_beam_ids]) # att_c: bb x odim
            else:
                raise ValueError("ctxenc, decstate and decpresm are only supported.")

            vscores = accum_best_scores
            vidx = to_cuda(self, torch.LongTensor(accum_padded_beam_ids))

            a_prev = torch.index_select(att_w.view(n_bb, -1), 0, vidx)
            z_prev = [torch.index_select(z_list[li].view(n_bb, -1), 0, vidx) for li in range(self.dlayers)]
            c_prev = [torch.index_select(c_list[li].view(n_bb, -1), 0, vidx) for li in range(self.dlayers)]

            if rnnlm:
                rnnlm_prev = index_select_lm_state(rnnlm_state, 0, vidx)
            if lpz is not None:
                ctc_vidx = to_cuda(self, torch.LongTensor(accum_padded_odim_ids))
                ctc_scores_prev = torch.index_select(ctc_scores.view(-1), 0, ctc_vidx)
                ctc_scores_prev = ctc_scores_prev.view(-1, 1).repeat(1, self.odim).view(batch, n_bo)

                ctc_states = torch.transpose(ctc_states, 1, 3).contiguous()
                ctc_states = ctc_states.view(n_bbo, 2, -1)
                ctc_states_prev = torch.index_select(ctc_states, 0, ctc_vidx).view(n_bb, 2, -1)
                ctc_states_prev = torch.transpose(ctc_states_prev, 1, 2)

            # pick ended hyps
            if i > minlen:
                k = 0
                penalty_i = (i + 1) * penalty
                thr = accum_best_scores[:, -1]
                for samp_i in six.moves.range(batch):
                    if stop_search[samp_i]:
                        k = k + beam
                        continue
                    for beam_j in six.moves.range(beam):
                        if eos_vscores[samp_i, beam_j] > thr[samp_i]:
                            yk = y_prev[k][:]
                            yk.append(self.eos)
                            bnfk = torch.stack(bnf_prev[k][:], dim=0) # len(yk-1) x hdim
                            assert bnfk.size()[0] == len(yk)-2
                            if len(yk) < hlens[samp_i]:
                                _vscore = eos_vscores[samp_i][beam_j] + penalty_i
                                if normalize_score:
                                    _vscore = _vscore / len(yk)
                                _score = _vscore.data.cpu().numpy()
                                _bnfk = bnfk.data.cpu().numpy()
                                ended_hyps[samp_i].append({'yseq': yk, 'vscore': _vscore, 'score': _score, 'bnf': _bnfk})
                        k = k + 1

            # end detection
            stop_search = [stop_search[samp_i] or end_detect(ended_hyps[samp_i], i)
                           for samp_i in six.moves.range(batch)]
            stop_search_summary = list(set(stop_search))
            if len(stop_search_summary) == 1 and stop_search_summary[0]:
                break

            torch.cuda.empty_cache()

        dummy_hyps = [{'yseq': [self.sos, self.eos], 'score':np.array([-float('inf')])}]
        ended_hyps = [ended_hyps[samp_i] if len(ended_hyps[samp_i]) != 0 else dummy_hyps
                      for samp_i in six.moves.range(batch)]
        nbest_hyps = [sorted(ended_hyps[samp_i], key=lambda x: x['score'],
                             reverse=True)[:min(len(ended_hyps[samp_i]), recog_args.nbest)]
                      for samp_i in six.moves.range(batch)]

        return nbest_hyps

    def calculate_all_attentions(self, hs_pad, hlen, ys_pad):
        '''Calculate all of attentions

        :param torch.Tensor hs_pad: batch of padded hidden state sequences (B, Tmax, D)
        :param torch.Tensor hlens: batch of lengths of hidden state sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :return: attention weights with the following shape,
            1) multi-head case => attention weights (B, H, Lmax, Tmax),
            2) other case => attention weights (B, Lmax, Tmax).
        :rtype: float ndarray
        '''
        # TODO(kan-bayashi): need to make more smart way
        ys = [y[y != self.ignore_id] for y in ys_pad]  # parse padded ys

        # hlen should be list of integer
        hlen = list(map(int, hlen))

        self.loss = None
        # prepare input and output word sequences with sos/eos IDs
        eos = ys[0].new([self.eos])
        sos = ys[0].new([self.sos])
        ys_in = [torch.cat([sos, y], dim=0) for y in ys]
        ys_out = [torch.cat([y, eos], dim=0) for y in ys]

        # padding for ys with -1
        # pys: utt x olen
        ys_in_pad = pad_list(ys_in, self.eos)
        ys_out_pad = pad_list(ys_out, self.ignore_id)

        # get length info
        olength = ys_out_pad.size(1)

        # initialization
        c_list = [self.zero_state(hs_pad)]
        z_list = [self.zero_state(hs_pad)]
        for l in six.moves.range(1, self.dlayers):
            c_list.append(self.zero_state(hs_pad))
            z_list.append(self.zero_state(hs_pad))
        att_w = None
        att_ws = []
        self.att.reset()  # reset pre-computation of h

        # pre-computation of embedding
        eys = self.embed(ys_in_pad)  # utt x olen x zdim

        # loop for an output sequence
        for i in six.moves.range(olength):
            att_c, att_w = self.att(hs_pad, hlen, z_list[0], att_w)
            ey = torch.cat((eys[:, i, :], att_c), dim=1)  # utt x (zdim + hdim)
            z_list[0], c_list[0] = self.decoder[0](ey, (z_list[0], c_list[0]))
            for l in six.moves.range(1, self.dlayers):
                z_list[l], c_list[l] = self.decoder[l](
                    z_list[l - 1], (z_list[l], c_list[l]))
            att_ws.append(att_w)

        # convert to numpy array with the shape (B, Lmax, Tmax)
        if isinstance(self.att, AttLoc2D):
            # att_ws => list of previous concate attentions
            att_ws = torch.stack([aw[:, -1] for aw in att_ws], dim=1).cpu().numpy()
        elif isinstance(self.att, (AttCov, AttCovLoc)):
            # att_ws => list of list of previous attentions
            att_ws = torch.stack([aw[-1] for aw in att_ws], dim=1).cpu().numpy()
        elif isinstance(self.att, AttLocRec):
            # att_ws => list of tuple of attention and hidden states
            att_ws = torch.stack([aw[0] for aw in att_ws], dim=1).cpu().numpy()
        elif isinstance(self.att, (AttMultiHeadDot, AttMultiHeadAdd, AttMultiHeadLoc, AttMultiHeadMultiResLoc)):
            # att_ws => list of list of each head attetion
            n_heads = len(att_ws[0])
            att_ws_sorted_by_head = []
            for h in six.moves.range(n_heads):
                att_ws_head = torch.stack([aw[h] for aw in att_ws], dim=1)
                att_ws_sorted_by_head += [att_ws_head]
            att_ws = torch.stack(att_ws_sorted_by_head, dim=1).cpu().numpy()
        else:
            # att_ws => list of attetions
            att_ws = torch.stack(att_ws, dim=1).cpu().numpy()
        return att_ws


class Decoder_MulEnc(torch.nn.Module):
    """Decoder_MulEnc module

    :param int eprojs: # encoder projection units
    :param int odim: dimension of outputs
    :param int dlayers: # decoder layers
    :param int dunits: # decoder units
    :param int sos: start of sequence symbol id
    :param int eos: end of sequence symbol id
    :param torch.nn.Module encatt: encoder-level attention module
    :param list of torch.nn.Module att: list of attention module [encatt, att0, att1, ... attN]
    :param int verbose: verbose level
    :param list char_list: list of character strings
    :param ndarray labeldist: distribution of label smoothing
    :param float lsm_weight: label smoothing weight
    """

    def __init__(self, eprojs, odim, dlayers, dunits, sos, eos, encatt, att_list, verbose=0,
                 char_list=None, labeldist=None, lsm_weight=0., sampling_probability=0.0):
        super(Decoder_MulEnc, self).__init__()
        self.dunits = dunits
        self.dlayers = dlayers
        self.embed = torch.nn.Embedding(odim, dunits)
        self.decoder = torch.nn.ModuleList()
        self.decoder += [torch.nn.LSTMCell(dunits + eprojs, dunits)]
        for l in six.moves.range(1, self.dlayers):
            self.decoder += [torch.nn.LSTMCell(dunits, dunits)]
        self.ignore_id = -1
        self.output = torch.nn.Linear(dunits, odim)

        self.loss = None
        self.encatt = encatt
        self.att_list = att_list
        self.dunits = dunits
        self.sos = sos
        self.eos = eos
        self.odim = odim
        self.verbose = verbose
        self.char_list = char_list
        # for label smoothing
        self.labeldist = labeldist
        self.vlabeldist = None
        self.lsm_weight = lsm_weight
        self.sampling_probability = sampling_probability
        self.num_enc = len(att_list)

        self.logzero = -10000000000.0

    def zero_state(self, hs_pad):
        return hs_pad.new_zeros(hs_pad.size(0), self.dunits)


    def forward(self, hs_pad_list, hlens_list, ys_pad):
        '''Decoder forward. Only related to decoder. No CTC

        :param list of torch.Tensor hs_pad_list: list of batch of padded hidden state sequences [(B, Tmax, D), (B, Tmax, D), ..., ]
        :param list of torch.Tensor hlens_list: list of batch of lengths of hidden state sequences [(B), (B), ..., ]
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy
        :rtype: float
        '''
        # TODO(kan-bayashi): need to make more smart way
        ys = [y[y != self.ignore_id] for y in ys_pad]  # parse padded ys

        # check the length for att, hs and hlens;
        assert len(self.att_list) == len(hs_pad_list) == len(hlens_list)
        num_enc = len(hlens_list)

        # hlen_list should be list of list of integer
        hlens_list = [list(map(int, hlens)) for hlens in hlens_list]

        self.loss = None
        # prepare input and output word sequences with sos/eos IDs
        eos = ys[0].new([self.eos])
        sos = ys[0].new([self.sos])
        ys_in = [torch.cat([sos, y], dim=0) for y in ys]
        ys_out = [torch.cat([y, eos], dim=0) for y in ys]

        # padding for ys with -1
        # pys: utt x olen
        ys_in_pad = pad_list(ys_in, self.eos)
        ys_out_pad = pad_list(ys_out, self.ignore_id)

        # get dim, length info
        batch = ys_out_pad.size(0)
        olength = ys_out_pad.size(1)

        for i, hlens in enumerate(hlens_list): logging.info(self.__class__.__name__ + ' enc{}: input lengths:  {}'.format(i, hlens))
        logging.info(self.__class__.__name__ + ' output lengths: ' + str([y.size(0) for y in ys_out]))

        # initialization
        c_list = [self.zero_state(hs_pad_list[0])]
        z_list = [self.zero_state(hs_pad_list[0])]
        for l in six.moves.range(1, self.dlayers):
            c_list.append(self.zero_state(hs_pad_list[0]))
            z_list.append(self.zero_state(hs_pad_list[0]))
        att_w_list = [None] * num_enc
        att_c_list = [None] * num_enc
        encatt_w = None
        z_all = []
        for att in self.att_list : att.reset()  # reset pre-computation of h
        self.encatt.reset()  # reset pre-computation of h

        # pre-computation of embedding
        eys = self.embed(ys_in_pad)  # utt x olen x zdim

        # loop for an output sequence
        for i in six.moves.range(olength):
            # att
            dec_z = z_list[0].view(batch, self.dunits)
            for idx, att in enumerate(self.att_list):
                att_c_list[idx], att_w_list[idx] = att(hs_pad_list[idx], hlens_list[idx], dec_z, att_w_list[idx])
            # str
            encatt_hs_pad = torch.stack(att_c_list, dim=1)
            encatt_hlens = [num_enc] * batch
            encatt_c, encatt_w = self.encatt(encatt_hs_pad, encatt_hlens, dec_z, encatt_w)
            if i > 0 and random.random() < self.sampling_probability:
                logging.info(' scheduled sampling ')
                z_out = self.output(z_all[-1])
                z_out = np.argmax(z_out.detach(), axis=1)
                z_out = self.embed(z_out.cuda())
                ey = torch.cat((z_out, encatt_c), dim=1)  # utt x (zdim + hdim)
            else:
                ey = torch.cat((eys[:, i, :], encatt_c), dim=1)  # utt x (zdim + hdim)
            z_list[0], c_list[0] = self.decoder[0](ey, (z_list[0], c_list[0]))
            for l in six.moves.range(1, self.dlayers):
                z_list[l], c_list[l] = self.decoder[l](
                    z_list[l - 1], (z_list[l], c_list[l]))
            z_all.append(z_list[-1])

        z_all = torch.stack(z_all, dim=1).view(batch * olength, self.dunits)
        # compute loss
        y_all = self.output(z_all)
        self.loss = F.cross_entropy(y_all, ys_out_pad.view(-1),
                                    ignore_index=self.ignore_id,
                                    size_average=True)
        # -1: eos, which is removed in the loss computation
        self.loss *= (np.mean([len(x) for x in ys_in]) - 1)
        acc = th_accuracy(y_all, ys_out_pad, ignore_label=self.ignore_id)
        logging.info('att loss:' + ''.join(str(self.loss.item()).split('\n')))

        # show predicted character sequence for debug
        if self.verbose > 0 and self.char_list is not None:
            ys_hat = y_all.view(batch, olength, -1)
            ys_true = ys_out_pad
            for (i, y_hat), y_true in zip(enumerate(ys_hat.detach().cpu().numpy()),
                                          ys_true.detach().cpu().numpy()):
                if i == MAX_DECODER_OUTPUT:
                    break
                idx_hat = np.argmax(y_hat[y_true != self.ignore_id], axis=1)
                idx_true = y_true[y_true != self.ignore_id]
                seq_hat = [self.char_list[int(idx)] for idx in idx_hat]
                seq_true = [self.char_list[int(idx)] for idx in idx_true]
                seq_hat = "".join(seq_hat)
                seq_true = "".join(seq_true)
                logging.info("groundtruth[%d]: " % i + seq_true)
                logging.info("prediction [%d]: " % i + seq_hat)

        if self.labeldist is not None:
            if self.vlabeldist is None:
                self.vlabeldist = to_cuda(self, torch.from_numpy(self.labeldist))
            loss_reg = - torch.sum((F.log_softmax(y_all, dim=1) * self.vlabeldist).view(-1), dim=0) / len(ys_in) # todo BEN read paper KL div
            self.loss = (1. - self.lsm_weight) * self.loss + self.lsm_weight * loss_reg

        return self.loss, acc

    def recognize_beam(self, h_list, lpz_list, recog_args, char_list, rnnlm=None):
        '''beam search implementation

        :param list of torch.Tensor h: encoder hidden state list of (T, eprojs)
        :param list of torch.Tensor lpz: ctc log softmax output list of (T, odim)
        :param Namespace recog_args: argument namespace contraining options
        :param char_list: list of character strings
        :param torch.nn.Module rnnlm: language module
        :return: N-best decoding results
        :rtype: list of dicts
        '''
        for idx in range(self.num_enc): logging.info('input lengths: enc{} {}'.format(idx, h_list[idx].size(0)))
        # initialization
        c_list = [self.zero_state(h_list[0].unsqueeze(0))]
        z_list = [self.zero_state(h_list[0].unsqueeze(0))]
        for l in six.moves.range(1, self.dlayers):
            c_list.append(self.zero_state(h_list[0].unsqueeze(0)))
            z_list.append(self.zero_state(h_list[0].unsqueeze(0)))
        a_list = [None] * self.num_enc
        enca = None
        for att in self.att_list: att.reset()  # reset pre-computation of h
        self.encatt.reset()  # reset pre-computation of h

        # search parms
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = recog_args.ctc_weight
        ctc_ws = recog_args.ctc_ws
        ctc_ws = np.array([float(item) for item in ctc_ws.split("_")])
        ctc_ws = ctc_ws / np.sum(ctc_ws) # normalize
        ctc_ws = [float(item) for item in ctc_ws]
        logging.info('ctc weights (decoding): ' + ' '.join([str(x) for x in ctc_ws]))

        # preprate sos
        y = self.sos
        vy = h_list[0].new_zeros(1).long()

        maxlen = np.amin([h.shape[0] for h in h_list])
        if recog_args.maxlenratio != 0:
            # maxlen >= 1
            maxlen = max(1, int(recog_args.maxlenratio * maxlen))
        minlen = int(recog_args.minlenratio * maxlen)
        logging.info('max output length: ' + str(maxlen))
        logging.info('min output length: ' + str(minlen))

        # initialize hypothesis
        if rnnlm:
            hyp = {'score': 0.0, 'yseq': [y], 'c_prev': c_list,
                   'z_prev': z_list, 'a_prev_list': a_list, 'enca_prev': enca, 'rnnlm_prev': None}
        else:
            hyp = {'score': 0.0, 'yseq': [y], 'c_prev': c_list, 'z_prev': z_list, 'a_prev_list': a_list, 'enca_prev': enca}
        if lpz_list[0] is not None:
            ctc_prefix_score_list = [CTCPrefixScore(lpz_list[idx].detach().numpy(), 0, self.eos, np) for idx in range(self.num_enc)]
            hyp['ctc_state_prev_list'] = [ctc_prefix_score_list[idx].initial_state() for idx in range(self.num_enc)]
            hyp['ctc_score_prev_list'] = [0.0] * self.num_enc
            if ctc_weight != 1.0: # In batch recog, ctc_beam is always odim so batch recog might have better recog
                # pre-pruning based on attention scores
                ctc_beam = min(lpz_list[0].shape[-1], int(beam * CTC_SCORING_RATIO))
            else:
                ctc_beam = lpz_list[0].shape[-1]
        hyps = [hyp]
        ended_hyps = []

        for i in six.moves.range(maxlen):
            logging.debug('position ' + str(i))

            hyps_best_kept = []
            for hyp in hyps:
                vy.unsqueeze(1)
                vy[0] = hyp['yseq'][i]
                ey = self.embed(vy)
                ey.unsqueeze(0)
                att_w_list = [None] * self.num_enc
                att_c_list = [None] * self.num_enc
                dec_z = hyp['z_prev'][0].view(1, self.dunits)
                for idx, att in enumerate(self.att_list):
                    att_c_list[idx], att_w_list[idx] = att(h_list[idx].unsqueeze(0), [h_list[idx].size(0)], dec_z, hyp['a_prev_list'][idx])
                encatt_hs_pad = torch.stack(att_c_list, dim=1)
                encatt_hlens = [self.num_enc]
                encatt_c, encatt_w = self.encatt(encatt_hs_pad, encatt_hlens, dec_z, hyp['enca_prev'])
                ey = torch.cat((ey, encatt_c), dim=1) # utt(1) x (zdim + hdim)
                z_list[0], c_list[0] = self.decoder[0](ey, (hyp['z_prev'][0], hyp['c_prev'][0]))
                for l in six.moves.range(1, self.dlayers):
                    z_list[l], c_list[l] = self.decoder[l](
                        z_list[l - 1], (hyp['z_prev'][l], hyp['c_prev'][l]))

                # get nbest local scores and their ids
                local_att_scores = F.log_softmax(self.output(z_list[-1]), dim=1)
                if rnnlm:
                    rnnlm_state, local_lm_scores = rnnlm.predict(hyp['rnnlm_prev'], vy)
                    local_scores = local_att_scores + recog_args.lm_weight * local_lm_scores
                else:
                    local_scores = local_att_scores

                if lpz_list[0] is not None:
                    local_best_scores, local_best_ids = torch.topk(
                        local_att_scores, ctc_beam, dim=1)
                    ctc_scores_list = [None] * self.num_enc
                    ctc_states_list = [None] * self.num_enc
                    local_scores = \
                         (1.0 - ctc_weight) * local_att_scores[:, local_best_ids[0]]
                    for idx in range(self.num_enc):
                        ctc_scores_list[idx], ctc_states_list[idx] = ctc_prefix_score_list[idx](
                            hyp['yseq'], local_best_ids[0], hyp['ctc_state_prev_list'][idx])
                        local_scores = local_scores + ctc_weight * ctc_ws[idx] * torch.from_numpy(ctc_scores_list[idx] - hyp['ctc_score_prev_list'][idx])


                    if rnnlm:
                        local_scores += recog_args.lm_weight * local_lm_scores[:, local_best_ids[0]]
                    local_best_scores, joint_best_ids = torch.topk(local_scores, beam, dim=1)
                    local_best_ids = local_best_ids[:, joint_best_ids[0]]
                else:
                    local_best_scores, local_best_ids = torch.topk(local_scores, beam, dim=1)

                for j in six.moves.range(beam):
                    new_hyp = {}
                    # [:] is needed!
                    new_hyp['z_prev'] = z_list[:]
                    new_hyp['c_prev'] = c_list[:]
                    new_hyp['a_prev_list'] = [att_w_list[idx][:] for idx in range(self.num_enc)]
                    new_hyp['enca_prev'] = encatt_w[:]
                    new_hyp['score'] = hyp['score'] + local_best_scores[0, j]
                    new_hyp['yseq'] = [0] * (1 + len(hyp['yseq']))
                    new_hyp['yseq'][:len(hyp['yseq'])] = hyp['yseq']
                    new_hyp['yseq'][len(hyp['yseq'])] = int(local_best_ids[0, j])
                    if rnnlm:
                        new_hyp['rnnlm_prev'] = rnnlm_state
                    if lpz_list[0] is not None:
                        new_hyp['ctc_state_prev_list'] = [ctc_states_list[idx][joint_best_ids[0, j]] for idx in range(self.num_enc)]
                        new_hyp['ctc_score_prev_list'] = [ctc_scores_list[idx][joint_best_ids[0, j]] for idx in range(self.num_enc)]
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(
                    hyps_best_kept, key=lambda x: x['score'], reverse=True)[:beam]

            # sort and get nbest
            hyps = hyps_best_kept
            logging.debug('number of pruned hypothes: ' + str(len(hyps)))
            logging.debug(
                'best hypo: ' + ''.join([char_list[int(x)] for x in hyps[0]['yseq'][1:]]))

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                logging.info('adding <eos> in the last postion in the loop')
                for hyp in hyps:
                    hyp['yseq'].append(self.eos)

            # add ended hypothes to a final list, and removed them from current hypothes
            # (this will be a probmlem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp['yseq'][-1] == self.eos:
                    # only store the sequence that has more than minlen outputs
                    # also add penalty
                    if len(hyp['yseq']) > minlen:
                        hyp['score'] += (i + 1) * penalty
                        if rnnlm:  # Word LM needs to add final <eos> score
                            hyp['score'] += recog_args.lm_weight * rnnlm.final(
                                hyp['rnnlm_prev'])
                        ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            # end detection
            if end_detect(ended_hyps, i) and recog_args.maxlenratio == 0.0:
                logging.info('end detected at %d', i)
                break

            hyps = remained_hyps
            if len(hyps) > 0:
                logging.debug('remeined hypothes: ' + str(len(hyps)))
            else:
                logging.info('no hypothesis. Finish decoding.')
                break

            for hyp in hyps:
                logging.debug(
                    'hypo: ' + ''.join([char_list[int(x)] for x in hyp['yseq'][1:]]))

            logging.debug('number of ended hypothes: ' + str(len(ended_hyps)))

        nbest_hyps = sorted(
            ended_hyps, key=lambda x: x['score'], reverse=True)[:min(len(ended_hyps), recog_args.nbest)]

        # check number of hypotheis
        if len(nbest_hyps) == 0:
            logging.warn('there is no N-best results, perform recognition again with smaller minlenratio.')
            # should copy becasuse Namespace will be overwritten globally
            recog_args = Namespace(**vars(recog_args))
            recog_args.minlenratio = max(0.0, recog_args.minlenratio - 0.1)
            return self.recognize_beam(h_list, lpz_list, recog_args, char_list, rnnlm)

        logging.info('total log probability: ' + str(nbest_hyps[0]['score']))
        logging.info('normalized log probability: ' + str(nbest_hyps[0]['score'] / len(nbest_hyps[0]['yseq'])))

        # remove sos
        return nbest_hyps

    def recognize_beam_batch(self, h_list, hlens_list, lpz_list, recog_args, char_list, rnnlm=None,
                             normalize_score=True):
        for idx in range(self.num_enc):
            logging.info('input lengths: enc{} {}'.format(idx, h_list[idx].size(1)))
            h_list[idx] = mask_by_length(h_list[idx], hlens_list[idx], 0.0)

        # search params
        batch = len(hlens_list[0])
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = recog_args.ctc_weight
        att_weight = 1.0 - ctc_weight
        ctc_ws = recog_args.ctc_ws
        if isinstance(ctc_ws, str):
            ctc_ws = np.array([float(item) for item in ctc_ws.split("_")])
            ctc_ws = ctc_ws / np.sum(ctc_ws) # normalize
            ctc_ws = [float(item) for item in ctc_ws]
        logging.info('ctc weights (decoding): ' + ' '.join([str(x) for x in ctc_ws]))

        n_bb = batch * beam
        n_bo = beam * self.odim
        n_bbo = n_bb * self.odim
        pad_b = to_cuda(self, torch.LongTensor([i * beam for i in six.moves.range(batch)]).view(-1, 1))
        pad_bo = to_cuda(self, torch.LongTensor([i * n_bo for i in six.moves.range(batch)]).view(-1, 1))
        pad_o = to_cuda(self, torch.LongTensor([i * self.odim for i in six.moves.range(n_bb)]).view(-1, 1))

        max_hlen = np.amin([max(hlens) for hlens in hlens_list])
        if recog_args.maxlenratio == 0:
            maxlen = max_hlen
        else:
            maxlen = max(1, int(recog_args.maxlenratio * max_hlen))
        minlen = int(recog_args.minlenratio * max_hlen)
        logging.info('max output length: ' + str(maxlen))
        logging.info('min output length: ' + str(minlen))

        # initialization
        c_prev = [to_cuda(self, torch.zeros(n_bb, self.dunits)) for _ in range(self.dlayers)]
        z_prev = [to_cuda(self, torch.zeros(n_bb, self.dunits)) for _ in range(self.dlayers)]
        c_list = [to_cuda(self, torch.zeros(n_bb, self.dunits)) for _ in range(self.dlayers)]
        z_list = [to_cuda(self, torch.zeros(n_bb, self.dunits)) for _ in range(self.dlayers)]
        vscores = to_cuda(self, torch.zeros(batch, beam)) # keep track of accum scores

        a_prev_list = [None] * self.num_enc
        enca_prev = None
        rnnlm_prev = None

        for att in self.att_list : att.reset()  # reset pre-computation of h
        self.encatt.reset()  # reset pre-computation of h

        yseq = [[self.sos] for _ in six.moves.range(n_bb)]
        accum_odim_ids = [self.sos for _ in six.moves.range(n_bb)]
        stop_search = [False for _ in six.moves.range(batch)]
        nbest_hyps = [[] for _ in six.moves.range(batch)]
        ended_hyps = [[] for _ in six.moves.range(batch)]


        exp_hlens_list = [hlens.repeat(beam).view(beam, batch).transpose(0, 1).contiguous() for hlens in hlens_list]
        exp_hlens_list = [exp_hlens.view(-1).tolist() for exp_hlens in exp_hlens_list]
        exp_h_list = [h.unsqueeze(1).repeat(1, beam, 1, 1).contiguous() for h in h_list]
        exp_h_list = [exp_h_list[idx].view(n_bb, h_list[idx].size()[1], h_list[idx].size()[2]) for idx in range(self.num_enc)]

        if lpz_list[0] is not None:
            device_id = torch.cuda.device_of(next(self.parameters()).data).idx
            ctc_prefix_score_list = [CTCPrefixScoreTH(lpz_list[idx], 0, self.eos, beam, exp_hlens_list[idx], device_id) for idx in range(self.num_enc)]
            ctc_states_prev_list = [ctc_prefix_score.initial_state() for ctc_prefix_score in ctc_prefix_score_list]
            ctc_scores_prev_list = [to_cuda(self, torch.zeros(batch, n_bo)) for _ in range(self.num_enc)]

        for i in six.moves.range(maxlen):
            logging.debug('position ' + str(i))

            vy = to_cuda(self, torch.LongTensor(get_last_yseq(yseq)))
            ey = self.embed(vy)
            att_w_list = [None] * self.num_enc
            att_c_list = [None] * self.num_enc
            dec_z = z_prev[0].view(n_bb, self.dunits)
            for idx, att in enumerate(self.att_list):
                att_c_list[idx], att_w_list[idx] = att(exp_h_list[idx], exp_hlens_list[idx], dec_z, a_prev_list[idx])
            encatt_hs_pad = torch.stack(att_c_list, dim=1)
            encatt_hlens = [self.num_enc] * n_bb
            encatt_c, encatt_w = self.encatt(encatt_hs_pad, encatt_hlens, dec_z, enca_prev)
            ey = torch.cat((ey, encatt_c), dim=1)

            # attention decoder
            z_list[0], c_list[0] = self.decoder[0](ey, (z_prev[0], c_prev[0]))
            for l in six.moves.range(1, self.dlayers):
                z_list[l], c_list[l] = self.decoder[l](z_list[l - 1], (z_prev[l], c_prev[l]))
            local_scores = att_weight * F.log_softmax(self.output(z_list[-1]), dim=1)

            # rnnlm
            if rnnlm:
                rnnlm_state, local_lm_scores = rnnlm.buff_predict(rnnlm_prev, vy, n_bb)
                local_scores = local_scores + recog_args.lm_weight * local_lm_scores
            local_scores = local_scores.view(batch, n_bo)

            # ctc
            if lpz_list[0] is not None:
                ctc_scores_list = [None] * self.num_enc
                ctc_states_list = [None] * self.num_enc
                for idx in range(self.num_enc):
                    ctc_scores_list[idx], ctc_states_list[idx] = ctc_prefix_score_list[idx](yseq, ctc_states_prev_list[idx], accum_odim_ids)
                    ctc_scores_list[idx] = ctc_scores_list[idx].view(batch, n_bo)
                    local_scores = local_scores + ctc_weight * ctc_ws[idx] * (ctc_scores_list[idx] - ctc_scores_prev_list[idx])
            local_scores = local_scores.view(batch, beam, self.odim)

            if i == 0:
                local_scores[:, 1:, :] = self.logzero
            local_best_scores, local_best_odims = torch.topk(local_scores.view(batch, beam, self.odim),
                                                             beam, 2)
            # local pruning (via xp)
            local_scores = np.full((n_bbo,), self.logzero)
            _best_odims = local_best_odims.view(n_bb, beam) + pad_o
            _best_odims = _best_odims.view(-1).cpu().numpy()
            _best_score = local_best_scores.view(-1).cpu().detach().numpy()
            local_scores[_best_odims] = _best_score
            local_scores = to_cuda(self, torch.from_numpy(local_scores).float()).view(batch, beam, self.odim)

            # (or indexing)
            # local_scores = to_cuda(self, torch.full((batch, beam, self.odim), self.logzero))
            # _best_odims = local_best_odims
            # _best_score = local_best_scores
            # for si in six.moves.range(batch):
            # for bj in six.moves.range(beam):
            # for bk in six.moves.range(beam):
            # local_scores[si, bj, _best_odims[si, bj, bk]] = _best_score[si, bj, bk]

            eos_vscores = local_scores[:, :, self.eos] + vscores
            vscores = vscores.view(batch, beam, 1).repeat(1, 1, self.odim)
            vscores[:, :, self.eos] = self.logzero
            vscores = (vscores + local_scores).view(batch, n_bo)

            # global pruning
            accum_best_scores, accum_best_ids = torch.topk(vscores, beam, 1) # batch x beam
            accum_odim_ids = torch.fmod(accum_best_ids, self.odim).view(-1).data.cpu().tolist()
            accum_padded_odim_ids = (torch.fmod(accum_best_ids, n_bo) + pad_bo).view(-1).data.cpu().tolist() # TODO: torch.fmod(accum_best_ids, n_bo) should be the same as  accum_best_ids
            accum_padded_beam_ids = (torch.div(accum_best_ids, self.odim) + pad_b).view(-1).data.cpu().tolist()

            y_prev = yseq[:][:]
            yseq = index_select_list(yseq, accum_padded_beam_ids)
            yseq = append_ids(yseq, accum_odim_ids)
            vscores = accum_best_scores
            vidx = to_cuda(self, torch.LongTensor(accum_padded_beam_ids))

            a_prev_list = [torch.index_select(att_w.view(n_bb, -1), 0, vidx) for att_w in att_w_list]
            enca_prev = torch.index_select(encatt_w.view(n_bb, -1), 0, vidx)
            z_prev = [torch.index_select(z_list[li].view(n_bb, -1), 0, vidx) for li in range(self.dlayers)]
            c_prev = [torch.index_select(c_list[li].view(n_bb, -1), 0, vidx) for li in range(self.dlayers)]

            if rnnlm:
                rnnlm_prev = index_select_lm_state(rnnlm_state, 0, vidx)
            if lpz_list[0] is not None:
                ctc_vidx = to_cuda(self, torch.LongTensor(accum_padded_odim_ids))
                for idx in range(self.num_enc):
                    ctc_scores_prev_list[idx] = torch.index_select(ctc_scores_list[idx].view(-1), 0, ctc_vidx)
                    ctc_scores_prev_list[idx] = ctc_scores_prev_list[idx].view(-1, 1).repeat(1, self.odim).view(batch, n_bo)

                    ctc_states_list[idx] = torch.transpose(ctc_states_list[idx], 1, 3).contiguous()
                    ctc_states_list[idx] = ctc_states_list[idx].view(n_bbo, 2, -1)
                    ctc_states_prev_list[idx] = torch.index_select(ctc_states_list[idx], 0, ctc_vidx).view(n_bb, 2, -1)
                    ctc_states_prev_list[idx] = torch.transpose(ctc_states_prev_list[idx], 1, 2)

            # pick ended hyps
            if i > minlen:
                k = 0
                penalty_i = (i + 1) * penalty
                thr = accum_best_scores[:, -1]
                for samp_i in six.moves.range(batch):
                    if stop_search[samp_i]:
                        k = k + beam
                        continue
                    for beam_j in six.moves.range(beam):
                        if eos_vscores[samp_i, beam_j] > thr[samp_i]:
                            yk = y_prev[k][:]
                            yk.append(self.eos)
                            if len(yk) < min(hlens[samp_i] for hlens in hlens_list):
                                _vscore = eos_vscores[samp_i][beam_j] + penalty_i
                                if normalize_score:
                                    _vscore = _vscore / len(yk)
                                _score = _vscore.data.cpu().numpy()
                                ended_hyps[samp_i].append({'yseq': yk, 'vscore': _vscore, 'score': _score})
                        k = k + 1

            # end detection
            stop_search = [stop_search[samp_i] or end_detect(ended_hyps[samp_i], i)
                           for samp_i in six.moves.range(batch)]
            stop_search_summary = list(set(stop_search))
            if len(stop_search_summary) == 1 and stop_search_summary[0]:
                break

            torch.cuda.empty_cache()

        dummy_hyps = [{'yseq': [self.sos, self.eos], 'score':np.array([-float('inf')])}]
        ended_hyps = [ended_hyps[samp_i] if len(ended_hyps[samp_i]) != 0 else dummy_hyps
                      for samp_i in six.moves.range(batch)]
        nbest_hyps = [sorted(ended_hyps[samp_i], key=lambda x: x['score'],
                             reverse=True)[:min(len(ended_hyps[samp_i]), recog_args.nbest)]
                      for samp_i in six.moves.range(batch)]

        return nbest_hyps

    def calculate_all_attentions(self, hs_pad_list, hlens_list, ys_pad):
        '''Calculate all of attentions

        :param list of torch.Tensor hs_pad_list: batch of padded hidden state sequences list of (B, Tmax, D)
        :param list of torch.Tensor hlens_list: batch of lengths of hidden state sequences list of (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :return: attention weights with the following shape,
            1) multi-head case => attention weights (B, H, Lmax, Tmax),
            2) multi-encoder case => [encatt(B, Lmax, NumEnc), [att1(B, Lmax, Tmax1), att2(B, Lmax, Tmax2), ...]]
            2) other case => attention weights (B, Lmax, Tmax).
        :rtype: float ndarray list(multi-encoder case)
        '''
        # TODO(kan-bayashi): need to make more smart way
        ys = [y[y != self.ignore_id] for y in ys_pad]  # parse padded ys
        ylens = [len(y) for y in ys]

        # check the length for att, hs and hlens;
        assert len(self.att_list) == len(hs_pad_list) == len(hlens_list)

        # hlen_list should be list of list of integer
        hlens_list = [list(map(int, hlens)) for hlens in hlens_list]

        self.loss = None
        # prepare input and output word sequences with sos/eos IDs
        eos = ys[0].new([self.eos])
        sos = ys[0].new([self.sos])
        ys_in = [torch.cat([sos, y], dim=0) for y in ys]
        ys_out = [torch.cat([y, eos], dim=0) for y in ys]

        # padding for ys with -1
        # pys: utt x olen
        ys_in_pad = pad_list(ys_in, self.eos)
        ys_out_pad = pad_list(ys_out, self.ignore_id)

        # get length info
        batch = ys_out_pad.size(0)
        olength = ys_out_pad.size(1)

        # initialization
        c_list = [self.zero_state(hs_pad_list[0])]
        z_list = [self.zero_state(hs_pad_list[0])]
        for l in six.moves.range(1, self.dlayers):
            c_list.append(self.zero_state(hs_pad_list[0]))
            z_list.append(self.zero_state(hs_pad_list[0]))
        att_w_list = [None] * self.num_enc
        att_c_list = [None] * self.num_enc
        encatt_w = None
        attws_list = [[] for _ in range(self.num_enc)]
        encatt_ws = []
        for att in self.att_list : att.reset()  # reset pre-computation of h
        self.encatt.reset()  # reset pre-computation of h

        # pre-computation of embedding
        eys = self.embed(ys_in_pad)  # utt x olen x zdim

        # loop for an output sequence
        for i in six.moves.range(olength):
            # att
            dec_z = z_list[0].view(batch, self.dunits)
            for idx, att in enumerate(self.att_list):
                att_c_list[idx], att_w_list[idx] = att(hs_pad_list[idx], hlens_list[idx], dec_z, att_w_list[idx])
                attws_list[idx].append(att_w_list[idx])
            # str
            encatt_hs_pad = torch.stack(att_c_list, dim=1)
            encatt_hlens = [self.num_enc] * batch
            encatt_c, encatt_w = self.encatt(encatt_hs_pad, encatt_hlens, dec_z, encatt_w)
            ey = torch.cat((eys[:, i, :], encatt_c), dim=1)  # utt x (zdim + hdim)
            z_list[0], c_list[0] = self.decoder[0](ey, (z_list[0], c_list[0]))
            for l in six.moves.range(1, self.dlayers):
                z_list[l], c_list[l] = self.decoder[l](
                    z_list[l - 1], (z_list[l], c_list[l]))
            encatt_ws.append(encatt_w)

        # convert to numpy array with the shape (B, Lmax, Tmax)
        # attws_list = [att1_ws, att2_ws, attN_ws]; attN = [BxTmax, BxTmax, ...., BxTmax] (len(attN) = Lmax)
        # ==> [att1_ws, att2_ws, attN_ws]; attN = B x Lmax x Tmax (numpy)
        # attws_list = [torch.stack(attws_list[idx], dim=1).cpu().numpy() for idx in range(self.num_enc)]
        # encatt_ws = torch.stack(encatt_ws, dim=1).cpu().numpy()
        encatt_ws = None # todo
        attws_list = [None] * self.num_enc
        return [encatt_ws, attws_list, ylens]

# ------------- Encoder Network ----------------------------------------------------------------------------------------
class Encoder(torch.nn.Module):
    '''Encoder module

    :param str etype: type of encoder network
    :param int idim: number of dimensions of encoder network
    :param int elayers: number of layers of encoder network
    :param int eunits: number of lstm units of encoder network
    :param int epojs: number of projection units of encoder network
    :param list subsample: list of subsampling numbers
    :param float dropout: dropout rate
    :param int in_channel: number of input channels
    '''

    def __init__(self, etype, idim, elayers, eunits, eprojs, subsample, dropout, in_channel=1):
        super(Encoder, self).__init__()

        if etype == 'blstm':
            self.enc1 = BLSTM(idim, elayers, eunits, eprojs, dropout)
            logging.info('BLSTM without projection for encoder')
        elif etype == 'blstmp':
            self.enc1 = BLSTMP(idim, elayers, eunits,
                               eprojs, subsample, dropout)
            logging.info('BLSTM with every-layer projection for encoder')
        elif etype == 'vggblstmp':
            self.enc1 = VGG(in_channels=in_channel)
            self.enc2 = BLSTMP(get_maxpooling2_odim(idim, in_channel=in_channel),
                               elayers, eunits, eprojs,
                               subsample, dropout)
            # # todo BEN
            # self.enc1 = VGG2L(in_channel)
            # self.enc2 = BLSTMP(get_vgg2l_odim(idim, in_channel=in_channel),
            #                    elayers, eunits, eprojs,
            #                    subsample, dropout)
            logging.info('Use CNN-VGG + BLSTMP for encoder')
        elif etype == 'vggblstm':
            self.enc1 = VGG2L(in_channel)
            self.enc2 = BLSTM(get_vgg2l_odim(idim, in_channel=in_channel),
                              elayers, eunits, eprojs, dropout)
            logging.info('Use CNN-VGG + BLSTM for encoder')
        else:
            logging.error(
                "Error: need to specify an appropriate encoder archtecture")
            sys.exit()

        self.etype = etype

    def forward(self, xs_pad, ilens):
        '''Encoder forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, D)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :return: batch of hidden state sequences (B, Tmax, erojs)
        :rtype: torch.Tensor
        '''
        if self.etype == 'blstm':
            xs_pad, ilens = self.enc1(xs_pad, ilens)
        elif self.etype == 'blstmp':
            xs_pad, ilens = self.enc1(xs_pad, ilens)
        elif self.etype == 'vggblstmp':
            xs_pad, ilens = self.enc1(xs_pad, ilens)
            xs_pad, ilens = self.enc2(xs_pad, ilens)
        elif self.etype == 'vggblstm':
            xs_pad, ilens = self.enc1(xs_pad, ilens)
            xs_pad, ilens = self.enc2(xs_pad, ilens)
        else:
            logging.error(
                "Error: need to specify an appropriate encoder archtecture")
            sys.exit()

        # make mask to remove bias value in padded part
        mask = to_cuda(self, make_pad_mask(ilens).unsqueeze(-1))

        return xs_pad.masked_fill(mask, 0.0), ilens


class BLSTMP(torch.nn.Module):
    """Bidirectional LSTM with projection layer module

    :param int idim: dimension of inputs
    :param int elayers: number of encoder layers
    :param int cdim: number of lstm units (resulted in cdim * 2 due to biderectional)
    :param int hdim: number of projection units
    :param list subsample: list of subsampling numbers
    :param float dropout: dropout rate
    """

    def __init__(self, idim, elayers, cdim, hdim, subsample, dropout):
        super(BLSTMP, self).__init__()
        for i in six.moves.range(elayers):
            if i == 0:
                inputdim = idim
            else:
                inputdim = hdim
            setattr(self, "bilstm%d" % i, torch.nn.LSTM(inputdim, cdim, dropout=dropout,
                                                        num_layers=1, bidirectional=True, batch_first=True))
            # bottleneck layer to merge
            setattr(self, "bt%d" % i, torch.nn.Linear(2 * cdim, hdim))

        self.elayers = elayers
        self.cdim = cdim
        self.subsample = subsample

    def forward(self, xs_pad, ilens):
        '''BLSTMP forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :return: batch of hidden state sequences (B, Tmax, hdim)
        :rtype: torch.Tensor
        '''
        # logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))
        for layer in six.moves.range(self.elayers):
            xs_pack = pack_padded_sequence(xs_pad, ilens, batch_first=True)
            bilstm = getattr(self, 'bilstm' + str(layer))
            bilstm.flatten_parameters()
            ys, _ = bilstm(xs_pack)
            # ys: utt list of frame x cdim x 2 (2: means bidirectional)
            ys_pad, ilens = pad_packed_sequence(ys, batch_first=True)
            sub = self.subsample[layer + 1]
            if sub > 1:
                ys_pad = ys_pad[:, ::sub]
                ilens = torch.LongTensor([int(i + 1) // sub for i in ilens])
            # (sum _utt frame_utt) x dim
            projected = getattr(self, 'bt' + str(layer)
                                )(ys_pad.contiguous().view(-1, ys_pad.size(2)))
            xs_pad = torch.tanh(projected.view(ys_pad.size(0), ys_pad.size(1), -1))

        return xs_pad, ilens  # x: utt list of frame x dim


class BLSTM(torch.nn.Module):
    """Bidirectional LSTM module

    :param int idim: dimension of inputs
    :param int elayers: number of encoder layers
    :param int cdim: number of lstm units (resulted in cdim * 2 due to biderectional)
    :param int hdim: number of final projection units
    :param list subsample: list of subsampling numbers
    :param float dropout: dropout rate
    """

    def __init__(self, idim, elayers, cdim, hdim, dropout):
        super(BLSTM, self).__init__()
        self.nblstm = torch.nn.LSTM(idim, cdim, elayers, batch_first=True,
                                    dropout=dropout, bidirectional=True)
        self.l_last = torch.nn.Linear(cdim * 2, hdim)

    def forward(self, xs_pad, ilens):
        '''BLSTM forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, D)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :return: batch of hidden state sequences (B, Tmax, erojs)
        :rtype: torch.Tensor
        '''
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))
        xs_pack = pack_padded_sequence(xs_pad, ilens, batch_first=True)
        self.nblstm.flatten_parameters()
        ys, _ = self.nblstm(xs_pack)
        # ys: utt list of frame x cdim x 2 (2: means bidirectional)
        ys_pad, ilens = pad_packed_sequence(ys, batch_first=True)
        # (sum _utt frame_utt) x dim
        projected = torch.tanh(self.l_last(
            ys_pad.contiguous().view(-1, ys_pad.size(2))))
        xs_pad = projected.view(ys_pad.size(0), ys_pad.size(1), -1)
        return xs_pad, ilens  # x: utt list of frame x dim


class VGG2L(torch.nn.Module):
    """VGG-like module

    :param int in_channel: number of input channels
    """

    def __init__(self, in_channel=1):
        super(VGG2L, self).__init__()
        # CNN layer (VGG motivated)
        self.conv1_1 = torch.nn.Conv2d(in_channel, 64, 3, stride=1, padding=1)
        self.conv1_2 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv2_1 = torch.nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv2_2 = torch.nn.Conv2d(128, 128, 3, stride=1, padding=1)

        self.in_channel = in_channel

    def forward(self, xs_pad, ilens):
        '''VGG2L forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, D)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :return: batch of padded hidden state sequences (B, Tmax // 4, 128)
        :rtype: torch.Tensor
        '''
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))

        # x: utt x frame x dim
        # xs_pad = F.pad_sequence(xs_pad)

        # x: utt x 1 (input channel num) x frame x dim
        xs_pad = xs_pad.view(xs_pad.size(0), xs_pad.size(1), self.in_channel,
                             xs_pad.size(2) // self.in_channel).transpose(1, 2)

        # NOTE: max_pool1d ?
        xs_pad = F.relu(self.conv1_1(xs_pad))
        xs_pad = F.relu(self.conv1_2(xs_pad))
        xs_pad = F.max_pool2d(xs_pad, 2, stride=2, ceil_mode=True)

        xs_pad = F.relu(self.conv2_1(xs_pad))
        xs_pad = F.relu(self.conv2_2(xs_pad))
        xs_pad = F.max_pool2d(xs_pad, 2, stride=2, ceil_mode=True)
        ilens = np.array(
            np.ceil(np.array(ilens, dtype=np.float32) / 2), dtype=np.int64)
        ilens = np.array(
            np.ceil(np.array(ilens, dtype=np.float32) / 2), dtype=np.int64).tolist()

        # x: utt_list of frame (remove zeropaded frames) x (input channel num x dim)
        xs_pad = xs_pad.transpose(1, 2)
        xs_pad = xs_pad.contiguous().view(
            xs_pad.size(0), xs_pad.size(1), xs_pad.size(2) * xs_pad.size(3))
        return xs_pad, ilens


# todo BEN add VGG
######################## Reimplementation of CNN part (Ruizhi)

class ConvBasicBlock(torch.nn.Module):
    """convolution followed by optional batchnorm, nonlinear, spatialdropout, maxpooling"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, nonlinear=False,
                 batch_norm=False, spatial_dp=False, maxpooling=False, maxpool_kernel_size=3, maxpool_stride=2,
                 maxpool_padding=1, maxpool_ceil_mode=False, inplace=True, dilation=1):
        super(ConvBasicBlock, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                    padding=padding, bias=bias, dilation=dilation)
        # todo see performace diff with/without eval() in dp and bn case
        # todo (BEN) batchnorm initialization
        if batch_norm: self.batchnorm = torch.nn.BatchNorm2d(out_channels)
        if nonlinear: self.relu = torch.nn.ReLU(inplace=inplace)
        if spatial_dp: self.sdp = torch.nn.Dropout2d()  # spatial dropout
        if maxpooling: self.maxpool = torch.nn.MaxPool2d(kernel_size=maxpool_kernel_size, stride=maxpool_stride,
                                                         padding=maxpool_padding, ceil_mode=maxpool_ceil_mode)

        self.batch_norm = batch_norm
        self.nonlinear = nonlinear
        self.spatial_dp = spatial_dp
        self.maxpooling = maxpooling

    def forward(self, x):
        out = self.conv(x)
        if self.batch_norm: out = self.batchnorm(out)
        if self.nonlinear: out = self.relu(out)
        if self.spatial_dp: out = self.sdp(out)
        if self.maxpooling: out = self.maxpool(out)
        return out


class CNNBasicBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, batch_norm=False, spatial_dp=False, resnet=False,
                 bias=False, dilation=1):
        super(CNNBasicBlock, self).__init__()

        # residual: conv3*3 --> (spatial_dp) --> (batchnorm) --> relu --> cov3*3 --> (spatial_dp) --> (batchnorm)
        # shortcut: conv1*1 --> (spatial_dp) --> (batchnorm)  (optional for resnet)
        # out: relu(residual + [shortcut])

        self.conv1 = ConvBasicBlock(in_channels, out_channels, 3, stride, 1, bias, True, batch_norm, spatial_dp,
                                    dilation=dilation)
        self.conv2 = ConvBasicBlock(out_channels, out_channels, 3, 1, 1, bias, False, batch_norm, spatial_dp,
                                    dilation=dilation)
        self.relu = torch.nn.ReLU(inplace=True)
        if (in_channels != out_channels or stride != 1) and resnet:
            self.downsample = ConvBasicBlock(in_channels, out_channels, 1, stride, 0, bias, False, batch_norm,
                                             spatial_dp)
            self.downsampling = True
        else:
            self.downsampling = False
        self.resnet = resnet

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsampling:
            residual = self.downsample(residual)

        if self.resnet:
            out += residual
        out = self.relu(out)
        return out


class VGG(torch.nn.Module):

    # default is vgg
    # VGG:  Vgg()
    # add Residual connection: (bias=False, resnet=True)

    def __init__(self, batch_norm=False, spatial_dp=False, ceil_mode=False, in_channels=1, bias=True, resnet=False,
                 dilation=1):
        super(VGG, self).__init__()

        # self.conv0 = ConvBasicBlock(in_channels, 16, 1, 1, 0, False, True, batch_norm, spatial_dp)
        self.resblock1 = CNNBasicBlock(in_channels, 64, batch_norm=batch_norm, spatial_dp=spatial_dp, resnet=resnet,
                                       bias=bias, dilation=dilation)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=ceil_mode)

        self.resblock2 = CNNBasicBlock(64, 128, batch_norm=batch_norm, spatial_dp=spatial_dp, resnet=resnet, bias=bias,
                                       dilation=dilation)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=ceil_mode)
        self.in_channels = in_channels
        self.ceil_mode = ceil_mode
        self.dilation = dilation

    def forward(self, xs, ilens):
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))

        # xs: input: utt x frame x dim
        # xs: output: utt x 1 (input channel num) x frame x dim
        xs = xs.contiguous().view(xs.size(0), xs.size(1), self.in_channels, xs.size(2) // self.in_channels).transpose(1,
                                                                                                                      2)

        # xs = self.conv0(xs)
        xs = self.resblock1(xs)
        xs = self.maxpool1(xs)
        xs = self.resblock2(xs)
        xs = self.maxpool2(xs)

        # change ilens accordingly
        # in maxpooling layer: stride(2), padding(0), kernel(2)

        fnc = np.ceil if self.ceil_mode else np.floor

        # s, p, k = [2, 0, 2]
        # ilens = np.array(
        #     fnc(((np.array(ilens, dtype=np.float32)+2*p-k)/s)+1), dtype=np.int64)
        # ilens = np.array(
        #     fnc(((np.array(ilens, dtype=np.float32)+2*p-k)/s)+1), dtype=np.int64).tolist()

        s, p, k = [1, self.dilation, 3];
        ilens = np.array(fnc(((np.array(ilens, dtype=np.float32) + 2 * p - self.dilation * (k - 1) - 1) / s) + 1),
                         dtype=np.int64)
        s, p, k = [1, self.dilation, 3];
        ilens = np.array(fnc(((np.array(ilens, dtype=np.float32) + 2 * p - self.dilation * (k - 1) - 1) / s) + 1),
                         dtype=np.int64)
        s, p, k = [2, 0, 2];
        ilens = np.array(fnc(((np.array(ilens, dtype=np.float32) + 2 * p - k) / s) + 1), dtype=np.int64)

        s, p, k = [1, self.dilation, 3];
        ilens = np.array(fnc(((np.array(ilens, dtype=np.float32) + 2 * p - self.dilation * (k - 1) - 1) / s) + 1),
                         dtype=np.int64)
        s, p, k = [1, self.dilation, 3];
        ilens = np.array(fnc(((np.array(ilens, dtype=np.float32) + 2 * p - self.dilation * (k - 1) - 1) / s) + 1),
                         dtype=np.int64)
        s, p, k = [2, 0, 2];
        ilens = np.array(fnc(((np.array(ilens, dtype=np.float32) + 2 * p - k) / s) + 1), dtype=np.int64).tolist()

        # x: utt_list of frame (remove zeropaded frames) x (input channel num x dim)
        xs = xs.transpose(1, 2)

        xs = xs.contiguous().view(
            xs.size(0), xs.size(1), xs.size(2) * xs.size(3))

        xs = [xs[i, :ilens[i]] for i in range(len(ilens))]
        xs = pad_list(xs, 0.0)

        return xs, ilens



class Enc2AttAdd(torch.nn.Module):
    '''add attention with 2 encoder streams
    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int k : attention dimension
    :param int num_enc: # of encoder stream
    :param int l2_dropout: flag to apply level-2 dropout
    :param int l2_weight: fix the first encoder with l2_weight, second one has 1-l2_weight. default: None, do adaptive l2_weight learning.
    '''

    # __init__ will define the architecture of the attention model
    # chnage the l2Weight when recog
    def __init__(self, eprojs, dunits, att_dim, num_enc=2):
        super(Enc2AttAdd, self).__init__()

        self.num_enc = num_enc

        # att1
        self.mlp_enc_att1 = torch.nn.Linear(eprojs, att_dim)
        self.mlp_dec_att1 = torch.nn.Linear(dunits, att_dim, bias=False)
        self.gvec_att1 = torch.nn.Linear(att_dim, 1)
        self.dunits_att1 = dunits
        self.eprojs_att1 = eprojs
        self.att_dim_att1 = att_dim
        self.h_length_att1 = None
        self.enc_h_att1 = None
        self.pre_compute_enc_h_att1 = None
        self.mask_att1 = None
        self.store_hs_att1 = True

        # att2
        self.mlp_enc_att2 = torch.nn.Linear(eprojs, att_dim)
        self.mlp_dec_att2 = torch.nn.Linear(dunits, att_dim, bias=False)
        self.gvec_att2 = torch.nn.Linear(att_dim, 1)
        self.dunits_att2 = dunits
        self.eprojs_att2 = eprojs
        self.att_dim_att2 = att_dim
        self.h_length_att2 = None
        self.enc_h_att2 = None
        self.pre_compute_enc_h_att2 = None
        self.mask_att2 = None
        self.store_hs_att2 = True

        # att3
        self.mlp_enc_att3 = torch.nn.Linear(eprojs, att_dim)
        self.mlp_dec_att3 = torch.nn.Linear(dunits, att_dim, bias=False)
        self.gvec_att3 = torch.nn.Linear(att_dim, 1)
        self.dunits_att3 = dunits
        self.eprojs_att3 = eprojs
        self.att_dim_att3 = att_dim
        self.h_length_att3 = None
        self.enc_h_att3 = None
        self.pre_compute_enc_h_att3 = None
        self.mask_att3 = None
        self.store_hs_att3 = False



    def reset(self):
        '''reset states'''
        # att1
        self.h_length_att1 = None
        self.enc_h_att1 = None
        self.pre_compute_enc_h_att1 = None
        self.mask_att1 = None

        # att2
        self.h_length_att2 = None
        self.enc_h_att2 = None
        self.pre_compute_enc_h_att2 = None
        self.mask_att2 = None

        # att3
        self.h_length_att3 = None
        self.enc_h_att3 = None
        self.pre_compute_enc_h_att3 = None
        self.mask_att3 = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev, scaling=2.0):

        batch_att1 = len(enc_hs_pad[0])
        batch_att2 = len(enc_hs_pad[1])
        batch_att3 = len(enc_hs_pad[0])

        # if dec_z is None: # TODO make a difference
        #     dec_z_att1 =enc_hs_pad[0].new_zeros(batch_att1, self.dunits_att1)
        # else:
        #     dec_z_att1= dec_z.view(batch_att1, self.dunits_att1)
        # if dec_z is None:
        #     dec_z_att2 = enc_hs_pad[1].new_zeros(batch_att2, self.dunits_att2)
        # else:
        #     dec_z_att2 = dec_z.view(batch_att2, self.dunits_att2)
        # if dec_z is None:
        #     dec_z_att3 = enc_hs_pad[1].new_zeros(batch_att3, self.dunits_att3)
        # else:
        #     dec_z_att3 = dec_z.view(batch_att3, self.dunits_att3)


        if dec_z is None:
            dec_z_att1 = enc_hs_pad[0].new_zeros(batch_att1, self.dunits_att1)
            dec_z_att2 = enc_hs_pad[0].new_zeros(batch_att2, self.dunits_att2)
            dec_z_att3 = enc_hs_pad[0].new_zeros(batch_att3, self.dunits_att3)
        else:
            # dec_z_att1 = dec_z.view(batch_att1, self.dunits_att1)
            # dec_z_att2 = dec_z.view(batch_att1, self.dunits_att1)
            # dec_z_att3 = dec_z.view(batch_att1, self.dunits_att1)

             dec_z_att1 = dec_z_att2 =dec_z_att3 = dec_z.view(batch_att1, self.dunits_att1)

        # att1

        if self.pre_compute_enc_h_att1 is None or not self.store_hs_att1:
            self.enc_h_att1 = enc_hs_pad[0]  # utt x frame x hdim
            self.h_length_att1 = self.enc_h_att1.size(1)
            self.pre_compute_enc_h_att1 = self.mlp_enc_att1(self.enc_h_att1)

        dec_z_tiled_att1 = self.mlp_dec_att1(dec_z_att1).view(batch_att1, 1, self.att_dim_att1)
        e_att1 = self.gvec_att1(torch.tanh(self.pre_compute_enc_h_att1 + dec_z_tiled_att1)).squeeze(2)
        if self.mask_att1 is None:
            self.mask_att1 = to_cuda(self, make_pad_mask(enc_hs_len[0]))
        e_att1.masked_fill_(self.mask_att1, -float('inf'))
        w_att1 = F.softmax(scaling * e_att1, dim=1)
        c_att1 = torch.matmul(w_att1.unsqueeze(1), self.enc_h_att1).squeeze(1)


        # att2
        if self.pre_compute_enc_h_att2 is None or not self.store_hs_att2:
            self.enc_h_att2 = enc_hs_pad[1]  # utt x frame x hdim
            self.h_length_att2 = self.enc_h_att2.size(1)
            self.pre_compute_enc_h_att2 = self.mlp_enc_att2(self.enc_h_att2)

        dec_z_tiled_att2 = self.mlp_dec_att2(dec_z_att2).view(batch_att2, 1, self.att_dim_att2)
        e_att2 = self.gvec_att2(torch.tanh(self.pre_compute_enc_h_att2 + dec_z_tiled_att2)).squeeze(2)
        if self.mask_att2 is None:
            self.mask_att2 = to_cuda(self, make_pad_mask(enc_hs_len[1]))
        e_att2.masked_fill_(self.mask_att2, -float('inf'))
        w_att2 = F.softmax(scaling * e_att2, dim=1)
        c_att2 = torch.matmul(w_att2.unsqueeze(1), self.enc_h_att2).squeeze(1)


        # connection
        att_c_list = [c_att1, c_att2]
        enc_hs_pad_att3 = torch.stack(att_c_list, dim=1)
        enc_hs_len_att3 = [self.num_enc] * batch_att2

        # att3
        if self.pre_compute_enc_h_att3 is None or not self.store_hs_att3:
            self.enc_h_att3 = enc_hs_pad_att3  # utt x frame x hdim
            self.h_length_att3 = self.enc_h_att3.size(1)
            self.pre_compute_enc_h_att3 = self.mlp_enc_att3(self.enc_h_att3)
        dec_z_tiled_att3 = self.mlp_dec_att3(dec_z_att3).view(batch_att3, 1, self.att_dim_att3)
        e_att3 = self.gvec_att3(torch.tanh(self.pre_compute_enc_h_att3 + dec_z_tiled_att3)).squeeze(2)
        if self.mask_att3 is None:
            self.mask_att3 = to_cuda(self, make_pad_mask(enc_hs_len_att3))
        e_att3.masked_fill_(self.mask_att3, -float('inf'))
        w_att3 = F.softmax(scaling * e_att3, dim=1)
        c_att3 = torch.matmul(w_att3.unsqueeze(1), self.enc_h_att3).squeeze(1)

        return c_att3, [[w_att1, w_att2], w_att3]
