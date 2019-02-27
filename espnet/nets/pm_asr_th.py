#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Ruizhi Li)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


from __future__ import division

import argparse
import logging
import math
import sys

from argparse import Namespace

import chainer
import numpy as np
import random
import six
import torch
import torch.nn.functional as F

from chainer import reporter
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


PMERR_LOSS_THRESHOLD = PMCLASS_LOSS_THRESHOLD = 10000
CTC_SCORING_RATIO = 1.5
MAX_DECODER_OUTPUT = 5

from espnet.nets.e2e_asr_th import to_cuda, make_pad_mask
from espnet.nets.e2e_asr_th import pad_list

# ------------- Utility functions --------------------------------------------------------------------------------------
class Reporter(chainer.Chain):
    def reportPmErr(self, loss):
        reporter.report({'loss': loss}, self)

    def reportPmClass(self, loss, acc, err):
        reporter.report({'loss': loss}, self)
        reporter.report({'acc': acc}, self)
        reporter.report({'err': err}, self)


# ------------- PMERR Network ----------------------------------------------------------------------------------------
class PmErrLoss(torch.nn.Module):
    """PmErr learning loss module

    :param torch.nn.Module predictor: E2E model instance
    """

    def __init__(self, predictor):
        super(PmErrLoss, self).__init__()
        self.loss = None
        self.predictor = predictor
        self.reporter = Reporter()

    def forward(self, xs_pad, ilens, ys_pad):
        '''loss forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :return: loss value
        :rtype: torch.Tensor
        '''
        self.loss = None
        self.loss = self.predictor(xs_pad, ilens, ys_pad)
        loss_data = float(self.loss)
        if loss_data < PMERR_LOSS_THRESHOLD and not math.isnan(loss_data):
            self.reporter.reportPmErr(loss_data)
        else:
            logging.warning('loss (=%f) is not correct', loss_data)

        return self.loss


class PMERR(torch.nn.Module):
    """PMERR module

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param namespace args: argument namespace containing options
    """

    def __init__(self, idim, args):
        super(PMERR, self).__init__()
        self.model_type = args.model_type
        self.verbose = args.verbose
        self.outdir = args.outdir
        self.loss_type = args.loss_type

        # subsample info
        # +1 means input (+1) and layers outputs (args.elayer)
        subsample = np.ones(args.blayers + 1, dtype=np.int)
        if args.model_type in ['BlstmpAvgFwd']:
            ss = args.subsample.split("_")
            for j in range(min(args.blayers + 1, len(ss))):
                subsample[j] = int(ss[j])
        else:
            logging.warning(
                'Subsampling is not performed for vgg*. It is performed in max pooling layers at CNN.')
        logging.info('subsample: ' + ' '.join([str(x) for x in subsample]))
        self.subsample = subsample

        # model architecture
        self.pmodel = PmErrModel(args.model_type, idim, args.blayers, args.bunits, args.bprojs, self.subsample, args.flayers, args.funits)

        # weight initialization
        self.init_like_chainer()

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


    def forward(self, xs_pad, ilens, ys_pad):
        '''PMERR forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, 1)
        :return: loss value
        :rtype: torch.Tensor
        '''

        hs_pad = self.pmodel(xs_pad, ilens)

        # loss
        if self.loss_type == 'bceloss':
            # hs_pad (B,1) --> hs_pad (B), ys_pad (B,1) --> ys_pad (B)
            loss = F.binary_cross_entropy_with_logits(hs_pad.squeeze(1), ys_pad.squeeze(1))
        elif self.loss_type == 'mseloss':
            # hs_pad (B,1) --> hs_pad (B), ys_pad (B,1) --> ys_pad (B)
            loss = F.mse_loss(F.hardtanh(hs_pad.squeeze(1), min_val=0., max_val=1.), ys_pad.squeeze(1))
        else:
            logging.error(
                "Error: need to specify an appropriate loss")
            sys.exit()
        return loss

    def recognize(self, x):
        '''PM ERR Decoding

        :param ndarray x: input acouctic feature (T, D)
        :return: decoding results
        :rtype: float y
        '''
        prev = self.training
        self.eval()
        # subsample frame
        x = x[::self.subsample[0], :]
        ilen = [x.shape[0]]
        h = to_cuda(self, torch.from_numpy(
            np.array(x, dtype=np.float32)))

        # make a utt list (1) to use the same interface for encoder
        h = h.contiguous()
        h = self.pmodel(h.unsqueeze(0), ilen)

        # decode the first utterance
        if self.loss_type == 'bceloss':
            y = torch.sigmoid(h)
        elif self.loss_type == 'mseloss':
            y = F.hardtanh(h, min_val=0., max_val=1.)
        else:
            logging.error(
                "Error: need to specify an appropriate loss for decoding")
            sys.exit()

        if prev:
            self.train()
        return y.tolist()

    def recognize_batch(self, xs):
        '''PM ERR Decoding
        :param ndarray x: input acouctic feature (N, T, D)
        :return: decoding results
        :rtype: list ys
        '''
        prev = self.training
        self.eval()
        # subsample frame
        xs = [xx[::self.subsample[0], :] for xx in xs]
        ilens = np.fromiter((xx.shape[0] for xx in xs), dtype=np.int64)
        hs = [to_cuda(self, torch.from_numpy(np.array(xx, dtype=np.float32)))
              for xx in xs]

        xpad = pad_list(hs, 0.0)
        hpad = self.pmodel(xpad, ilens)

        # 2. decoder
        if self.loss_type == 'bceloss':
            ys = torch.sigmoid(hpad)
        elif self.loss_type == 'mseloss':
            ys = F.hardtanh(hpad, min_val=0., max_val=1.)
        else:
            logging.error(
                "Error: need to specify an appropriate loss for decoding")
            sys.exit()

        if prev:
            self.train()
        return ys.tolist()


class PmErrModel(torch.nn.Module):
    '''PMERR module

    :param str etype: type of encoder network
    :param int idim: number of dimensions of encoder network
    :param int elayers: number of layers of encoder network
    :param int eunits: number of lstm units of encoder network
    :param int epojs: number of projection units of encoder network
    :param list subsample: list of subsampling numbers
    :param float dropout: dropout rate
    :param int in_channel: number of input channels
    '''

    def __init__(self, model_type, idim, blayers, bunits, bprojs, subsample, flayers, funits):
        super(PmErrModel, self).__init__()

        if model_type == 'BlstmpAvgFwd':
            self.pmodel1 = BLSTMPAVGFWD(idim, 1,  blayers, bunits,
                               bprojs, subsample, flayers, funits)
            logging.info('BLSTMP + Avg-States + FeedForward')
        else:
            logging.error(
                "Error: need to specify an appropriate PM ERR archtecture")
            sys.exit()

        self.model_type = model_type

    def forward(self, xs_pad, ilens):
        '''PMEER forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, D)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :return: batch of hidden state sequences (B, 1)
        :rtype: torch.Tensor
        '''
        if self.model_type in ['BlstmpAvgFwd']:
            xs_pad = self.pmodel1(xs_pad, ilens)
        else:
            logging.error(
                "Error: need to specify an appropriate PM archtecture")
            sys.exit()

        return xs_pad


class BLSTMPAVGFWD(torch.nn.Module):
    """Bidirectional LSTM with projection layer +
    averaged hidden states from last BLSTMp layer feeds into feedforward network.

    :param int idim: dimension of inputs
    :param int blayers: number of blstm layers
    :param int bunits: number of lstm units (resulted in cdim * 2 due to biderectional)
    :param int bprojs: number of projection units
    :param list subsample: list of subsampling numbers
    :param int flayers: number of feedforward layers
    :param int funits: number of feedforward units
    """

    def __init__(self, idim, odim, blayers, bunits, bprojs, subsample, flayers, funits):
        super(BLSTMPAVGFWD, self).__init__()
        for i in six.moves.range(blayers):
            if i == 0:
                inputdim = idim
            else:
                inputdim = bprojs
            setattr(self, "bilstm%d" % i, torch.nn.LSTM(inputdim, bunits,
                                                        num_layers=1, bidirectional=True, batch_first=True))
            # bottleneck layer to merge
            setattr(self, "bt%d" % i, torch.nn.Linear(2 * bunits, bprojs))

        for i in six.moves.range(flayers):
            if i == 0:
                inputdim = bprojs
            else:
                inputdim = funits
            setattr(self, "linear%d" % i, torch.nn.Linear(inputdim, funits))
        setattr(self, "linear%d" % flayers, torch.nn.Linear(funits, odim))

        self.blayers = blayers
        self.flayers = flayers
        self.subsample = subsample

    def forward(self, xs_pad, ilens):
        '''BLSTMPAVGFWD forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :return: batch of hidden state sequences (B, 1)
        :rtype: torch.Tensor
        '''

        for layer in six.moves.range(self.blayers):
            xs_pack = pack_padded_sequence(xs_pad, ilens, batch_first=True)
            bilstm = getattr(self, 'bilstm' + str(layer))
            bilstm.flatten_parameters()
            ys, _ = bilstm(xs_pack)
            # ys: utt list of frame x (cdim x 2) (2: means bidirectional)
            ys_pad, ilens = pad_packed_sequence(ys, batch_first=True)
            sub = self.subsample[layer + 1]
            if sub > 1:
                ys_pad = ys_pad[:, ::sub]
                ilens = torch.LongTensor([int(i + 1) // sub for i in ilens])
            # (sum _utt frame_utt) x dim
            projected = getattr(self, 'bt' + str(layer)
                                )(ys_pad.contiguous().view(-1, ys_pad.size(2)))
            xs_pad = torch.tanh(projected.view(ys_pad.size(0), ys_pad.size(1), -1))

        mask = to_cuda(self, make_pad_mask(ilens).unsqueeze(-1))
        xs_pad = xs_pad.masked_fill(mask, 0.0),
        # xs_pad: (B, Tmax, dim); xs_sum: (B, dim)
        xs_sum = torch.sum(xs_pad[0].contiguous(), dim=1)
        ilens = to_cuda(self, ilens.unsqueeze(1).type(torch.FloatTensor))

        xs_avg = xs_sum / ilens

        for layer in six.moves.range(self.flayers):
            linear = getattr(self, 'linear' + str(layer))
            ys_avg = linear(xs_avg)
            xs_avg = F.relu(ys_avg)

        linear = getattr(self, 'linear' + str(self.flayers))
        # ys_avg  tensor (B,1)
        ys_avg = linear(xs_avg)
        return ys_avg



# ------------- PMCLASS Network ----------------------------------------------------------------------------------------
class PmClassLoss(torch.nn.Module):
    """PmClass learning loss module

    :param torch.nn.Module predictor: E2E model instance
    """

    def __init__(self, predictor):
        super(PmClassLoss, self).__init__()
        self.loss = None
        self.accuracy = None
        self.predictor = predictor
        self.reporter = Reporter()

    def forward(self, xs_pad, ilens, ys_pad):
        '''loss forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :return: loss value
        :rtype: torch.Tensor
        '''
        self.loss = None
        self.loss, acc, err = self.predictor(xs_pad, ilens, ys_pad)
        loss_data = float(self.loss)
        if loss_data < PMCLASS_LOSS_THRESHOLD and not math.isnan(loss_data):
            self.reporter.reportPmClass(loss_data, acc, err)
        else:
            logging.warning('loss (=%f) is not correct', loss_data)

        return self.loss


class PMCLASS(torch.nn.Module):
    """PMCLASS module

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param namespace args: argument namespace containing options
    """

    def __init__(self, idim, odim, args):
        super(PMCLASS, self).__init__()
        self.model_type = args.model_type
        self.verbose = args.verbose
        self.label_list = args.label_list
        self.outdir = args.outdir
        self.loss_type = args.loss_type

        # subsample info
        # +1 means input (+1) and layers outputs (args.elayer)
        subsample = np.ones(args.blayers + 1, dtype=np.int)
        if args.model_type in ['BlstmpAvgFwd']:
            ss = args.subsample.split("_")
            for j in range(min(args.blayers + 1, len(ss))):
                subsample[j] = int(ss[j])
        else:
            logging.warning(
                'Subsampling is not performed for vgg*. It is performed in max pooling layers at CNN.')
        logging.info('subsample: ' + ' '.join([str(x) for x in subsample]))
        self.subsample = subsample

        # model architecture
        self.pmodel = PmClassModel(args.model_type, idim, odim, args.blayers, args.bunits, args.bprojs, self.subsample, args.flayers, args.funits)

        # weight initialization
        self.init_like_chainer()

        if 'report_err' in vars(args) and args.report_err:
            self.report_err = args.report_err
        else:
            self.report_err = args.report_err

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


    def forward(self, xs_pad, ilens, ys_pad):
        '''PMCLASS forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, 1)
        :return: loss value
        :rtype: torch.Tensor
        '''

        hs_pad = self.pmodel(xs_pad, ilens)

        # loss
        if self.loss_type == 'xentloss':
            # hs_pad: tensor(B, odim); ys_pad: tensor(B, 1)
            loss = F.cross_entropy(hs_pad, ys_pad.squeeze(1))
        else:
            logging.error(
                "Error: need to specify an appropriate loss")
            sys.exit()

        if self.training:
            acc, err = None, 0.0
        else:
            y = F.softmax(hs_pad, dim=1)
            _, y = torch.topk(y, 1, dim=1)
            acc = ys_pad.squeeze(1) == y.squeeze(1)
            acc = 1.0*sum(acc.cpu().numpy())/len(acc.cpu().numpy())
            err = 1.0 - acc

        return loss, acc, err

    def recognize(self, x):
        '''PM CLASS Decoding

        :param ndarray x: input acouctic feature (T, D)
        :return: decoding results
        :rtype: float y
        '''
        prev = self.training
        self.eval()
        # subsample frame
        x = x[::self.subsample[0], :]
        ilen = [x.shape[0]]
        h = to_cuda(self, torch.from_numpy(
            np.array(x, dtype=np.float32)))

        # make a utt list (1) to use the same interface for encoder
        h = h.contiguous()
        h = self.pmodel(h.unsqueeze(0), ilen)

        # decode the first utterance
        if self.loss_type == 'xentloss':
            # h: tensor(B,odim)
            y = F.softmax(h, dim=1)
            _, y = torch.topk(y,1, dim=1)
        else:
            logging.error(
                "Error: need to specify an appropriate loss for decoding")
            sys.exit()

        if prev:
            self.train()
        return y.squeeze(1).tolist()

    def recognize_batch(self, xs):
        '''PM ERR Decoding
        :param ndarray x: input acouctic feature (N, T, D)
        :return: decoding results
        :rtype: list ys
        '''
        prev = self.training
        self.eval()
        # subsample frame
        xs = [xx[::self.subsample[0], :] for xx in xs]
        ilens = np.fromiter((xx.shape[0] for xx in xs), dtype=np.int64)
        hs = [to_cuda(self, torch.from_numpy(np.array(xx, dtype=np.float32)))
              for xx in xs]

        xpad = pad_list(hs, 0.0)
        hpad = self.pmodel(xpad, ilens)

        # 2. decoder
        if self.loss_type == 'xentloss':
            ys = F.softmax(hpad, dim=1)
            _, ys = torch.topk(ys,1, dim=1)
        else:
            logging.error(
                "Error: need to specify an appropriate loss for decoding")
            sys.exit()

        if prev:
            self.train()
        return ys.squeeze(1).tolist()


class PmClassModel(torch.nn.Module):
    '''PMCLASS module

    :param str etype: type of encoder network
    :param int idim: number of dimensions of encoder network
    :param int elayers: number of layers of encoder network
    :param int eunits: number of lstm units of encoder network
    :param int epojs: number of projection units of encoder network
    :param list subsample: list of subsampling numbers
    :param float dropout: dropout rate
    :param int in_channel: number of input channels
    '''

    def __init__(self, model_type, idim, odim, blayers, bunits, bprojs, subsample, flayers, funits):
        super(PmClassModel, self).__init__()


        if model_type == 'BlstmpAvgFwd':
            self.pmodel1 = BLSTMPAVGFWD(idim, odim, blayers, bunits,
                               bprojs, subsample, flayers, funits)
            logging.info('BLSTMP + Avg-States + FeedForward')
        else:
            logging.error(
                "Error: need to specify an appropriate PM ERR archtecture")
            sys.exit()

        self.model_type = model_type

    def forward(self, xs_pad, ilens):
        '''PMEER forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, D)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :return: batch of hidden state sequences (B, 1)
        :rtype: torch.Tensor
        '''
        if self.model_type in ['BlstmpAvgFwd']:
            xs_pad = self.pmodel1(xs_pad, ilens)
        else:
            logging.error(
                "Error: need to specify an appropriate PM archtecture")
            sys.exit()

        return xs_pad


