# coding: utf-8

import argparse
import importlib
import json
import numpy as np
import os
import pytest
import tempfile
import torch

from espnet.nets.pytorch_backend.nets_utils import pad_list


def make_arg(**kwargs):
    train_defaults = dict(
        elayers=1,
        subsample="1_2_2_1_1",
        etype="vggblstm",
        eunits=16,
        eprojs=8,
        dtype="lstm",
        dlayers=1,
        dunits=16,
        atype="location",
        aheads=2,
        awin=5,
        aconv_chans=4,
        aconv_filts=10,
        mtlalpha=0.5,
        lsm_type="",
        lsm_weight=0.0,
        sampling_probability=0.0,
        adim=16,
        dropout_rate=0.0,
        dropout_rate_decoder=0.0,
        nbest=5,
        beam_size=2,
        penalty=0.5,
        maxlenratio=1.0,
        minlenratio=0.0,
        ctc_weight=0.2,
        lm_weight=0.0,
        rnnlm=None,
        verbose=2,
        char_list=["a", "e", "i", "o", "u"],
        outdir=None,
        ctc_type="warpctc",
        report_cer=False,
        report_wer=False,
        sym_space="<space>",
        sym_blank="<blank>",
        replace_sos=False,
        tgt_lang=False,
        enc_init=None,
        enc_init_mods='enc.',
        dec_init=None,
        dec_init_mods='dec.,att.',
        model_module='espnet.nets.pytorch_backend.e2e_asr:E2E'
    )
    train_defaults.update(kwargs)

    return argparse.Namespace(**train_defaults)


def make_arg_mulenc(**kwargs):
    train_defaults = dict(
        num_encs=2,
        elayers=[1, 1],
        subsample=["1_2_2_1_1", "1_2_2_1_1"],
        etype=["vggblstm", "vggblstm"],
        eunits=[16, 16],
        eprojs=8,
        dtype="lstm",
        dlayers=1,
        dunits=16,
        atype=["location", "location"],
        aheads=[2, 2],
        awin=[5, 5],
        aconv_chans=[4, 4],
        aconv_filts=[10, 10],
        han_type="multi_head_add",
        han_heads=2,
        han_win=5,
        han_conv_chans=4,
        han_conv_filts=10,
        han_dim=16,
        mtlalpha=0.5,
        lsm_type="",
        lsm_weight=0.0,
        sampling_probability=0.0,
        adim=[16, 16],
        dropout_rate=[0.0, 0.0],
        dropout_rate_decoder=0.0,
        nbest=5,
        beam_size=2,
        penalty=0.5,
        maxlenratio=1.0,
        minlenratio=0.0,
        ctc_weight=0.2,
        lm_weight=0.0,
        rnnlm=None,
        verbose=2,
        char_list=["a", "e", "i", "o", "u"],
        outdir=None,
        ctc_type="warpctc",
        report_cer=False,
        report_wer=False,
        sym_space="<space>",
        sym_blank="<blank>",
        replace_sos=False,
        tgt_lang=False,
        share_ctc=False,
        weights_ctc_train=[0.5, 0.5],
        weights_ctc_dec=[0.5, 0.5],
    )
    train_defaults.update(kwargs)

    return argparse.Namespace(**train_defaults)


def get_default_scope_inputs():
    idim = 40
    odim = 5
    ilens = [20, 15]
    olens = [4, 3]

    return idim, odim, ilens, olens


def get_default_scope_inputs_mulenc():
    idim_list = [40, 40]
    odim = 5
    ilens_list = [[20, 15], [19, 14]]
    olens = [4, 3]

    return idim_list, odim, ilens_list, olens


def pytorch_prepare_inputs(idim, odim, ilens, olens, is_cuda=False):
    np.random.seed(1)

    xs = [np.random.randn(ilen, idim).astype(np.float32) for ilen in ilens]
    ys = [np.random.randint(1, odim, olen).astype(np.int32) for olen in olens]
    ilens = np.array([x.shape[0] for x in xs], dtype=np.int32)

    xs_pad = pad_list([torch.from_numpy(x).float() for x in xs], 0)
    ys_pad = pad_list([torch.from_numpy(y).long() for y in ys], -1)
    ilens = torch.from_numpy(ilens).long()

    if is_cuda:
        xs_pad = xs_pad.cuda()
        ys_pad = ys_pad.cuda()
        ilens = ilens.cuda()

    return xs_pad, ilens, ys_pad


def pytorch_prepare_inputs_mulenc(idim_list, odim, ilens_list, olens, is_cuda=False):
    np.random.seed(1)

    xs_list = [[np.random.randn(ilen, idim_list[idx]).astype(np.float32) for ilen in ilens]
               for idx, ilens in enumerate(ilens_list)]
    ys = [np.random.randint(1, odim, olen).astype(np.int32) for olen in olens]
    ilens_list = [np.array([x.shape[0] for x in xs], dtype=np.int32) for xs in xs_list]

    xs_pad_list = [pad_list([torch.from_numpy(x).float() for x in xs], 0) for xs in xs_list]
    ys_pad = pad_list([torch.from_numpy(y).long() for y in ys], -1)
    ilens_list = [torch.from_numpy(ilens).long() for ilens in ilens_list]

    if is_cuda:
        xs_pad_list = [xs_pad.cuda() for xs_pad in xs_pad_list]
        ys_pad = ys_pad.cuda()
        ilens_list = [ilens.cuda() for ilens in ilens_list]

    return xs_pad_list, ilens_list, ys_pad


@pytest.mark.parametrize("enc_init, enc_mods, dec_init, dec_mods, mtlalpha", [
    (None, "enc.", None, "dec., att.", 0.0),
    (None, "enc.", None, "dec., att.", 0.5),
    (None, "enc.", None, "dec., att.", 1.0),
    (True, "enc.", None, "dec., att.", 0.5),
    (None, "enc.", True, "dec., att.", 0.0),
    (None, "enc.", True, "dec., att.", 0.5),
    (None, "enc.", True, "dec., att.", 1.0),
    (True, "enc.", True, "dec., att.", 0.0),
    (True, "enc.", True, "dec., att.", 0.5),
    (True, "enc.", True, "dec., att.", 1.0),
    (True, "test", None, "dec., att.", 0.0),
    (True, "test", None, "dec., att.", 0.5),
    (True, "test", None, "dec., att.", 1.0),
    (None, "enc.", True, "test", 0.0),
    (None, "enc.", True, "test", 0.5),
    (None, "enc.", True, "test", 1.0),
    (True, "enc.enc.0", None, "dec., att.", 0.0),
    (True, "enc.enc.0", None, "dec., att.", 0.5),
    (True, "enc.enc.0", None, "dec., att.", 1.0),
    (None, "enc.", True, "dec.embed.", 0.0),
    (None, "enc.", True, "dec.embed.", 0.5),
    (None, "enc.", True, "dec.embed.", 1.0),
    (True, "enc.enc.0, enc.enc.1", True, "dec., att.", 0.0),
    (True, "enc.enc.0", True, "dec.embed.,dec.decoder.1", 0.5),
    (True, "enc.enc.0, enc.enc.1", True, "dec.embed.,dec.decoder.1", 1.0)])
def test_pytorch_trainable_transferable_and_decodable(enc_init, enc_mods, dec_init, dec_mods, mtlalpha):
    idim, odim, ilens, olens = get_default_scope_inputs()
    args = make_arg()

    module = importlib.import_module('espnet.nets.pytorch_backend.e2e_asr')
    model = module.E2E(idim, odim, args)

    batch = pytorch_prepare_inputs(idim, odim, ilens, olens)

    loss = model(*batch)
    loss.backward()

    with torch.no_grad():
        in_data = np.random.randn(20, idim)
        model.recognize(in_data, args, args.char_list)

    if not os.path.exists(".pytest_cache"):
        os.makedirs(".pytest_cache")
    utils = importlib.import_module('espnet.asr.asr_utils')

    tmppath = tempfile.mktemp()
    utils.torch_save(tmppath, model)

    if enc_init is not None:
        enc_init = tmppath
    if dec_init is not None:
        dec_init = tmppath

    # create dummy model.json for saved model to go through
    # get_model_conf(...) called in load_trained_modules method.
    model_conf = os.path.dirname(tmppath) + '/model.json'
    with open(model_conf, 'wb') as f:
        f.write(json.dumps((40, 5, vars(args)),
                           indent=4, ensure_ascii=False, sort_keys=True).encode('utf_8'))

    args = make_arg(enc_init=enc_init, enc_init_mods=enc_mods,
                    dec_init=dec_init, dec_init_mods=dec_mods,
                    mtlalpha=mtlalpha)
    transfer = importlib.import_module('espnet.asr.pytorch_backend.asr_init')
    model = transfer.load_trained_modules(40, 5, args)

    loss = model(*batch)
    loss.backward()

    with torch.no_grad():
        in_data = np.random.randn(20, idim)
        model.recognize(in_data, args, args.char_list)


@pytest.mark.parametrize("prefix_src, prefix_tgt, freeze_params, mtlalpha", [
    ('enc', 'enc.0', True, 0.0),
    ('enc', 'enc.0', True, 0.5),
    ('enc', 'enc.0', True, 1.0),
    ('enc', 'enc.0', False, 0.0),
    ('enc', 'enc.0', False, 0.5),
    ('enc', 'enc.0', False, 1.0),
    ('att.0', 'att.1', True, 0.0),
    ('att.0', 'att.1', True, 0.5),
    ('att.0', 'att.1', True, 1.0),
    ('att.0', 'att.1', False, 0.0),
    ('att.0', 'att.1', False, 0.5),
    ('att.0', 'att.1', False, 1.0),
    ('ctc', 'ctc.0', True, 0.0),
    ('ctc', 'ctc.0', True, 0.5),
    ('ctc', 'ctc.0', True, 1.0),
    ('ctc', 'ctc.0', False, 0.0),
    ('ctc', 'ctc.0', False, 0.5),
    ('ctc', 'ctc.0', False, 1.0),
    ('dec', 'dec', True, 0.0),
    ('dec', 'dec', True, 0.5),
    ('dec', 'dec', True, 1.0),
    ('dec', 'dec', False, 0.0),
    ('dec', 'dec', False, 0.5),
    ('dec', 'dec', False, 1.0)])
def test_pytorch_trainable_transferable_and_decodable_mulenc(prefix_src, prefix_tgt, freeze_params, mtlalpha):
    # 1. create a single encoder model
    idim, odim, ilens, olens = get_default_scope_inputs()
    args = make_arg()

    module = importlib.import_module('espnet.nets.pytorch_backend.e2e_asr')
    model = module.E2E(idim, odim, args)

    batch = pytorch_prepare_inputs(idim, odim, ilens, olens)

    loss = model(*batch)
    loss.backward()

    with torch.no_grad():
        in_data = np.random.randn(20, idim)
        model.recognize(in_data, args, args.char_list)

    if not os.path.exists(".pytest_cache"):
        os.makedirs(".pytest_cache")
    utils = importlib.import_module('espnet.asr.asr_utils')

    tmppath = tempfile.mktemp()
    utils.torch_save(tmppath, model)

    # 2. create a multi-encoder model using trained single model.
    # (for example, initialize encoders in multi-encoder model using encoder from single encoder model.)
    idim_list, odim, ilens_list, olens = get_default_scope_inputs_mulenc()
    args = make_arg_mulenc(mtlalpha=mtlalpha)

    module = importlib.import_module('espnet.nets.pytorch_backend.e2e_asr_mulenc')
    model = module.E2E(idim_list, odim, args)

    batch = pytorch_prepare_inputs_mulenc(idim_list, odim, ilens_list, olens)

    pretrain_conf_dict = \
        {"prefix_src": prefix_src, "prefix_tgt": prefix_tgt, "trained_model": tmppath, "freeze_params": freeze_params}

    transfer = importlib.import_module('espnet.asr.pytorch_backend.asr_init')
    transfer.load_trained_modules_mulenc(model, pretrain_conf_dict)

    loss = model(*batch)
    loss.backward()

    with torch.no_grad():
        in_data = [np.random.randn(20, idim_list[0]), np.random.randn(19, idim_list[1])]
        model.recognize(in_data, args, args.char_list)
