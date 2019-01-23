#!/usr/bin/env python
# encoding: utf-8

# Copyright 2019 Johns Hopkins University (Ruizhi Li)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import argparse
import logging
import os
import random
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    # general configuration
    parser.add_argument('--ngpu', default=0, type=int,
                        help='Number of GPUs')
    parser.add_argument('--backend', default='chainer', type=str,
                        choices=['chainer', 'pytorch'],
                        help='Backend library')
    parser.add_argument('--debugmode', default=1, type=int,
                        help='Debugmode')
    parser.add_argument('--seed', default=1, type=int,
                        help='Random seed')
    parser.add_argument('--verbose', '-V', default=1, type=int,
                        help='Verbose option')
    parser.add_argument('--batchsize', default=1, type=int,
                        help='Batch size for bnf extraction')
    # task related
    parser.add_argument('--feat-json', type=str,
                        help='Filename of input data (json)')
    parser.add_argument('--out', type=str, required=True,
                        help='Output filename of bnf data')
    # model (parameter) related
    parser.add_argument('--model', type=str, required=True,
                        help='Model file parameters to read')
    parser.add_argument('--model-conf', type=str, default=None,
                        help='Model config file')
    parser.add_argument('--bnf-component', type=str, required=True,
                        help='Component to extract bnf')
    args = parser.parse_args()

    # logging info
    if args.verbose == 1:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    elif args.verbose == 2:
        logging.basicConfig(level=logging.DEBUG,
                            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    else:
        logging.basicConfig(
            level=logging.WARN, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
        logging.warning("Skip DEBUG/INFO messages")

    # check CUDA_VISIBLE_DEVICES
    if args.ngpu > 0:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd is None:
            logging.warn("CUDA_VISIBLE_DEVICES is not set.")
        elif args.ngpu != len(cvd.split(",")):
            logging.error("#gpus is not matched with CUDA_VISIBLE_DEVICES.")
            sys.exit(1)

        # TODO(mn5k): support of multiple GPUs
        if args.ngpu > 1:
            logging.error("The program only supports ngpu=1.")
            sys.exit(1)

    # display PYTHONPATH
    logging.info('python path = ' + os.environ.get('PYTHONPATH', '(None)'))

    # seed setting
    random.seed(args.seed)
    np.random.seed(args.seed)
    logging.info('set random seed = %d' % args.seed)

    # bnf
    logging.info('backend = ' + args.backend)
    if args.backend == "chainer":
        # TODO: support in chainer
        logging.error("Bnf extraction in chainer is not supported yet.")
        sys.exit(1)
    elif args.backend == "pytorch":
        from espnet.asr.asr_pytorch import bnf
        bnf(args)
    else:
        raise ValueError("chainer and pytorch are only supported.")


if __name__ == '__main__':
    main()
