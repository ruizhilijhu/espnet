#!/usr/bin/env python

# Copyright 2019 Johns Hopkins University (Ruizhi Li)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import logging
import os
import csv
import re

import numpy as np
import random

import kaldi_io_py

EPS = 1e-10


def main():
    parser = argparse.ArgumentParser(description="Make RASTA features from Fbank.")
    parser.add_argument('--fbankscp', type=str,
                        help='Fbank feat scp [in]')
    parser.add_argument('--rastascp', type=str,
                        help='RASTA feat scp [out]')
    parser.add_argument('--rastaark', type=str,
                        help='RASTA feat ark [out]')
    parser.add_argument('--sigma', type=float, default=7,
                        help='standard deviation of first gaussian')
    parser.add_argument('--sigma2', type=float, default=7,
                        help='standard deviation of second gaussian')
    parser.add_argument('--exclude-last-dims', type=float, default=3,
                        help='last dimemsions not to process rasta. [for pitch]')
    parser.add_argument('--gauss-mode', default='gauss1order', type=str,
                        choices=['gauss1order', 'gauss2order', '2gauss'],
                        help='Order of gaussian derivative.')



    args = parser.parse_args()

    # logging info
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

    if not os.path.exists(os.path.dirname(args.rastaark)):
        os.makedirs(os.path.dirname(args.rastaark))

    if not os.path.exists(os.path.dirname(args.rastascp)):
        os.makedirs(os.path.dirname(args.rastascp))

    # load scp
    reader = kaldi_io_py.read_mat_scp('{}'.format(args.fbankscp))

    # filter
    def pick_filter(name, sigma, sigma2):
        pts = np.array(range(-50, 51))
        if name == '1_deri':
            filter = lambda x: -x * np.exp(-0.5 * np.power(x, 2) / np.power(sigma, 2)) / np.sqrt(2 * np.pi) / np.power(
                sigma, 3)
        elif name == '2_deri':
            filter = lambda x: -(np.power(sigma, 2) - np.power(x, 2)) * np.exp(
                -0.5 * np.power(x, 2) / np.power(sigma, 2)) / np.sqrt(2 * np.pi) / np.power(sigma, 5)
        elif name == '2gauss':
            filter = lambda x: np.exp(-0.5 * np.power(x, 2) / np.power(sigma, 2)) / np.sqrt(2 * np.pi) / np.power(sigma,
                                                                                                                  2) - np.exp(
                -0.5 * np.power(x, 2) / np.power(sigma2, 2)) / np.sqrt(2 * np.pi) / np.power(sigma2, 2)
        filter = filter(pts)
        return filter

    filter = pick_filter(args.gauss_mode, args.sigma, args.sigma2)

    # extract feature and then write as ark with scp format
    cnt = 0
    ark_scp_output = 'ark:| copy-feats --compress=true ark:- ark,scp:{},{}'.format(args.rastaark, args.rastascp)
    with kaldi_io_py.open_or_fd(ark_scp_output, 'wb') as f:
        for idx, (utt_id, x_fbank) in enumerate(reader, 1):
            if x_fbank.shape[0] >= 101:
                x_rasta = np.copy(x_fbank)
                dim = x_fbank.shape[1]
                for i in range(dim-int(args.exclude_last_dims)):
                    x_rasta[:, i] = np.convolve(filter, x_fbank[:, i], mode='same')
                kaldi_io_py.write_mat(f, x_rasta, key=utt_id)
                cnt += 1


if __name__ == "__main__":
    main()
