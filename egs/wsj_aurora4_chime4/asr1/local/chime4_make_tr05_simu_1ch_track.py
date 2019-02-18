#!/usr/bin/env python

# Copyright 2019 Johns Hopkins University (Ruizhi Li)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import logging
import os

import random

EPS = 1e-10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='/Users/ben_work',
                        help='Output directory')
    parser.add_argument('--scp', type=str, default='/Users/ben_work/wav.scp',
                        help='scp file')


    args = parser.parse_args()

    # seed for random
    random.seed(1)

    # logging info
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

    # load scp
    uttlist=[]
    outlist=[]
    with open(args.scp) as scp_file:
        for line in scp_file.readlines():
            uttid = line.split(' ')[0].split('.')[0]
            if uttid not in  uttlist:
                uttlist+=[uttid]

    # chech direcitory
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # output format F02_011C0204_STR.CH1_SIMU
    line_count = 0
    with open('{}/tr05_simu_1ch_track.list'.format(args.outdir),'w') as f:
        for utt in uttlist:
            ch_selected = random.choice([1,2,3,4,5,6])
            l = '{}.CH{}_SIMU'.format(utt, ch_selected)
            f.write('{}\n'.format(l))
            line_count+=1

    print('Processed {} utterance.'.format(line_count))


if __name__ == "__main__":
    main()
