#!/usr/bin/env python

# Copyright 2019 Johns Hopkins University (Ruizhi Li)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import logging
import os
import csv

import random


EPS = 1e-10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--thd', type=float, default=0.8,
                        help='Threshold to select mics with high cross correlation')
    parser.add_argument('--outdir', type=str, default='/Users/ben_work',
                        help='Output directory')
    parser.add_argument('--csv', type=str, default='/Users/ben_work/mic_error.csv',
                        help='filename of mic_error.csv')


    args = parser.parse_args()

    # seed for random
    random.seed(1)

    # speaker ids in train amd condition filter
    spk_list = ['F02','F03','M01','M02']
    cond_list = ['STR', 'BUS', 'PED', 'CAF']


    # logging info
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

    # load mic_error.csv
    outlist=[]
    with open(args.csv) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(", ".join(row))
                line_count += 1
            else:
                spkid = row[0][:3]
                condid = row[0][-3:]
                if spkid in spk_list and condid in cond_list:
                    scores = [float(s) for s in row[1:]]
                    chs_filtered = [idx+1 for idx, s in enumerate(scores) if s>args.thd]
                    if len(chs_filtered) <1:
                        # no single channel is above thd, pick the highest one
                        ch_selected = scores.index(max(scores))+1
                    else:
                        ch_selected = random.choice(chs_filtered)
                    line = '{}.CH{}_REAL'.format(row[0], ch_selected)
                    outlist+=[line]
                    line_count += 1
        print('Processed {} utterance.'.format(line_count-1))

    # chech direcitory
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # output format F02_011C0204_STR.CH1_REAL
    with open('{}/tr05_real_1ch_track.list'.format(args.outdir),'w') as f:
        for l in outlist:
            f.write('{}\n'.format(l))


if __name__ == "__main__":
    main()
