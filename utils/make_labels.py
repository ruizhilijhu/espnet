#!/usr/bin/env python

# Copyright 2019 Johns Hopkins University (Ruizhi Li)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import logging
import os
import csv
import re

import random


EPS = 1e-10


def main():
    parser = argparse.ArgumentParser(description="Make labels for PM.")
    parser.add_argument('--infile', type=str, default="/Users/ben_work/result.wrd.txt",
                        help='result.txt or result.wrd.txt')
    parser.add_argument('--err', default='cer', choices=['cer','wer'],
                        help='mode to report CER or WER.')
    parser.add_argument('--outdir', type=str, default='/Users/ben_work',
                        help='Output directory')
    parser.add_argument('--prefix', type=str, default='',
                        help='prefix to output files')
    parser.add_argument('--capital-uttid', default= False, action='store_true', help='Capitalize the utterance name')
    parser.add_argument('--capital-first-uttid', default= False, action='store_true', help='Capitalize the first letter of utterance name')

    parser.add_argument('--exclude-utts-list', type=str, default='/Users/ben_work/PycharmProjects/espnet_one_stream/egs/wsj_aurora4_chime4/asr1/local/pm_exclude_utts.list',
                        help='list of excluded utterance list. In result.txt, [Scores:] are different from [Eval:].')

    args = parser.parse_args()


    # logging info
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

    # load file
    uttnames = []
    csdis = []
    csdirates = []
    csdiseqs = []
    errs = []

    line_count_id = 0
    line_count_Scores = 0


    if args.exclude_utts_list != '':
        with open(args.exclude_utts_list) as fh:
            excl_utts = [utt.strip() for utt in fh]



    with open(args.infile) as f:
        for line in f:
            if line.startswith("id:"):
                line_id = line.strip().split('-',1)[1][:-1]
                if args.capital_uttid:
                    line_id = line_id.upper()
                if args.capital_first_uttid:
                    line_id = line_id[0].upper()+line_id[1:]
                # append
                uttnames.append(line_id)
                line_count_id += 1
                # print line_id
            elif line.startswith("Scores:"):
                line_Scores = tuple(int(i) for i in line.split()[-4:])
                C, S, D, I = line_Scores
                csdirate = tuple(i*1.0/(C + S + D)  for i in line_Scores)
                # append
                csdis.append(line_Scores)
                csdirates.append(csdirate)
                errs.append((S+D+I)*1.0/(C + S + D))
                line_count_Scores += 1
            elif line.startswith("REF:"):
                line_REF_orig = line.strip()
            elif line.startswith("HYP:"):
                line_HYP_orig = line.strip()
            elif line.startswith("Eval:"):
                line_Eval_orig = line.strip()
                ali_idxs = [6]
                min_len = min(len(line_REF_orig), len(line_HYP_orig))
                csdiseq = []

                for i in range(7, min_len):  # first 6 is prefix e.g. "REF:  ", "HYP:  ", "Eval: "
                    if line_REF_orig[i - 1] == line_HYP_orig[i - 1] == ' ':
                        if not(i <= len(line_Eval_orig) and line_Eval_orig[i-1] != ' '):
                            ali_idxs += [i]
                ali_idxs_1 = [idx for idx in ali_idxs if idx < len(line_Eval_orig)]
                for i in ali_idxs:
                    if i in ali_idxs_1:
                        s = 'C' if line_Eval_orig[i] == ' ' else line_Eval_orig[i]
                        csdiseq.append(s)
                    else:
                        csdiseq.append('C')

                if line_id in excl_utts:
                    del uttnames[-1], csdis[-1], csdirates[-1], errs[-1]
                else:
                    assert csdiseq.count('C') == C
                    assert csdiseq.count('S') == S
                    assert csdiseq.count('I') == I
                    assert csdiseq.count('D') == D
                    # append
                    csdiseqs.append(csdiseq)
        assert line_count_id == line_count_Scores
        print('Processed {} utterance.'.format(line_count_Scores))


    # chech direcitory
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # writing utt2wer or utt2cer
    with open('{}/{}utt2{}'.format(args.outdir, args.prefix, args.err),'w') as f:
        for idx, uttname in enumerate(uttnames):
            f.write('{} {:.8f}\n'.format(uttname, errs[idx]))

    # writing utt2csdi
    with open('{}/{}utt2csdi{}'.format(args.outdir, args.prefix, args.err),'w') as f:
        for idx, uttname in enumerate(uttnames):
            f.write('{} {} {} {} {}\n'.format(uttname, csdis[idx][0], csdis[idx][1], csdis[idx][2], csdis[idx][3]))

    # writing utt2csdirate
    with open('{}/{}utt2csdirate{}'.format(args.outdir, args.prefix, args.err),'w') as f:
        for idx, uttname in enumerate(uttnames):
            f.write('{} {:.8f} {:.8f} {:.8f} {:.8f}\n'.format(uttname, csdirates[idx][0], csdirates[idx][1], csdirates[idx][2], csdirates[idx][3]))

    # writing utt2csdiseq
    with open('{}/{}utt2csdiseq{}'.format(args.outdir, args.prefix, args.err),'w') as f:
        for idx, uttname in enumerate(uttnames):
            f.write('{} {}\n'.format(uttname, ''.join(csdiseqs[idx])))


if __name__ == "__main__":
    main()
