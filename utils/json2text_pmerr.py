#!/usr/bin/env python2
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import json
import argparse
import logging

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('json', type=str, help='json files')
    parser.add_argument('labeltype', type=str, help='label type')
    parser.add_argument('out', type=str, help='out')

    args = parser.parse_args()

    # logging info
    logging.basicConfig(level=logging.INFO, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

    logging.info("reading %s", args.json)
    with open(args.json, 'r') as f:
        j = json.load(f)


    outfile = open(args.out, 'w')

    for x in j['utts']:

        err = j['utts'][x]['output'][0][args.labeltype]
        rec_err = j['utts'][x]['output'][0]['rec_{}'.format(args.labeltype)]
        outfile.write("{} {} {}\n".format(x, err, rec_err))


