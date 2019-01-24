#!/bin/bash

# Copyright 2012  Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)
#           2015  Guoguo Chen
#           2016  Mitsubishi Electric Research Laboratories (Author: Shinji Watanabe)
# Apache 2.0

# This script takes data prepared in a corpus-dependent way
# in data/local/, and converts it into the "canonical" form,
# in various subdirectories of data/, e.g. data/train_si84, etc.

echo "$0 $@"  # Print the command line for logging
. utils/parse_options.sh || exit 1;

. ./path.sh || exit 1;

echo "Preparing train and test data"
srcdir=data/local/chime4_data

for x in et05_orig_clean dt05_orig_clean tr05_orig_clean; do
  mkdir -p data/chime4_${x}
  cp ${srcdir}/${x}_wav.scp data/chime4_${x}/wav.scp || exit 1;
  cp ${srcdir}/${x}.txt data/chime4_${x}/text || exit 1;
  cp ${srcdir}/${x}.spk2utt data/chime4_${x}/spk2utt || exit 1;
  cp ${srcdir}/${x}.utt2spk data/chime4_${x}/utt2spk || exit 1;
  utils/filter_scp.pl data/chime4_${x}/spk2utt ${srcdir}/spk2gender > data/chime4_${x}/spk2gender || exit 1;
done

echo "Succeeded in formatting data."
