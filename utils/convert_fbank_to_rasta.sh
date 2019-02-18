#!/bin/bash

# Copyright 2013  The Shenzhen Key Laboratory of Intelligent Media and Speech,
#                 PKU-HKUST Shenzhen Hong Kong Institution (Author: Wei Shi)
#           2016  Johns Hopkins University (author: Daniel Povey)
# Apache 2.0
# Combine filterbank and pitch features together
# Note: This file is based on make_fbank.sh and make_pitch_kaldi.sh

# Begin configuration section.
nj=4
cmd=run.pl
compress=true
pitchdims=3
sigma=6
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -lt 2 ] || [ $# -gt 4 ]; then
   echo "Usage: $0 [options] <fbank-data-dir> <rasta-data-dir> [<log-dir> [<rasta-dir>] ]";
   echo "e.g.: $0 data/train data-rasta/train exp/convert_fbank_to_rasta/train ratsa"
   echo "Note: <log-dir> defaults to <rasta-data-dir>/log, and <rasta-dir> defaults to <rasta-data-dir>/data"
   echo "Options: "
   echo "  --pitchdims                <pitch-dims>              # pitch dimensions to avoid filtering "
   echo "  --sigma                    <sigma>                   # standard deviation of gaussian"
   echo "  --nj                       <nj>                      # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>)     # how to run jobs."
   exit 1;
fi

fbankdata=$1
rastadata=$2
if [ $# -ge 3 ]; then
  logdir=$3
else
  logdir=$rastadata/log
fi
if [ $# -ge 4 ]; then
  rasta_dir=$4
else
  rasta_dir=$rastadata/data
fi


# make $fbank_pitch_dir an absolute pathname.
rasta_dir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $rasta_dir ${PWD}`

# use "name" as part of name of the archive.
name=`basename $fbankdata`


utils/copy_data_dir.sh $fbankdata $rastadata
rm -r $rastadata/feats.scp

mkdir -p $rasta_dir || exit 1;
mkdir -p $logdir || exit 1;


scp=$fbankdata/feats.scp

required="$scp"

for f in $required; do
  if [ ! -f $f ]; then
    echo "$0: no such file $f"
    exit 1;
  fi
done


for n in $(seq $nj); do
  # the next command does nothing unless $fbank_pitch_dir/storage/ exists, see
  # utils/create_data_link.pl for more info.
  utils/create_data_link.pl $rasta_dir/raw_rasta_$name.$n.ark
done



split_scps=""
for n in $(seq $nj); do
split_scps="$split_scps $logdir/fbank_feats.$n.scp"
done

utils/split_scp.pl $scp $split_scps || exit 1;


$cmd JOB=1:$nj $logdir/convert_fbank_to_rasta_${name}.JOB.log \
python ../../../utils/convert_fbank_to_rasta.py --fbankscp $logdir/fbank_feats.JOB.scp \
--rastascp $rasta_dir/rasta_$name.JOB.scp --rastaark $rasta_dir/rasta_$name.JOB.ark \
--sigma $sigma --exclude-last-dims $pitchdims || exit 1


# concatenate the .scp files together.
for n in $(seq $nj); do
  cat $rasta_dir/rasta_$name.$n.scp || exit 1;
done > $rastadata/feats.scp



rm $logdir/fbank_feats.*.scp  2>/dev/null

nf=`cat $rastadata/feats.scp | wc -l`
nu=`cat $rastadata/utt2spk | wc -l`
if [ $nf -ne $nu ]; then
  echo "It seems not all of the feature files were successfully processed ($nf != $nu);"
  echo "consider using utils/fix_data_dir.sh $rastadata"
fi

utils/fix_data_dir.sh $rastadata

echo "Succeeded convert fbank to rasta for $name"
