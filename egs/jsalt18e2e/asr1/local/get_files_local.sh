


# get all files into local



    function get_decode(){
        local outfile=$1
        local ertype=$2

        [ -f $outfile ] && rm $outfile && touch $outfile
        for expt_dir in `ls ${dir_orig} | grep "^tr_babel10"`; do

        if [ $ertype == 'wer' ];then
            d=$(grep -h -A 4 $n ${dir_orig}/$expt_dir/decode*/result.wrd.txt)
        else
            d=$(grep -h -A 4 $n ${dir_orig}/$expt_dir/decode*/result.txt)
        fi

        if [[ ! -z "$d" ]]; then
        echo "###################################################################" >> $outfile
        echo $expt_dir >> $outfile


        if [ $ertype == 'wer' ];then
            grep -h -A 4 $n ${dir_orig}/$expt_dir/decode*/result.wrd.txt >> $outfile
        else
            grep -h -A 4 $n ${dir_orig}/$expt_dir/decode*/result.txt >> $outfile
        fi

        echo "" >> $outfile
        echo "" >> $outfile
        fi
        done

        # get cer or wer
        raw=$(grep -B 2 Score $outfile)
    echo "$raw"| awk '{if(substr($1,1,10)=="tr_babel10")expt=$0;else if(substr($1,1,6)=="Scores"){c=$6;s=$7;d=$8;i=$9;print expt;print "Scores: (#C #S #D #I) "c,s,d,i; printf "ER: %2.4f\n",(s+d+i)/(s+d+c); print }}' > $dir/$ertype.${n}
    }



stage=1

. ~/.bash_profile

dir_orig=/Users/ben_work/expt_esp_orig
dir=/Users/ben_work/expt_esp

if [ $stage -le 1 ]; then
    rsync -avz --exclude '*snapshot*' --exclude '*train.log' --exclude '*model.acc.best' --exclude '*model.loss.best' --exclude '*.json' --exclude '*/decode.*.log' ruizhili@login.clsp.jhu.edu:/export/b19/ruizhili/espnet/egs/jsalt18e2e/asr1/exp_lang1/* $dir_orig

    rsync -avz --exclude '*snapshot*' --exclude '*train.log' --exclude '*model.acc.best' --exclude '*model.loss.best' --exclude '*.json' --exclude '*/decode.*.log' ruizhili@login.clsp.jhu.edu:/export/b19/ruizhili/espnet/egs/jsalt18e2e/asr1/exp/* $dir_orig

    rsync -avz --exclude '*snapshot*' --exclude '*train.log' --exclude '*model.acc.best' --exclude '*model.loss.best' --exclude '*.json' --exclude '*/decode.*.log' harish@login.clsp.jhu.edu:/export/b15/harish/espnet/egs/jsalt18e2e/asr1/exp_lang1/* $dir_orig

    rsync -avz --exclude '*snapshot*' --exclude '*train.log' --exclude '*model.acc.best' --exclude '*model.loss.best' --exclude '*.json' --exclude '*/decode.*.log' hltmbask@login.clsp.jhu.edu:/export/b19/hltmbask/espnet/egs/jsalt18e2e/asr1/exp_lang1/* $dir_orig
fi


if [ $stage -le 2 ]; then

    # acc.png loss.png
    for n in acc.png loss.png; do
    [ ! -d $dir/${n} ] && mkdir -p $dir/${n}
    for expt_dir in `ls ${dir_orig} | grep "^tr_babel10"`; do
        if [ -f ${dir_orig}/$expt_dir/results/${n} ]; then
#            echo $expt_dir
            cp ${dir_orig}/$expt_dir/results/${n} $dir/${n}/${expt_dir}.${n}
        fi
    done

    # all in one
    pngs=$(ls $dir/${n}/tr* | tr "\n" " ")
    echo $pngs
    magick montage -pointsize 9 -geometry +1+1 -label '%f' $pngs $dir/${n}/allinone.png
    done


    # att for validation utt:
    for n in 107_15460_A_20120426_224823_018187-vietnamese.ep 107_45931_A_20120322_143234_038255-vietnamese.ep; do
    [ ! -d $dir/${n} ] && mkdir -p $dir/${n}

    for expt_dir in `ls ${dir_orig} | grep "^tr_babel10"`; do
        if [ -f ${dir_orig}/$expt_dir/results/att_ws/${n}.1.png ]; then
            echo $expt_dir
            pngs=$(ls ${dir_orig}/$expt_dir/results/att_ws/${n}.* | sort -n -t . -k3 | tr "\n" " ")
            magick montage -geometry +1+1 -label '%f' $pngs -tile 5x4 $dir/${n}/${expt_dir}.png
        fi
    done
    done


    # for a given utternace, gather the decoding (word, frame) results from all expts
    for n in 107_83219_b-107_83219_b_20120421_172919_059066-vietnamese; do

        outfile=$dir/decode.${n}
        get_decode $outfile cer

        outfile=$dir/decode.wrd.${n}
        get_decode $outfile wer
    done

fi


