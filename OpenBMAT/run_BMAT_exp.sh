bs=$1
expdir=$2
metric=$3

declare -a encoders=(x-vectors openl3/_mean_std openl3/_delta_gap openl3/_last openl3/_max prosodic mfcc wav2vec/_mean_std wav2vec/_delta_gap wav2vec/_last wav2vec/_max crepe/_mean_std crepe/_delta_gap crepe/_last crepe/_max ecapa)

for encoder in "${encoders[@]}"
    do
        expname=${expdir}/BiLSTM_bs${bs}_${encoder}
        python ../train_fit.py -exp ${expname} -s_last -arc BiLSTM -enc ${encoder} -lr 1e-3 -hs -huss 256 -nlss 2 -diss 0 0.2 0.5 -doss 0 0.2 0.5 -data BMAT -bs ${bs} -ef BMAT_ATS1/${encoder} -lf BMAT_ATS1/labs_dict.pkl --metric ${metric} -loss FocalLoss -max 1000 -vp 0.10 -pat 50 -ar -as -split BMAT_split.json
    done
