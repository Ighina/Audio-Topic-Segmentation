bs=$1
expdir=$2
metric=$3
seg=$4

declare -a encoders=(wav2vec/_delta_gap wav2vec/_last wav2vec/_max crepe/_mean_std crepe/_delta_gap crepe/_last crepe/_max ecapa)
# x-vectors openl3/_mean_std openl3/_delta_gap openl3/_last openl3/_max prosodic mfcc wav2vec/_mean_std
# wav2vec_std wav2vec_delta_gap wav2vec_last wav2vec_max

for encoder in "${encoders[@]}"
    do
        expname=${expdir}/BiLSTM_bs${bs}_${encoder}
        python ../train_fit.py -exp ${expname} -s_last -arc BiLSTM -enc ${encoder} -lr 1e-3 -hs -huss 128 -nlss 2 -diss 0 -doss 0 -data BMAT -bs ${bs} -ef BMAT_${seg}/${encoder} -lf ${seg}_BMAT_labels/labs_dict.pkl --metric ${metric} -loss BinaryCrossEntropy -max 1000 -vp 0.10 -pat 50 -ar -kcv 100 -as
        # echo -exp ${expname} -arc SimpleBiLSTM -enc ${encoder} -lr 5e-3 -hs -huss 100 -nlss 1 -diss 0 0.2 0.4 0.6 0.8 -doss 0 0.2 0.4 0.6 0.8 -data BBCAudio -bs ${bs} -ef nonnews_${encoder}_${seg} -lf ${seg}_nonnews_labels/labs_dict.pkl --metric ${metric} -loss BinaryCrossEntropy -max 500 -vp 0.13 -pat 20
    done