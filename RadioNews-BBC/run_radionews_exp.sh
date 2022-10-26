bs=$1
expdir=$2
metric=$3
seg=$4

declare -a encoders=(x-vectors openL3/_mean_std openL3/_delta_gap openL3/_last openL3/_max prosodic mfcc Wav2Vec/_mean_std Wav2Vec/_delta_gap Wav2Vec/_last Wav2Vec/_max CREPE/_mean_std CREPE/_delta_gap CREPE/_last CREPE/_max)
# wav2vec_std wav2vec_delta_gap wav2vec_last wav2vec_max

for encoder in "${encoders[@]}"
    do
        expname=${expdir}/BiLSTM_bs${bs}_${encoder}
        python ../train_fit.py -exp ${expname} -s_last -arc BiLSTM -enc ${encoder} -lr 1e-3 -hs -huss 128 -nlss 2 -diss 0 -doss 0 -data RadioNews -bs ${bs} -ef Podcast_1/${encoder}_${seg} -lf Podcast_1/labs_dict.pkl --metric ${metric} -loss BinaryCrossEntropy -max 1000 -vp 0.15 -pat 50 -ar -as -kcv 48 -msk -msk_pr 0.70
    done