expdir=$1
lab_out=$2
uniform_time=$3

declare -a encoders=(x-vectors openl3 prosodic mfcc wav2vec crepe ecapa)
# x-vectors
for encoder in "${encoders[@]}"
    do
        if [[ ${encoder} == "x-vectors" ]]; then
            expname=${expdir}/x-vectors
            python extract_embeddings.py -data OpenBMAT/BMAT-ATS_transcripts  -audio OpenBMAT/BMAT-ATS -od ${expname} -lod ${lab_out} -lab data/OpenBMAT/BMAT-ATS_labels.json -vb -sb -ui ${uniform_time} -vd --BMAT --cont
        elif [[ ${encoder} == "openl3" ]]; then
            expname=${expdir}/openl3
            python extract_embeddings.py -data OpenBMAT/BMAT-ATS_transcripts -audio OpenBMAT/BMAT-ATS -od ${expname} -lod ${lab_out} -lab data/OpenBMAT/BMAT-ATS_labels.json -vb -sb -ui ${uniform_time} -vd --openl3 --BMAT --cont
        elif [[ ${encoder} == "prosodic" ]]; then
            expname=${expdir}/prosodic
            python extract_embeddings.py -data OpenBMAT/BMAT-ATS_transcripts -audio OpenBMAT/BMAT-ATS -od ${expname} -lod ${lab_out} -lab data/OpenBMAT/BMAT-ATS_labels.json -vb -sb -ui ${uniform_time} -vd --prosodic --BMAT --cont
        elif [[ ${encoder} == "mfcc" ]]; then
            expname=${expdir}/mfcc
            python extract_embeddings.py -data OpenBMAT/BMAT-ATS_transcripts -audio OpenBMAT/BMAT-ATS -od ${expname} -lod ${lab_out} -lab data/OpenBMAT/BMAT-ATS_labels.json -vb -sb -ui ${uniform_time} -vd --mfcc --BMAT --cont
        elif [[ ${encoder} == "wav2vec" ]]; then
            expname=${expdir}/wav2vec
            python extract_embeddings.py -data OpenBMAT/BMAT-ATS_transcripts -audio OpenBMAT/BMAT-ATS -od ${expname} -lod ${lab_out} -lab data/OpenBMAT/BMAT-ATS_labels.json -vb -sb -ui ${uniform_time} -vd --wav2vec --BMAT --cont
        elif [[ ${encoder} == "crepe" ]]; then
            expname=${expdir}/crepe
            python extract_embeddings.py -data OpenBMAT/BMAT-ATS_transcripts -audio OpenBMAT/BMAT-ATS -od ${expname} -lod ${lab_out} -lab data/OpenBMAT/BMAT-ATS_labels.json -vb -sb -ui ${uniform_time} -vd --CREPE --BMAT --cont
        else
            expname=${expdir}/ecapa
            python extract_embeddings.py -data OpenBMAT/BMAT-ATS_transcripts -audio OpenBMAT/BMAT-ATS -od ${expname} -lod ${lab_out} -lab data/OpenBMAT/BMAT-ATS_labels.json -vb -sb -ui ${uniform_time} -vd --ecapa --BMAT --cont
        fi
    done