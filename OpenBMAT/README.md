# Create Dataset
The BMAT-ATS dataset first needs to be created from the publicly available OpenBMAT dataset. In order to do so, follow the instructions in the BMAT-ATS_instructions.txt file.
A short summary of the steps involved in the dataset creation is also shown below:

1) Download OpenBMAT and move it to this folder
2) run `python create_news_seg.py`
3) run `python create_dataset.py`

# Train the Topic Segmentation Model
After having the BMAT-ATS dataset, you first need to extract all the audio embeddings. This is done by following the instructions in the parent directory under BMAT-ATS.

Once extracted the embeddings, run the following:

`./run_BMAT_exp.sh 10 BMAT_Experiments F1 1`

Please note that if you want to re-run the experiments, a new name needs to be assigned, e.g.:

`./run_BMAT_exp.sh 10 BMAT_Experiments2 F1 1`
