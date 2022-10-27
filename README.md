# Audio-Topic-Segmentation
Repository for the paper "Exploring pre-trained Audio Neural Representations for Audio Topic Segmentation"

# Installation
Before using this repository, create a virtual environment such as:

`virtualenv audio_topic_seg`

Then, activate it:

`source audio_topic_seg/bin/activate`

And, from inside the environment, install all the dependencies with:

`pip install -r requirements.txt`

# Use
To replicate the results for the individual datasets presented in the original paper, follow below instructions

## NonNews-BBC
Follow the instructions inside the README.md file in the NonNews-BBC folder in this repository.

## RadioNews-BBC
Follow the instructions inside the README.md file in the RadioNews-BBC folder in this repository.

## BMAT-ATS
To use this dataset you first need to generate the dataset from the OpenBMAT dataset. To do so, first follow the instructions that you can find in the OpenBMAT folder in this repository. Once generated the dataset, change back your directory to the parent directory (this one) and run the following command:

`./run_uniform_extraction_BMAT.sh OpenBMAT/BMAT_1 1`

Once generated the audio embeddings with the above command. You can then change directory into OpenBMAT and follow the instructions under "Run the Topic Segmentation Training" in the README.md file inside that folder.
