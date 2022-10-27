# Download Dataset
In order to use the NonNews-BBC dataset, you first need to download it from [here]([https://zenodo.org/record/7255082).
From the downloaded zip file, extract the sub-folder "NonNewsUniform1" into this directory.

# Replicate the Results from the Paper
To replicate the results from the original paper, after having downloaded and unzipped the dataset you should have a folder named "NonNewsUniform1" inside this folder.
Once you're sure you have correctly extracted the dataset, run the following code:

`./run_nonnews_exp.sh 10 NonNews_Experiments F1 1`

The script will run all the experiments with all the encoders on the NonNews-BBC dataset and, when finished, you can find all results inside the results.txt of each subfolder
inside the newly generated NonNews_Experiments folder. Each subfolder is named after the relative encoder. 
Note that the script doesn't allow two output folders with the same name and, for that, if you want to re-run the experiments change the "NonNews_Experiments" argument in the
script above (e.g. NonNews_Experiments2)
