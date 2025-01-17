# Download Dataset
In order to use the RadioNews-BBC dataset, you first need to download it from [here](https://zenodo.org/record/7372490).
From the downloaded zip file, extract the sub-folder "PodcastUniform1" into this directory.

# Replicate the Results from the Paper
To replicate the results from the original paper, after having downloaded and unzipped the dataset you should have a folder named "NonNewsUniform1" inside this folder.
Once you're sure you have correctly extracted the dataset, run the following code:

`./run_radionews_exp.sh 10 Experiments b`

The script will run all the experiments with all the encoders on the RadioNews-BBC dataset and, when finished, you can find all results inside the results.txt of each subfolder
inside the newly generated Experiments folder. Each subfolder is named after the relative encoder. 
Note that the script doesn't allow two output folders with the same name and, for that, if you want to re-run the experiments change the "Experiments" argument in the
script above (e.g. Experiments2)

To generate all the p-values reported in the original paper together with all the results with different metrics, make sure of having run the training code and named the output directory "Experiments", then run the following:

`python ../calculate_accuracy_metrics.py nonnews`

A csv file named final_bilstm_results.csv will be generated inside the Experiments directory containing all results and pvalues.
