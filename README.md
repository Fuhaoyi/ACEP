# ACEP
Open source software and datasets for the ACEP algorithm

1.Making predictions for sequences
---
1.1 Description
#
Input the peptide sequences and PSSM file, the model can predict whether the sequences are AMPs or non-AMPs. PSSM files of Sequences can be obtained through POSSUM website (http://possum.erc.monash.edu/). This software supports high-throughput predictions. The prediction results are stored in a file.

1.2 Requirements
#
Python3
Python packages: Tensorfollow(vr.1.6.0), Keras(vr.2.15), Matplotlib, scikit-learn, numpy, pandas and senborn(The senborn package is used for visualization).

We recommend using a GPU to speed up the calculations; if you use GPU acceleration, you also need to install cuda and cudnn.

1.3 Starting a prediction
#
Step 1. Download all files in the folderthe ACME_codes folder. 

Step 2. Enter the sequences into the AMP_prediction/inputs/sequences.csv file and put the PSSM files into the AMP_prediction/inputs/PSSM_files/ directory.

Step 3. Run ACEP_prediction.py.

Step 4. View predicted results in AMP_prediction/outputs/outputs.csv file. And sequences with probability greater than or equal to 0.5 are identified as AMPs, and sequences with probability less than 0.5 are identified as non-AMPs.

2.Repeating the experiments in the ACME paper.
-



<div align=center><img width="60%" height="60%" alt="Model_Structure" src="https://raw.githubusercontent.com/Fuhaoyi/ACEP/master/model_structure.png"/></div>

