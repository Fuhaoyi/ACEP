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
* **Step 1.** Download all files in the folderthe ACME_codes folder. 

* **Step 2.** Enter the sequences into the AMP_prediction/inputs/sequences.csv file and put the PSSM files into the AMP_prediction/inputs/PSSM_files/ directory.

* **Step 3.** Run ACEP_prediction.py.

* **Step 4.** View predicted results in AMP_prediction/outputs/outputs.csv file. And sequences with probability greater than or equal to 0.5 are identified as AMPs, and sequences with probability less than 0.5 are identified as non-AMPs.

2.Architecture of the ACEP modle.
---

We use convolutional layer, pooling layer, LSTM layer, fully connected layer and attention mechanism to build the model.
The yellow module, the blue module and the red module are used to generate features. The green module is used to fuse features; the purple module corresponds to the sigmoid node that outputs the prediction results.

<div align=center><img width="50%" height="50%" alt="Model_Structure" src="https://raw.githubusercontent.com/Fuhaoyi/ACEP/master/model_structure.png"/></div>


3.Repeating the experiments in the paper.
---

Experiments in the paper can be repeated by running the code in the ACME_codes/ folder. When running the codes, all experimental results will be displayed and stored in the ACME_codes/experiment_results/ folder.

3.1 Training model

You can train a new model by running ACEP_model_train.py files. And the code for the amino acid embedding tensor can be found in the EmbeddingRST.py file. The training data is placed in the AMPs_Experiment_Dataset / AMPs_Experiment_Dataset.zip file. This file needs to be decompressed before training the model. In addition, you can observe the training history by running the ACEP_training_history.py file.

3.2 Experimental comparison

You can view the performance of the model by running ACEP_model_performance.py. It can also be compared with other state-of-the-art antimicrobial peptides recognition methods by running ACEP_comparison_test.py and ACEP_ROC.py. In addition, run ACEP_R1_xxx.py to understand the role of different functional modules. Evaluate model performance using cross validation on all datasets by running ACEP_model_CV.py

3.3 Experimental analysis and visualization

Observe the amino acid clustering by running ACEP_cluster.py. Observe the attention intensity and the the fusion ratio by running ACEP_attention_intensity.py and ACEM_fusion_ratio.py. Observe the distribution of fusion features in space by running ACEP_fusion_feature.py.

3.4 Others


If your sequence is a fasta file, you can call the function in Data_pre_processing.py to convert the fasta file to a csv file so that it can be imported into the model.
If you want to see false negative sequences, you can run ACEP_false_negtive.py file.


If you have any problems, please contact fhy11235813@gmail.com

Contact Us
---

Haoyi Fu

School of Information Science and Engineering, Yunnan University


