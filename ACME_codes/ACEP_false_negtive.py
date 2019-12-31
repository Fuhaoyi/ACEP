
from EmbeddingRST import EmbeddingRST_model
from feature_generation_train import ConvertSequence2Feature


from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Lambda, Permute, Reshape, Flatten, Masking
from keras.layers import Multiply, Dot, RepeatVector,Concatenate
from keras.layers import Embedding
from keras.layers import LSTM,Bidirectional,BatchNormalization
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import TimeDistributed,Add
from keras.models import load_model

import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.callbacks import TensorBoard

import pandas as pd
import numpy as np
from sklearn.preprocessing import binarize


import os



data_all = pd.read_csv('AMPs_Experiment_Dataset\\AMP_sequecnces\\seq_all_data.csv', index_col=0)

train = data_all.iloc[0:1424].reset_index(drop=True)
tune = data_all.iloc[1424:2132].reset_index(drop=True)
test = data_all.iloc[2132:3556].reset_index(drop=True)
train_tune = data_all.iloc[0:2132].reset_index(drop=True)
train_tune_test = data_all.iloc[0:3556].reset_index(drop=True)


x_train_index, x_train_length, x_train_pssm, x_train_onehot,x_train_aac,y_train= ConvertSequence2Feature(sequence_data=train, pssmdir=os.path.join('AMPs_Experiment_Dataset','PSSM_files','train'))
x_test_index, x_test_length, x_test_pssm, x_test_onehot, x_test_aac, y_test = ConvertSequence2Feature(sequence_data=test, pssmdir=os.path.join('AMPs_Experiment_Dataset','PSSM_files','test'))
x_tune_index, x_tune_length, x_tune_pssm, x_tune_onehot,x_tune_aac,y_tune= ConvertSequence2Feature(sequence_data=tune, pssmdir=os.path.join('AMPs_Experiment_Dataset','PSSM_files','tune'))
x_train_tune_index, x_train_tune_length, x_train_tune_pssm, x_train_tune_onehot,x_train_tune_aac,y_train_tune= ConvertSequence2Feature(sequence_data=train_tune, pssmdir=os.path.join('AMPs_Experiment_Dataset','PSSM_files','train_tune'))
x_train_tune_test_index, x_train_tune_test_length, x_train_tune_test_pssm, x_train_tune_test_onehot,x_train_tune_test_aac,y_train_tune_test= ConvertSequence2Feature(sequence_data=train_tune_test, pssmdir=os.path.join('AMPs_Experiment_Dataset','PSSM_files','train_tune_test'))



model = load_model('models//ACEP_model_train_241_9304.h5',
                        custom_objects={'EmbeddingRST_model': EmbeddingRST_model})
print(model.evaluate([x_test_aac,x_test_onehot, x_test_pssm], y_test, batch_size=16, verbose=0))
print(model.summary())
#plot_model(model, to_file='ACEP_model_train_241_9304.png',show_shapes=True)



def length_n_p(x_length,ylabel):
    count = 0
    length_p = []
    length_n = []
    for i in range(len(ylabel)):
        if ylabel[i] == 1:
            length_p.append(x_length[i])
            count = count+1
        else:
            length_n.append(x_length[i])
    return count,length_p,len(ylabel)-count,length_n

test_p_c,test_p_l,test_n_c,test_n_l = length_n_p(x_test_length,y_test)
print(test_p_c,test_n_c)
tune_p_c,tune_p_l,tune_n_c,tune_n_l = length_n_p(x_tune_length,y_tune)
print(tune_p_c,tune_n_c)
train_p_c,train_p_l,train_n_c,train_n_l = length_n_p(x_train_length,y_train)
print(train_p_c,train_n_c)


y_pred_prob = model.predict([x_test_aac, x_test_onehot, x_test_pssm])
y_pred_prob = y_pred_prob.flatten()
y_pred_class = binarize([y_pred_prob], 0.5)[0]

err_seq_index = []
err_seq = []
for i in range(1424):
    if  y_test[i]==1 and y_pred_class[i]==0:
        err_seq_index.append(i)
        err_seq.append(test.iloc[i,0])
        print(i,test.iloc[i,0])



apd_id = []
f = open('AMP_Scan2_OrigPaper_Dataset\\AMP.te.fa', 'r')
str1 = f.read()
f.close()
list1 = str1.split()
for i in range(len(err_seq)):
    for j in range(len(list1)):
        if err_seq[i]==list1[j]:
            apd_id.append(list1[j-1][1:])
            break

err_seqence_pd = pd.DataFrame({'index':err_seq_index,'apd_id':apd_id,'sequence':err_seq})
err_seqence_pd.to_csv('experiment_results\\err_seqence_adp.csv')


