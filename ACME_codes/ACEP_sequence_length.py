
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
#--------------------------------------------------------

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



text_font1={
    #'family':'Times New Roman',
    'style':'normal',
    'weight':'bold',
      'color':'k',
      'size':18
}

text_font2={
    #'family':'Times New Roman',
    'style':'normal',
    'weight':'normal',
      'color':'k',
      'size':14
}

test_length_pd = pd.Series(train_p_l)
plt.figure(figsize=(11,6))
test_length_pd.plot.hist(grid=True, bins=40, rwidth=0.65,
                   color='#607c8e')
plt.title('Training Set AMPs Length',fontdict=text_font1)
plt.xlabel('Sequence Length(AA)',fontdict=text_font2)
plt.ylabel('Frequency',fontdict=text_font2)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(axis='y', alpha=0.75)
plt.savefig('experiment_results\\Train_AMPs.png',dpi=600,format='png')


test_length_pd = pd.Series(train_n_l)
plt.figure(figsize=(11,6))
test_length_pd.plot.hist(grid=True, bins=40, rwidth=0.65,
                   color='brown')
plt.title('Training Set non-AMPs Length',fontdict=text_font1)
plt.xlabel('Sequence Length(AA)',fontdict=text_font2)
plt.ylabel('Frequency',fontdict=text_font2)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(axis='y', alpha=0.75)
plt.savefig('experiment_results\\Train_nonAMPs.png',dpi=600,format='png')

test_length_pd = pd.Series(tune_p_l)
plt.figure(figsize=(11,6))
test_length_pd.plot.hist(grid=True, bins=40, rwidth=0.65,
                   color='#607c8e')
plt.title('Tuning Set AMPs Length',fontdict=text_font1)
plt.xlabel('Sequence Length(AA)',fontdict=text_font2)
plt.ylabel('Frequency',fontdict=text_font2)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(axis='y', alpha=0.75)
plt.savefig('experiment_results\\Tune_AMPs.png',dpi=600,format='png')

test_length_pd = pd.Series(tune_n_l)
plt.figure(figsize=(11,6))
test_length_pd.plot.hist(grid=True, bins=40, rwidth=0.65,
                   color='brown')
plt.title('Tuning Set non-AMPs Length',fontdict=text_font1)
plt.xlabel('Sequence Length(AA)',fontdict=text_font2)
plt.ylabel('Frequency',fontdict=text_font2)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(axis='y', alpha=0.75)
plt.savefig('experiment_results\\Tune_nonAMPs.png',dpi=600,format='png')

test_length_pd = pd.Series(test_p_l)
plt.figure(figsize=(11,6))
test_length_pd.plot.hist(grid=True, bins=40, rwidth=0.65,
                   color='#607c8e')
plt.title('Testing Set AMPs Length',fontdict=text_font1)
plt.xlabel('Sequence Length(AA)',fontdict=text_font2)
plt.ylabel('Frequency',fontdict=text_font2)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(axis='y', alpha=0.75)
plt.savefig('experiment_results\\Test_AMPs.png',dpi=600,format='png')

test_length_pd = pd.Series(test_n_l)
plt.figure(figsize=(11,6))
test_length_pd.plot.hist(grid=True, bins=40, rwidth=0.65,
                   color='brown')
plt.title('Testing Set non-AMPs Length',fontdict=text_font1)
plt.xlabel('Sequence Length(AA)',fontdict=text_font2)
plt.ylabel('Frequency',fontdict=text_font2)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(axis='y', alpha=0.75)
plt.savefig('experiment_results\\Test_nonAMPs.png',dpi=600,format='png')

plt.show()



all_p_c,all_p_l,all_n_c,all_n_l = length_n_p(x_train_tune_test_length,y_train_tune_test)
print(all_p_c,all_n_c)
amp_pd = pd.Series(all_p_l)
print(amp_pd.describe())

count = 0
for i in (all_p_l+all_n_l):
    if i>=100:
        count = count +1
print(count)
print(len(all_p_l+all_n_l))