
from EmbeddingRST import EmbeddingRST_model
from feature_generation_train import ConvertSequence2Feature
from model_evaluation import evaluation

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Lambda, Permute, Reshape, Flatten, Masking
from keras.layers import Multiply, Dot, RepeatVector,Concatenate
from keras.layers import Embedding
from keras.layers import LSTM,Bidirectional,BatchNormalization
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import TimeDistributed,Add

import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.callbacks import TensorBoard

import pandas as pd
import numpy as np
#--------------------------------------------------------

import os


data_all = pd.read_csv('AMPs_Experiment_Dataset\\AMP_sequecnces\\seq_all_data.csv', index_col=0)

train = data_all.iloc[0:2132].reset_index(drop=True)
test = data_all.iloc[2132:3556].reset_index(drop=True)


x_train_index, x_train_length, x_train_pssm, x_train_onehot,x_train_aac,y_train= ConvertSequence2Feature(sequence_data=train, pssmdir=os.path.join('AMPs_Experiment_Dataset','PSSM_files','train_tune'))
x_test_index, x_test_length, x_test_pssm, x_test_onehot, x_test_aac, y_test = ConvertSequence2Feature(sequence_data=test, pssmdir=os.path.join('AMPs_Experiment_Dataset','PSSM_files','test'))


def length_index(length_list, feature_data, label):
    index30_200 = []
    index1_29 = []
    labelnp = np.array(label)
    for i in range(len(length_list)):
        if(length_list[i])>29:
            index30_200.append(i)
        else:
            index1_29.append(i)
    feature_data30_200 = feature_data[index30_200]
    labelnp30_200 = labelnp[index30_200]
    feature_data1_29 = feature_data[index1_29]
    labelnp1_29 = labelnp[index1_29]


    return feature_data30_200,labelnp30_200,feature_data1_29,labelnp1_29

x_train_onehot30_200,y_train_onehot30_200,x_train_onehot1_29,y_train_onehot1_29 = length_index(x_train_length, x_train_onehot,y_train)
x_train_pssm30_200,y_train_pssm30_200,x_train_pssm1_29,y_train_pssm1_29 = length_index(x_train_length, x_train_pssm,y_train)

x_test_onehot30_200,y_test_onehot30_200,x_test_onehot1_29,y_test_onehot1_29 = length_index(x_test_length, x_test_onehot,y_test)
x_test_pssm30_200,y_test_pssm30_200,x_test_pssm1_29,y_test_pssm1_29 = length_index(x_test_length, x_test_pssm,y_test)
x_test_aac30_200,y_test_aac30_200,x_test_aac1_29,y_test_aac1_29 = length_index(x_test_length, x_test_aac,y_test)




def creat_model_R3():
    droupot_para = 0.3
    feture_dim = 64

    inputs_aac = Input(shape=(21,),name='main_input_aac')
    hidden_acc = Dense(100, activation='relu')(inputs_aac)
    hidden_acc = Dropout(droupot_para)(hidden_acc)
    hidden_acc = Dense(feture_dim, activation='relu')(hidden_acc)
    hidden_acc = Dropout(droupot_para)(hidden_acc)

    y_out = Dense(1, activation='sigmoid')(hidden_acc)

    model = Model(inputs=[inputs_aac], outputs=y_out)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model



model_R3 = creat_model_R3()
print(model_R3.summary())
plot_model(model_R3, to_file='ACEP_model_R3.png',show_shapes=True)
model_R3.fit([x_train_aac], y_train, batch_size=16, epochs=30)
y_pre_all = model_R3.predict([x_test_aac]).flatten()
y_pre_1_29 = model_R3.predict([x_test_aac1_29]).flatten()
y_pre_30_200 = model_R3.predict([x_test_aac30_200]).flatten()
print('####################       R3        ######################')
print('all__________________________________')
evaluation(y_pre_all,y_test)
print('1_29_________________________________')
evaluation(y_pre_1_29,y_test_aac1_29)
print('30_200_______________________________')
evaluation(y_pre_30_200,y_test_aac30_200)
print()










