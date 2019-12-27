
from EmbeddingRST import EmbeddingRST_model


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
import math


def getpssm(filename):
    f = open(filename, 'r')
    f.readline()
    f.readline()
    f.readline()
    linelist = f.readlines()
    pssm = []
    for i in linelist:
        if i!='\n':
            oneline = i.split()
            m = []
            m.append(oneline[1])
            m.extend([int(i) for i in oneline[2:22]])
            pssm.append(m)
        else:
            break
    f.close()

    pssmdf = pd.DataFrame(pssm,columns=['seq','A','R','N','D','C','Q','E','G','H','I',
                                        'L','K','M','F','P','S','T','W','Y','V'])
    col = pssmdf.columns.insert(1, 'X')
    pssmdf = pssmdf.reindex(columns=col, fill_value=0)

    return pssmdf


def getpssmlist(dir):
    filenamelist = os.listdir(dir)
    pssmlist = []
    pssm_file_list = []
    for i in filenamelist:
        filename = os.path.join(dir,i)
        pssmlist.append(getpssm(filename))
        pssm_file_list.append(i)
    print('pssmlist is ok')
    print(len(pssm_file_list))
    #print(pssm_file_list)
    return  pssmlist, pssm_file_list




def psssmfeature(sequence_index,one_sequence, label, pssm_matrix_list):

    str1 = one_sequence
    padding_vst_pssm = -1
    sequence_pssm = []
    str2 = ''.join(pssm_matrix_list[sequence_index].iloc[:, 0].values)
    if str1 == str2:
        pssm = pssm_matrix_list[sequence_index].iloc[:, 1:22].values
        padding = np.zeros((200 - pssm.shape[0], 21), dtype=np.float32)
        padding_vst_pssm = np.vstack([padding, pssm])
        sequence_pssm = [sequence_index, pssm_matrix_list[sequence_index]]

    if type(padding_vst_pssm) == type(-1):
        print('error pssm in empty')
        print(sequence_index)
        print(one_sequence)

    sequence_label = label
    sequence_index_out = sequence_index
    return sequence_index_out, padding_vst_pssm, sequence_label, sequence_pssm  



def encode2numerical(sequence_index, one_sequence, label):
    dict1 = {'X':0,
             'A':1,
             'R':2,
             'N':3,
             'D':4,
             'C':5,
             'Q':6,
             'E':7,
             'G':8,
             'H':9,
             'I':10,
             'L':11,
             'K':12,
             'M':13,
             'F':14,
             'P':15,
             'S':16,
             'T':17,
             'W':18,
             'Y':19,
             'V':20
             }

    str1 = one_sequence
    str1 = 'X' * (200 - len(str1)) + str1
    str2 = ''
    for j in str1:
        if j not in dict1.keys():
            j = 'X'
        str2 =str2 + str(dict1[j]) + ' '

    sequence_num = str2.split()
    sequence_num = [int(k) for k in sequence_num]
    sequence_label = label
    sequence_index_out = sequence_index

    return sequence_index_out ,sequence_num, sequence_label



def one_hot_vector(sequence_index, one_sequence, label):
    sequence_index_out, sequence_num, sequence_label = encode2numerical(sequence_index, one_sequence, label)
    one_hot_matrix = np.zeros((200, 21), dtype=np.float32)
    for i in range(200):
        if sequence_num[i]!=0:
            one_hot_matrix[i,sequence_num[i]] = 1.0

    return one_hot_matrix

def AAC_ft(sequence_index_x, one_sequence_x, label_x):
    dict1 = {'X': 0,
             'A': 1,
             'R': 2,
             'N': 3,
             'D': 4,
             'C': 5,
             'Q': 6,
             'E': 7,
             'G': 8,
             'H': 9,
             'I': 10,
             'L': 11,
             'K': 12,
             'M': 13,
             'F': 14,
             'P': 15,
             'S': 16,
             'T': 17,
             'W': 18,
             'Y': 19,
             'V': 20
             }

    str1 = one_sequence_x
    str2 = ''
    for j in str1:
        if j not in dict1.keys():
            j = 'X'
        str2 = str2 + str(dict1[j]) + ' '

    sequence_num = str2.split()
    sequence_num = [int(k) for k in sequence_num]
    sequence_aac = np.zeros((21,),dtype=np.float32)
    for i in range(21):
        count = 0
        for j in sequence_num:
            if i == j:
                count = count+1
        sequence_aac[i] = count
    sequence_label = label_x
    sequence_index_out = sequence_index_x

    return sequence_aac


def ConvertSequence2Feature(sequence_data, pssmdir):
    sequence_count = sequence_data.shape[0]
    feature_pssm = np.zeros((sequence_count, 200, 21), dtype=np.float32)
    feature_onehot = np.zeros((sequence_count, 200, 21), dtype=np.float32)
    feature_aac = np.zeros((sequence_count, 21), dtype=np.float32)

    label_list = np.zeros((sequence_count), dtype=np.int32)
    index_out = np.zeros((sequence_count), dtype=np.int32)
    pssm_list, pssm_file_list = getpssmlist(pssmdir)
    length_list = []
    for i in range(sequence_count):
        length_list.append(len(sequence_data.iloc[i, 0]))
    count_1 = 0
    for i in range(sequence_count):
        one_sequence = sequence_data.iloc[i,0]
        label = sequence_data.iloc[i,2]
        index_out[i], feature_pssm[i], label_list[i], sequence_pssm = psssmfeature(i, one_sequence, label, pssm_list)
        feature_onehot[i] = one_hot_vector(i, one_sequence, label)
        feature_aac[i] = AAC_ft(i, one_sequence, label)
        count_1 = count_1 +1
    print('total    ',count_1)
    return index_out, length_list, feature_pssm, feature_onehot, feature_aac, label_list


#--------------------------------------------------------

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




#-------------------------evaluation----------------------
from sklearn.preprocessing import binarize
from sklearn import metrics


def evaluate_model(y_pre,y_test):
    y_pred_prob = y_pre
    y_pred_prob = y_pred_prob.flatten()
    y_pred_class = binarize([y_pred_prob], 0.5)[0]
    # print(y_pred_prob)
    # print(y_pred_class)

    confusion = metrics.confusion_matrix(y_test, y_pred_class)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    print(confusion)

    ACC = metrics.accuracy_score(y_test, y_pred_class)
    print('Classification Accuracy:', ACC)

    Error = 1 - metrics.accuracy_score(y_test, y_pred_class)
    print('Classification Error:', Error)

    Sens = metrics.recall_score(y_test, y_pred_class)
    print('Sensitivity:', Sens)

    Spec = TN / float(TN + FP)
    print('Specificity:', Spec)

    FPR = FP / float(TN + FP)
    print('False Positive Rate:', FPR)

    Precision = metrics.precision_score(y_test, y_pred_class)
    print('Precision:', Precision)

    F1_score = metrics.f1_score(y_test, y_pred_class)
    print('F1 score:', F1_score)

    MCC = metrics.matthews_corrcoef(y_test, y_pred_class)
    print('Matthews correlation coefficient:', MCC)


    AUC = metrics.roc_auc_score(y_test, y_pred_prob)
    print('ROC Curves and Area Under the Curve (AUC):', AUC)






def creat_model_R1():
    droupot_para = 0.3
    e_dim = 64
    feture_dim = 64


    inputs_pm = Input(shape=(200,21),name='main_input_pm')
    emb_pm = EmbeddingRST_model(input_dim=21,output_dim=e_dim,input_length=200,
                             name='emb_tensor_pm')(inputs_pm)  
    cov1d_pm = Conv1D(64, 16, activation='relu',strides=1, padding='same',
                   kernel_initializer='random_normal')(emb_pm)
    maxpool_pm = MaxPooling1D(5)(cov1d_pm) 
    lstm_out_pm = LSTM(feture_dim,dropout=droupot_para,return_sequences=True,unroll=True)(maxpool_pm)


    lstm_out_permute_pm = Permute((2,1))(lstm_out_pm)
    attention_weights_pm = TimeDistributed(Dense(1))(lstm_out_pm)
    attention_weights_pm = Flatten()(attention_weights_pm)
    attention_weights_pm = Activation('softmax',name='attention_weights_pm')(attention_weights_pm)
    attention_weights_pm = Dropout(droupot_para)(attention_weights_pm)
    merge_out_pm = Dot(-1,name='merge_out2')([attention_weights_pm,lstm_out_permute_pm])


    y_out = Dense(1, activation='sigmoid')(merge_out_pm)

    model = Model(inputs=[inputs_pm], outputs=y_out)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


# model_R1 = creat_model_R1()
# print(model_R1.summary())
# plot_model(model_R1, to_file='ACEP_model_R1.png',show_shapes=True)
# model_R1.fit([x_train_pssm], y_train, batch_size=16, epochs=30)
# y_pre_all = model_R1.predict([x_test_pssm])
# y_pre_1_29 = model_R1.predict([x_test_pssm1_29])
# y_pre_30_200 = model_R1.predict([x_test_pssm30_200])
# print('####################       R1        ######################')
# print('all__________________________________')
# evaluate_model(y_pre_all,y_test)
# print('1_29_________________________________')
# evaluate_model(y_pre_1_29,y_test_pssm1_29)
# print('30_200_______________________________')
# evaluate_model(y_pre_30_200,y_test_pssm30_200)
# print()





def creat_model_R2():
    droupot_para = 0.3
    e_dim = 64
    feture_dim = 64


    inputs_ot = Input(shape=(200,21),name='main_input_ot')
    emb_ot = EmbeddingRST_model(input_dim=21,output_dim=e_dim,input_length=200,
                             name='emb_tensor_ot')(inputs_ot)   
    cov1d_ot = Conv1D(64, 16, activation='relu',strides=1, padding='same',
                   kernel_initializer='random_normal')(emb_ot)  
    maxpool_ot = MaxPooling1D(3)(cov1d_ot)    
    lstm_out_ot = LSTM(feture_dim,dropout=droupot_para,return_sequences=True,unroll=True)(maxpool_ot)



    lstm_out_permute_ot = Permute((2,1))(lstm_out_ot)     
    attention_weights_ot = TimeDistributed(Dense(1))(lstm_out_ot)     
    attention_weights_ot = Flatten()(attention_weights_ot)
    attention_weights_ot = Activation('softmax',name='attention_weights_ot')(attention_weights_ot)
    attention_weights_ot = Dropout(droupot_para)(attention_weights_ot)
    merge_out_ot = Dot(-1,name='merge_out1')([attention_weights_ot,lstm_out_permute_ot])

    y_out = Dense(1, activation='sigmoid')(merge_out_ot)

    model = Model(inputs=[inputs_ot], outputs=y_out)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


# model_R2 = creat_model_R2()
# print(model_R2.summary())
# plot_model(model_R2, to_file='ACEP_model_R2.png',show_shapes=True)
# model_R2.fit([x_train_onehot], y_train, batch_size=16, epochs=30)
# y_pre_all = model_R2.predict([x_test_onehot])
# y_pre_1_29 = model_R2.predict([x_test_onehot1_29])
# y_pre_30_200 = model_R2.predict([x_test_onehot30_200])
# print('####################       R2        ######################')
# print('all__________________________________')
# evaluate_model(y_pre_all,y_test)
# print('1_29_________________________________')
# evaluate_model(y_pre_1_29,y_test_onehot1_29)
# print('30_200_______________________________')
# evaluate_model(y_pre_30_200,y_test_onehot30_200)
# print()






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



# model_R3 = creat_model_R3()
# print(model_R3.summary())
# plot_model(model_R3, to_file='ACEP_model_R3.png',show_shapes=True)
# model_R3.fit([x_train_aac], y_train, batch_size=16, epochs=30)
# y_pre_all = model_R3.predict([x_test_aac])
# y_pre_1_29 = model_R3.predict([x_test_aac1_29])
# y_pre_30_200 = model_R3.predict([x_test_aac30_200])
# print('####################       R3        ######################')
# print('all__________________________________')
# evaluate_model(y_pre_all,y_test)
# print('1_29_________________________________')
# evaluate_model(y_pre_1_29,y_test_aac1_29)
# print('30_200_______________________________')
# evaluate_model(y_pre_30_200,y_test_aac30_200)
# print()






def creat_model_R12():
    droupot_para = 0.3
    e_dim = 64
    feture_dim = 64


    inputs_ot = Input(shape=(200,21),name='main_input_ot')
    emb_ot = EmbeddingRST_model(input_dim=21,output_dim=e_dim,input_length=200,
                             name='emb_tensor_ot')(inputs_ot)  
    cov1d_ot = Conv1D(64, 16, activation='relu',strides=1, padding='same',
                   kernel_initializer='random_normal')(emb_ot)
    maxpool_ot = MaxPooling1D(3)(cov1d_ot)
    lstm_out_ot = LSTM(feture_dim,dropout=droupot_para,return_sequences=True,unroll=True)(maxpool_ot)



    inputs_pm = Input(shape=(200,21),name='main_input_pm')
    emb_pm = EmbeddingRST_model(input_dim=21,output_dim=e_dim,input_length=200,
                             name='emb_tensor_pm')(inputs_pm)
    cov1d_pm = Conv1D(64, 16, activation='relu',strides=1, padding='same',
                   kernel_initializer='random_normal')(emb_pm)
    maxpool_pm = MaxPooling1D(5)(cov1d_pm)
    lstm_out_pm = LSTM(feture_dim,dropout=droupot_para,return_sequences=True,unroll=True)(maxpool_pm)



    lstm_out_permute_ot = Permute((2,1))(lstm_out_ot)
    attention_weights_ot = TimeDistributed(Dense(1))(lstm_out_ot) 
    attention_weights_ot = Flatten()(attention_weights_ot)
    attention_weights_ot = Activation('softmax',name='attention_weights_ot')(attention_weights_ot) 
    attention_weights_ot = Dropout(droupot_para)(attention_weights_ot)
    merge_out_ot = Dot(-1,name='merge_out1')([attention_weights_ot,lstm_out_permute_ot])
    merge_out_ot = Reshape((1,-1))(merge_out_ot)



    lstm_out_permute_pm = Permute((2,1))(lstm_out_pm)
    attention_weights_pm = TimeDistributed(Dense(1))(lstm_out_pm)
    attention_weights_pm = Flatten()(attention_weights_pm)
    attention_weights_pm = Activation('softmax',name='attention_weights_pm')(attention_weights_pm)
    attention_weights_pm = Dropout(droupot_para)(attention_weights_pm)
    merge_out_pm = Dot(-1,name='merge_out2')([attention_weights_pm,lstm_out_permute_pm])
    merge_out_pm = Reshape((1,-1))(merge_out_pm)



    concat_one_pssm = Concatenate(axis=1)([merge_out_ot, merge_out_pm])
    cov_f = Conv1D(feture_dim, 2, activation='relu',strides=1, padding='valid',
                   kernel_initializer='random_normal')(concat_one_pssm)
    concat_one_pssm_ot = Concatenate(axis=2)([merge_out_ot, cov_f])
    concat_one_pssm_pm = Concatenate(axis=2)([merge_out_pm, cov_f])
    concat_3 = Concatenate(axis=1)([concat_one_pssm_ot,concat_one_pssm_pm])



    concat_permute = Permute((2, 1))(concat_one_pssm)
    concat_attention_weights = TimeDistributed(Dense(1))(concat_3)
    concat_attention_weights = Flatten()(concat_attention_weights)
    concat_attention_weights = Activation('softmax', name='select_weights')(concat_attention_weights)
    concat_attention_weights = Dropout(droupot_para)(concat_attention_weights)
    merge_out = Dot(-1, name='merge_select')([concat_attention_weights, concat_permute])



    y_out = Dense(1, activation='sigmoid')(merge_out)

    model = Model(inputs=[inputs_ot, inputs_pm], outputs=y_out)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model




# model_R12 = creat_model_R12()
# print(model_R12.summary())
# plot_model(model_R12, to_file='ACEP_model_R12.png',show_shapes=True)
# model_R12.fit([x_train_onehot, x_train_pssm], y_train, batch_size=16, epochs=30)
# y_pre_all = model_R12.predict([x_test_onehot, x_test_pssm])
# y_pre_1_29 = model_R12.predict([x_test_onehot1_29, x_test_pssm1_29])
# y_pre_30_200 = model_R12.predict([x_test_onehot30_200, x_test_pssm30_200])
# print('####################       R1+R2        ######################')
# print('all__________________________________')
# evaluate_model(y_pre_all,y_test)
# print('1_29_________________________________')
# evaluate_model(y_pre_1_29,y_test_pssm1_29)
# print('30_200_______________________________')
# evaluate_model(y_pre_30_200,y_test_pssm30_200)
# print()







def creat_model_R13():
    droupot_para = 0.3
    e_dim = 64
    feture_dim = 64

    inputs_aac = Input(shape=(21,),name='main_input_aac')
    hidden_acc = Dense(100, activation='relu')(inputs_aac)
    hidden_acc = Dropout(droupot_para)(hidden_acc)
    hidden_acc = Dense(64, activation='relu')(hidden_acc)
    hidden_acc = Dropout(droupot_para)(hidden_acc)
    hidden_acc = Reshape((1,-1))(hidden_acc)


    inputs_pm = Input(shape=(200,21),name='main_input_pm')
    emb_pm = EmbeddingRST_model(input_dim=21,output_dim=e_dim,input_length=200,
                             name='emb_tensor_pm')(inputs_pm)
    cov1d_pm = Conv1D(64, 16, activation='relu',strides=1, padding='same',
                   kernel_initializer='random_normal')(emb_pm)
    maxpool_pm = MaxPooling1D(5)(cov1d_pm)
    lstm_out_pm = LSTM(feture_dim,dropout=droupot_para,return_sequences=True,unroll=True)(maxpool_pm)


    lstm_out_permute_pm = Permute((2,1))(lstm_out_pm)
    attention_weights_pm = TimeDistributed(Dense(1))(lstm_out_pm)
    attention_weights_pm = Flatten()(attention_weights_pm) 
    attention_weights_pm = Activation('softmax',name='attention_weights_pm')(attention_weights_pm)
    attention_weights_pm = Dropout(droupot_para)(attention_weights_pm)
    merge_out_pm = Dot(-1,name='merge_out2')([attention_weights_pm,lstm_out_permute_pm])
    merge_out_pm = Reshape((1,-1))(merge_out_pm)



    concat_one_pssm = Concatenate(axis=1)([hidden_acc, merge_out_pm])
    cov_f = Conv1D(feture_dim, 2, activation='relu',strides=1, padding='valid',
                   kernel_initializer='random_normal')(concat_one_pssm)
    concat_one_pssm_aac = Concatenate(axis=2)([hidden_acc,cov_f])
    concat_one_pssm_pm = Concatenate(axis=2)([merge_out_pm, cov_f])
    concat_3 = Concatenate(axis=1)([concat_one_pssm_aac,concat_one_pssm_pm])



    concat_permute = Permute((2, 1))(concat_one_pssm)
    concat_attention_weights = TimeDistributed(Dense(1))(concat_3)
    concat_attention_weights = Flatten()(concat_attention_weights)
    concat_attention_weights = Activation('softmax', name='select_weights')(concat_attention_weights)
    concat_attention_weights = Dropout(droupot_para)(concat_attention_weights)
    merge_out = Dot(-1, name='merge_select')([concat_attention_weights, concat_permute])



    y_out = Dense(1, activation='sigmoid')(merge_out)

    model = Model(inputs=[inputs_aac, inputs_pm], outputs=y_out)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model



# model_R13 = creat_model_R13()
# print(model_R13.summary())
# plot_model(model_R13, to_file='ACEP_model_R13.png',show_shapes=True)
# model_R13.fit([x_train_aac, x_train_pssm], y_train, batch_size=16, epochs=30)
# y_pre_all = model_R13.predict([x_test_aac, x_test_pssm])
# y_pre_1_29 = model_R13.predict([x_test_aac1_29, x_test_pssm1_29])
# y_pre_30_200 = model_R13.predict([x_test_aac30_200, x_test_pssm30_200])
# print('####################       R1+R3        ######################')
# print('all__________________________________')
# evaluate_model(y_pre_all,y_test)
# print('1_29_________________________________')
# evaluate_model(y_pre_1_29,y_test_pssm1_29)
# print('30_200_______________________________')
# evaluate_model(y_pre_30_200,y_test_pssm30_200)
# print()






def creat_model_R23():
    droupot_para = 0.3
    e_dim = 64
    feture_dim = 64

    inputs_aac = Input(shape=(21,),name='main_input_aac')
    hidden_acc = Dense(100, activation='relu')(inputs_aac)
    hidden_acc = Dropout(droupot_para)(hidden_acc)
    hidden_acc = Dense(64, activation='relu')(hidden_acc)
    hidden_acc = Dropout(droupot_para)(hidden_acc)
    hidden_acc = Reshape((1,-1))(hidden_acc)


    inputs_ot = Input(shape=(200,21),name='main_input_ot')
    emb_ot = EmbeddingRST_model(input_dim=21,output_dim=e_dim,input_length=200,
                             name='emb_tensor_ot')(inputs_ot)
    cov1d_ot = Conv1D(64, 16, activation='relu',strides=1, padding='same',
                   kernel_initializer='random_normal')(emb_ot)
    maxpool_ot = MaxPooling1D(3)(cov1d_ot) 
    lstm_out_ot = LSTM(feture_dim,dropout=droupot_para,return_sequences=True,unroll=True)(maxpool_ot) 



    lstm_out_permute_ot = Permute((2,1))(lstm_out_ot)
    attention_weights_ot = TimeDistributed(Dense(1))(lstm_out_ot)
    attention_weights_ot = Flatten()(attention_weights_ot)
    attention_weights_ot = Activation('softmax',name='attention_weights_ot')(attention_weights_ot)
    attention_weights_ot = Dropout(droupot_para)(attention_weights_ot)
    merge_out_ot = Dot(-1,name='merge_out1')([attention_weights_ot,lstm_out_permute_ot])
    merge_out_ot = Reshape((1,-1))(merge_out_ot)




    concat_one_pssm = Concatenate(axis=1)([hidden_acc,merge_out_ot])
    cov_f = Conv1D(feture_dim, 2, activation='relu',strides=1, padding='valid',
                   kernel_initializer='random_normal')(concat_one_pssm)
    concat_one_pssm_aac = Concatenate(axis=2)([hidden_acc,cov_f])
    concat_one_pssm_ot = Concatenate(axis=2)([merge_out_ot, cov_f])
    concat_3 = Concatenate(axis=1)([concat_one_pssm_aac,concat_one_pssm_ot])



    concat_permute = Permute((2, 1))(concat_one_pssm)
    concat_attention_weights = TimeDistributed(Dense(1))(concat_3)
    concat_attention_weights = Flatten()(concat_attention_weights)
    concat_attention_weights = Activation('softmax', name='select_weights')(concat_attention_weights)
    concat_attention_weights = Dropout(droupot_para)(concat_attention_weights)
    merge_out = Dot(-1, name='merge_select')([concat_attention_weights, concat_permute])



    y_out = Dense(1, activation='sigmoid')(merge_out)

    model = Model(inputs=[inputs_aac, inputs_ot], outputs=y_out)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model





# model_R23 = creat_model_R23()
# print(model_R23.summary())
# plot_model(model_R23, to_file='ACEP_model_R23.png',show_shapes=True)
# model_R23.fit([x_train_aac, x_train_onehot], y_train, batch_size=16, epochs=30)
# y_pre_all = model_R23.predict([x_test_aac, x_test_onehot])
# y_pre_1_29 = model_R23.predict([x_test_aac1_29, x_test_onehot1_29])
# y_pre_30_200 = model_R23.predict([x_test_aac30_200, x_test_onehot30_200])
# print('####################       R2+R3        ######################')
# print('all__________________________________')
# evaluate_model(y_pre_all,y_test)
# print('1_29_________________________________')
# evaluate_model(y_pre_1_29,y_test_onehot1_29)
# print('30_200_______________________________')
# evaluate_model(y_pre_30_200,y_test_onehot30_200)
# print()




def creat_model_R123():
    droupot_para = 0.3
    e_dim = 64
    feture_dim = 64

    inputs_aac = Input(shape=(21,),name='main_input_aac')
    hidden_acc = Dense(100, activation='relu')(inputs_aac)
    hidden_acc = Dropout(droupot_para)(hidden_acc)
    hidden_acc = Dense(64, activation='relu')(hidden_acc)
    hidden_acc = Dropout(droupot_para)(hidden_acc)
    hidden_acc = Reshape((1,-1))(hidden_acc)


    inputs_ot = Input(shape=(200,21),name='main_input_ot')
    emb_ot = EmbeddingRST_model(input_dim=21,output_dim=e_dim,input_length=200,
                             name='emb_tensor_ot')(inputs_ot)
    cov1d_ot = Conv1D(64, 16, activation='relu',strides=1, padding='same',
                   kernel_initializer='random_normal')(emb_ot) 
    maxpool_ot = MaxPooling1D(3)(cov1d_ot) 
    lstm_out_ot = LSTM(feture_dim,dropout=droupot_para,return_sequences=True,unroll=True)(maxpool_ot)



    inputs_pm = Input(shape=(200,21),name='main_input_pm')
    emb_pm = EmbeddingRST_model(input_dim=21,output_dim=e_dim,input_length=200,
                             name='emb_tensor_pm')(inputs_pm) 
    cov1d_pm = Conv1D(64, 16, activation='relu',strides=1, padding='same',
                   kernel_initializer='random_normal')(emb_pm)  
    maxpool_pm = MaxPooling1D(5)(cov1d_pm)   
    lstm_out_pm = LSTM(feture_dim,dropout=droupot_para,return_sequences=True,unroll=True)(maxpool_pm)  



    lstm_out_permute_ot = Permute((2,1))(lstm_out_ot)    
    attention_weights_ot = TimeDistributed(Dense(1))(lstm_out_ot)  
    attention_weights_ot = Flatten()(attention_weights_ot)  
    attention_weights_ot = Activation('softmax',name='attention_weights_ot')(attention_weights_ot)  
    attention_weights_ot = Dropout(droupot_para)(attention_weights_ot)
    merge_out_ot = Dot(-1,name='merge_out1')([attention_weights_ot,lstm_out_permute_ot]) 
    merge_out_ot = Reshape((1,-1))(merge_out_ot)



    lstm_out_permute_pm = Permute((2,1))(lstm_out_pm)    
    attention_weights_pm = TimeDistributed(Dense(1))(lstm_out_pm)    
    attention_weights_pm = Flatten()(attention_weights_pm)   
    attention_weights_pm = Activation('softmax',name='attention_weights_pm')(attention_weights_pm)    
    attention_weights_pm = Dropout(droupot_para)(attention_weights_pm)
    merge_out_pm = Dot(-1,name='merge_out2')([attention_weights_pm,lstm_out_permute_pm]) 
    merge_out_pm = Reshape((1,-1))(merge_out_pm)



    concat_one_pssm = Concatenate(axis=1)([hidden_acc,merge_out_ot, merge_out_pm])
    cov_f = Conv1D(feture_dim, 3, activation='relu',strides=1, padding='valid',
                   kernel_initializer='random_normal')(concat_one_pssm)
    concat_one_pssm_aac = Concatenate(axis=2)([hidden_acc,cov_f])
    concat_one_pssm_ot = Concatenate(axis=2)([merge_out_ot, cov_f])
    concat_one_pssm_pm = Concatenate(axis=2)([merge_out_pm, cov_f])
    concat_3 = Concatenate(axis=1)([concat_one_pssm_aac,concat_one_pssm_ot,concat_one_pssm_pm])



    concat_permute = Permute((2, 1))(concat_one_pssm)
    concat_attention_weights = TimeDistributed(Dense(1))(concat_3)
    concat_attention_weights = Flatten()(concat_attention_weights)
    concat_attention_weights = Activation('softmax', name='select_weights')(concat_attention_weights)
    concat_attention_weights = Dropout(droupot_para)(concat_attention_weights)
    merge_out = Dot(-1, name='merge_select')([concat_attention_weights, concat_permute])



    y_out = Dense(1, activation='sigmoid')(merge_out)

    model = Model(inputs=[inputs_aac, inputs_ot, inputs_pm], outputs=y_out)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model



model_R123 = creat_model_R123()
print(model_R123.summary())
#plot_model(model_R123, to_file='ACEP_model_R123.png',show_shapes=True)
model_R123.fit([x_train_aac,x_train_onehot, x_train_pssm], y_train, batch_size=16, epochs=30)
y_pre_all = model_R123.predict([x_test_aac,x_test_onehot, x_test_pssm])
y_pre_1_29 = model_R123.predict([x_test_aac1_29,x_test_onehot1_29, x_test_pssm1_29])
y_pre_30_200 = model_R123.predict([x_test_aac30_200,x_test_onehot30_200, x_test_pssm30_200])
print('####################       R1+R2+R3        ######################')
print('all__________________________________')
evaluate_model(y_pre_all,y_test)
print('1_29_________________________________')
evaluate_model(y_pre_1_29,y_test_pssm1_29)
print('30_200_______________________________')
evaluate_model(y_pre_30_200,y_test_pssm30_200)
print()
