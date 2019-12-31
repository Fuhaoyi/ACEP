
from keras import backend


from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Lambda, Permute, Reshape, Flatten, Masking
from keras.layers import Multiply, Dot, RepeatVector,Concatenate,Add,Lambda
from keras.layers import Embedding
from keras.layers import LSTM,Bidirectional,BatchNormalization
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import TimeDistributed

import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.callbacks import TensorBoard

import pandas as pd
import numpy as np

from EmbeddingRST import EmbeddingRST_model
from feature_generation_train import ConvertSequence2Feature

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



def creat_model():
    droupot_para = 0.3
    e_dim = 64
    feture_dim = 64

    inputs_aac = Input(shape=(21,),name='main_input_aac')
    hidden_acc = Dense(100, activation='relu')(inputs_aac)
    hidden_acc = Dropout(droupot_para)(hidden_acc)
    hidden_acc = Dense(feture_dim, activation='relu')(hidden_acc)
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




from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import binarize
from sklearn import metrics


x_all_aac = np.vstack([x_train_aac,x_test_aac])
x_all_ot = np.vstack([x_train_onehot,x_test_onehot])
x_all_pm = np.vstack([x_train_pssm,x_test_pssm])
y_all = np.hstack([y_train,y_test])


import time


def CV_fun(randomstate=0):
    print('###########------   randomstate  %d   ------############'%randomstate)
    X1 = x_all_aac
    X2 = x_all_ot
    X3 = x_all_pm
    Y = y_all
    skf = StratifiedKFold(n_splits=10, random_state=randomstate, shuffle=True)
    count = 0
    sens_cv = np.array([])
    sepc_cv = np.array([])
    acc_cv = np.array([])
    mcc_cv = np.array([])
    auc_cv = np.array([])


    for train_index, test_index in skf.split(X1, Y):
        x_cv_tarin_aac, x_cv_tarin_ot, x_cv_tarin_pm, y_cv_train = X1[train_index], X2[train_index], X3[train_index], Y[
            train_index]
        x_cv_test_aac, x_cv_test_ot, x_cv_test_pm, y_cv_test = X1[test_index], X2[test_index], X3[test_index], Y[
            test_index]
        model0_1 = creat_model()
        model0_1.fit([x_cv_tarin_aac, x_cv_tarin_ot, x_cv_tarin_pm], y_cv_train, batch_size=16, epochs=30, verbose=0)
        y_pred_prob = model0_1.predict([x_cv_test_aac, x_cv_test_ot, x_cv_test_pm])

        del model0_1
        backend.clear_session()

        count = count + 1
        print('--------------', 'CV  ', count, '---------------------')


        y_pred_prob = y_pred_prob.flatten()
        y_pred_class = binarize([y_pred_prob], 0.5)[0]


        confusion = metrics.confusion_matrix(y_cv_test, y_pred_class)
        print(confusion)
        TN, FP, FN, TP = metrics.confusion_matrix(y_cv_test, y_pred_class).ravel()
        print('TN:', TN, 'FP:', FP, 'FN:', FN, 'TP:', TP)

        ACC = metrics.accuracy_score(y_cv_test, y_pred_class)
        print('Accuracy:', ACC)
        acc_cv = np.append(acc_cv, ACC)

        Error = 1 - metrics.accuracy_score(y_cv_test, y_pred_class)
        print('Error:', Error)

        if (TP + FN != 0):
            Sens = metrics.recall_score(y_cv_test, y_pred_class)
        else:
            Sens = 1
        print('Sensitivity:', Sens)
        sens_cv = np.append(sens_cv, Sens)

        if (TN + FP != 0):
            Spec = TN / float(TN + FP)
        else:
            Spec = 1
        print('Specificity:', Spec)
        sepc_cv = np.append(sepc_cv, Spec)

        FPR = 1 - Spec
        print('False Positive Rate:', FPR)

        if (TP + FP != 0):
            Precision = metrics.precision_score(y_cv_test, y_pred_class)
        else:
            Precision = 1
        print('Precision:', Precision)

        if (Precision != 0 and Sens != 0):
            F1_score = metrics.f1_score(y_cv_test, y_pred_class)
        else:
            F1_score = 0.5
        print('F1 score:', F1_score)

        if (TP + FP == 0 or TP + FN == 0 or TN + FP == 0 or TN + FN == 0):
            MCC = 1
        else:
            MCC = metrics.matthews_corrcoef(y_cv_test, y_pred_class)
        print('Matthews correlation coefficient:', MCC)
        mcc_cv = np.append(mcc_cv, MCC)

        if (TP + FN != 0 and TN + FP != 0):
            AUC = metrics.roc_auc_score(y_cv_test, y_pred_prob)
        else:
            AUC = 1
        print('ROC Curves and Area Under the Curve (AUC):', AUC)
        auc_cv = np.append(auc_cv, AUC)

        #if (count % 5 == 0):
         #   time.sleep(600)



    print()
    print('---------------END------------')
    print('Sensitivity:', '   ', sens_cv.mean(), '(', sens_cv.std(), ')')
    print('Specificity:', '   ', sepc_cv.mean(), '(', sepc_cv.std(), ')')
    print('ACC:', '   ', acc_cv.mean(), '(', acc_cv.std(), ')')
    print('MCC:', '   ', mcc_cv.mean(), '(', mcc_cv.std(), ')')
    print('AUC:', '   ', auc_cv.mean(), '(', auc_cv.std(), ')')
    print('\n\n\n')


for i in range(3):
    CV_fun(i)
