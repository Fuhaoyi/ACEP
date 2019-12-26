
from EmbeddingRST import EmbeddingRST_model


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
            #m.extend([int(i) for i in oneline[22:42]])   #--------------------------
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
    return sequence_index_out, padding_vst_pssm, sequence_label, sequence_pssm   #numpy  shape:(None,200,21)  (None,)



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

    #return sequence_index_out, one_hot_matrix, sequence_label   #numpy   shape:(None,200,21) (None,)
    return one_hot_matrix   #numpy   shape:(None,200,21) (None,)

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
    #sequence_aac = sequence_aac/len(sequence_num)
    sequence_label = label_x
    sequence_index_out = sequence_index_x

    #return sequence_index_out, sequence_aac, sequence_label
    return sequence_aac


def ConvertSequence2Feature(sequence_data, pssmdir):
    sequence_count = sequence_data.shape[0]
    #print(sequence_data)
    feature_pssm = np.zeros((sequence_count, 200, 21), dtype=np.float32)
    feature_onehot = np.zeros((sequence_count, 200, 21), dtype=np.float32)
    feature_aac = np.zeros((sequence_count, 21), dtype=np.float32)

    label_list = np.zeros((sequence_count), dtype=np.int32)
    index_out = np.zeros((sequence_count), dtype=np.int32)
    pssm_list, pssm_file_list = getpssmlist(pssmdir)
    #print(pssm_file_list)
    length_list = []
    for i in range(sequence_count):
        #length_list.append(len(sequence_data.iloc[i, 0])/sequence_data.iloc[i, 2])    #------------------------
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



data_all = pd.read_csv('pssm_files0_3555_all\seq_all_data.csv', index_col=0)

train = data_all.iloc[0:1424].reset_index(drop=True)
tune = data_all.iloc[1424:2132].reset_index(drop=True)
test = data_all.iloc[2132:3556].reset_index(drop=True)
train_tune = data_all.iloc[0:2132].reset_index(drop=True)
train_tune_test = data_all.iloc[0:3556].reset_index(drop=True)


x_train_index, x_train_length, x_train_pssm, x_train_onehot, x_train_aac, y_train = ConvertSequence2Feature(sequence_data=train, pssmdir=os.path.join('pssm_files0_3555_all','cut','train'))
x_test_index, x_test_length, x_test_pssm, x_test_onehot, x_test_aac, y_test = ConvertSequence2Feature(sequence_data=test, pssmdir=os.path.join('pssm_files0_3555_all','cut','test'))
x_tune_index, x_tune_length, x_tune_pssm, x_tune_onehot, x_tune_aac, y_tune = ConvertSequence2Feature(sequence_data=train, pssmdir=os.path.join('pssm_files0_3555_all','cut','train'))
x_train_tune_index, x_train_tune_length, x_train_tune_pssm, x_train_tune_onehot,x_train_tune_aac,y_train_tune= ConvertSequence2Feature(sequence_data=train_tune, pssmdir=os.path.join('pssm_files0_3555_all','cut','train_tune'))
x_train_tune_test_index, x_train_tune_test_length, x_train_tune_test_pssm, x_train_tune_test_onehot,x_train_tune_test_aac,y_train_tune_test= ConvertSequence2Feature(sequence_data=train_tune_test, pssmdir=os.path.join('pssm_files0_3555_all','cut','train_tune_test'))



model = load_model('result_data//ACEP_model_train_241_9304.h5',
                        custom_objects={'EmbeddingRST_model': EmbeddingRST_model})
print(model.evaluate([x_test_aac,x_test_onehot, x_test_pssm], y_test, batch_size=16, verbose=0))
print(model.summary())
plot_model(model, to_file='ACEP_model_train_241_9304.png',show_shapes=True)


# print(model.input)
# print(model.output)
# print(model.get_layer('main_input').input)
# print(model.get_layer('main_input').output)
# print(model.get_layer('dense_2').input)
# print(model.get_layer('dense_2').output)


from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


#---------------------------嵌入张量-------------------------------------
aac_alphabet=['X','A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
Emb_Tensor = model.get_layer('emb_tensor_pm').get_weights()[0]
dims64 = ['D'+str(i) for i in range(1,65)]
#print(Emb_Tensor.shape)

Emb_Tensor_pd = pd.DataFrame(data=Emb_Tensor,index=aac_alphabet,
                            columns=dims64)
print(Emb_Tensor_pd.head())
Emb_Tensor_pd.to_csv('embedding_tensor_64.csv')



#-------------------------注意力强度---------------------------------------------
attention_weights = Model(inputs=model.input,
                                 outputs=model.get_layer('attention_weights_pm').output)
#attention_weights.summary()
x_pm_p, x_pm_n= x_train_tune_pssm[0:500],x_train_tune_pssm[700:1200]
x_ot_p, x_ot_n= x_train_tune_onehot[0:500],x_train_tune_onehot[700:1200]
x_aac_p, x_aac_n= x_train_tune_aac[0:500],x_train_tune_aac[700:1200]
#print(y_train[0:500],y_train[700:1200])
attention_weights_p = attention_weights.predict([x_aac_p,x_ot_p,x_pm_p])
attention_weights_n = attention_weights.predict([x_aac_n,x_ot_n,x_pm_n])

attention_weights_p10 = attention_weights_p[300:310,:]

#print(attention_weights_p)
attention_weights_p_avg = np.mean(attention_weights_p,0)
attention_weights_n_avg = np.mean(attention_weights_n,0)
#print(attention_weights_p_avg)

att_pos = np.array(range(1,41))
att_pos = att_pos.tolist()
att_pos = ['P'+"{:0>2d}".format(i) for i in att_pos]

seq_lab = list(range(1,501))
seq_lab = ['Seq'+ '{:0>3d}'.format(i) for i in seq_lab]
attention_weights_p_pd = pd.DataFrame(data=attention_weights_p,index=seq_lab,
                            columns=att_pos)
print(attention_weights_p_pd.head())
attention_weights_p_pd.to_csv('attention_intensity500_R1.csv')


#------------------------u----------------------------------


select_weights = Model(inputs=model.input,
                                 outputs=model.get_layer('select_weights').output)

select_np = select_weights.predict([x_train_tune_aac, x_train_tune_onehot, x_train_tune_pssm], batch_size=16, verbose=0)
#print(select_np,select_np.shape)

length_np = np.array(x_train_tune_length)

sel_length = np.hstack([length_np.reshape(-1,1),select_np])

seq_lab = list(range(1,select_np.shape[0]+1))
seq_lab = ['Seq'+ '{:0>3d}'.format(i) for i in seq_lab]

sel_length_pd = pd.DataFrame(data=sel_length,index=seq_lab,
                            columns=['Length','R3','R2','R1'])
print(sel_length_pd.head())
sel_length_pd.to_csv('attention_intensity_R4.csv')




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



import pandas as pd



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
plt.savefig('supplementary_data//s11.svg',dpi=600,format='svg')


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
plt.savefig('supplementary_data//s12.svg',dpi=600,format='svg')

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
plt.savefig('supplementary_data//s13.svg',dpi=600,format='svg')

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
plt.savefig('supplementary_data//s14.svg',dpi=600,format='svg')

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
plt.savefig('supplementary_data//s15.svg',dpi=600,format='svg')

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
plt.savefig('supplementary_data//s16.svg',dpi=600,format='svg')

plt.show()


from sklearn.preprocessing import binarize

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
err_seqence_pd.to_csv('supplementary_data\\err_seqence_adp.csv')


all_p_c,all_p_l,all_n_c,all_n_l = length_n_p(x_train_tune_test_length,y_train_tune_test)
print(all_p_c,all_n_c)
amp_pd = pd.Series(all_p_l)
print(amp_pd.describe())