
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
from keras.models import load_model

import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.callbacks import TensorBoard

import pandas as pd
import numpy as np
import seaborn as sns

import os
from feature_generation_train import ConvertSequence2Feature
from model_evaluation import evaluation

data_all = pd.read_csv('AMPs_Experiment_Dataset\\AMP_sequecnces\\seq_all_data.csv', index_col=0)

train = data_all.iloc[0:1424].reset_index(drop=True)
tune = data_all.iloc[1424:2132].reset_index(drop=True)
test = data_all.iloc[2132:3556].reset_index(drop=True)
train_tune = data_all.iloc[0:2132].reset_index(drop=True)
train_tune_test = data_all.iloc[0:3556].reset_index(drop=True)



x_train_index, x_train_length, x_train_pssm, x_train_onehot,x_train_aac,y_train = ConvertSequence2Feature(sequence_data=train, pssmdir=os.path.join('AMPs_Experiment_Dataset','PSSM_files','train'))
x_test_index, x_test_length, x_test_pssm, x_test_onehot, x_test_aac, y_test = ConvertSequence2Feature(sequence_data=test, pssmdir=os.path.join('AMPs_Experiment_Dataset','PSSM_files','test'))
#x_tune_index, x_tune_length, x_tune_pssm, x_tune_onehot,x_tune_aac,y_tune= ConvertSequence2Feature(sequence_data=tune, pssmdir=os.path.join('AMPs_Experiment_Dataset','PSSM_files','tune'))
#x_train_tune_index, x_train_tune_length, x_train_tune_pssm, x_train_tune_onehot,x_train_tune_aac,y_train_tune= ConvertSequence2Feature(sequence_data=train_tune, pssmdir=os.path.join('AMPs_Experiment_Dataset','PSSM_files','train_tune'))
#x_train_tune_test_index, x_train_tune_test_length, x_train_tune_test_pssm, x_train_tune_test_onehot,x_train_tune_test_aac,y_train_tune_test= ConvertSequence2Feature(sequence_data=train_tune_test, pssmdir=os.path.join('AMPs_Experiment_Dataset','PSSM_files','train_tune_test'))



model = load_model('models//ACEP_model_train_241_9304.h5',
                        custom_objects={'EmbeddingRST_model': EmbeddingRST_model})
print(model.evaluate([x_test_aac,x_test_onehot, x_test_pssm], y_test, batch_size=16, verbose=0))
print(model.summary())
#plot_model(model, to_file='test1.png',show_shapes=True)
evaluation(model.predict([x_test_aac,x_test_onehot, x_test_pssm]).flatten(), y_test)



from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score



#-------------------------attention intendity---------------------------------------------
attention_weights = Model(inputs=model.input,
                                 outputs=model.get_layer('attention_weights_pm').output)
#attention_weights.summary()
x_pm_p= x_train_pssm[0:712]
x_ot_p= x_train_onehot[0:712]
x_aac_p= x_train_aac[0:712]
attention_weights = attention_weights.predict([x_aac_p,x_ot_p,x_pm_p])

motifts_index = [245,206,356,67,26,18,135,105,216,269]
attention_weights_p10 = attention_weights[motifts_index,:]

#print(train.iloc[:,0])
#print(attention_weights)
col_lab = list(range(1,41))
col_lab = ['P'+"{:0>2d}".format(i) for i in col_lab]
col_lab.insert(0,'Sequences')
attention_seq = pd.DataFrame(data = np.hstack([train.iloc[0:712,0].values.reshape(-1,1),attention_weights]),
                                columns=col_lab)
#print(attention_motifs)
# print(np.max(attention_weights.flatten()))
# print(np.min(attention_weights.flatten()))
# print(np.mean(attention_weights.flatten()))
attention_seq.to_csv('experiment_results\\attention_motifs.csv',index=False)

motifs_list = []
attention_motifs_np = attention_seq.values
for i in range(712):
    for j in range(1,41):
        if attention_motifs_np[i,j]>= 0.2:
            sequence_str = attention_motifs_np[i,0]
            str_end = (40-j)*5
            if str_end+5 <= len(sequence_str) and str_end != 0:
                hit_str = sequence_str[-(str_end+5):-(str_end)]
            else:
                if str_end == 0:
                    hit_str = sequence_str[-(str_end+5):]
                else:
                    hit_str = sequence_str[0:-str_end]
                    hit_str = '-'*(5-len(hit_str)) + hit_str
            #print(hit_str)
            motifs_list.append([hit_str,1])

#print(motifs_list)

def str_align(x,y,str1_len,str2_len):
    count = 0
    #print(x,'   ',y)
    for i in range(str1_len):
        iter_len1 = str1_len-(i+1)
        pos = y.find(x[i],0,str2_len)
        if pos != -1:
            iter_len2 = str2_len-(pos+1)
            if iter_len1>0 and iter_len2>0:
                count = str_align(x[i+1:],y[pos+1:],iter_len1,iter_len2)
            return count + 1
    return count

def find_align(s1,s2,max_align):
    for i in range(len(s1)-max_align+1):
        align_length = str_align(s1[i:],s2,len(s1[i:]),len(s2))
        #print(align_length)
        if align_length>=max_align:
            return True
    return False

#find_align('a0bcde000','120ab000c',4)


motifs_group = []
motifs_group_count = []
motifs_list_len = len(motifs_list)
for i in range(motifs_list_len-1):
    if motifs_list[i][1] == 1:
        one_group = [motifs_list[i][0]]
        one_count = 1
        for j in range(i+1,motifs_list_len):
            if motifs_list[j][1] == 1 and find_align(motifs_list[i][0],motifs_list[j][0],3):
                one_group.append(motifs_list[j][0])
                one_count = one_count+1
                motifs_list[j][1] = 0
        motifs_group.append(one_group)
        motifs_group_count.append(one_count)
if motifs_list[-1][1]==1:
    motifs_group.append(motifs_list[-1][1])
    motifs_group_count.append(1)

#print(motifs_group)
#print(motifs_group_count)

motifs_sort = list(zip(motifs_group,motifs_group_count))
motifs_sort.sort(key= lambda x : x[1],reverse=True)
for i in motifs_sort:
    print(i[1])
    print(i[0])
motifs_sort_goup,motifs_sort_count = zip(*motifs_sort)
motifts_count_np = np.array(motifs_sort_count)
motifts_group_np = np.array(list(motifs_sort_goup))
motifts_count_group_np = np.hstack([motifts_count_np.reshape(-1,1),motifts_group_np.reshape(-1,1)])
motifts_count_group_pd = pd.DataFrame(motifts_count_group_np)
motifts_count_group_pd.to_csv('experiment_results\\motifs_group_rank.csv',index=False)

#-------------------------------------------------------


samp_seq = train.iloc[motifts_index,0]
print(samp_seq)

def cut_str(x,n):
    cut_list = []
    seg = int(len(x)/n)
    seg_mod = len(x)%n
    if seg_mod!=0:
        cut_list.append(x[0:seg_mod])
    for i in range(seg):
        cut_list.append(x[seg_mod+i*5:seg_mod+i*5+5])
    return cut_list

def format_seq(x):
    format_list = []
    for i in range(len(x)):
        cut_list = cut_str(x[i],5)
        pad_list = ['-']*(40-len(cut_list))
        pad_list.extend(cut_list)
        format_list.append(pad_list)
    return format_list

sequence_list = format_seq(samp_seq.values.tolist())
sequen_pd = np.array(sequence_list)[:,30:40]
#print(sequen_pd)


pos_lab = list(range(1,41))
pos_lab = ['P'+"{:0>2d}".format(i) for i in pos_lab]
seq_lab = list(range(1,11))
seq_lab = ['Seq'+str(i) for i in seq_lab]
att_p10_pd = pd.DataFrame(data=attention_weights_p10,index=seq_lab,
                            columns=pos_lab)
att_p10_pd_2 = att_p10_pd.iloc[:,30:40]
#print(att_p10_pd_2)

sns.set(style="white")
f, axes = plt.subplots(nrows=1, ncols=1,figsize=(10, 5))

cmap = sns.diverging_palette(220, 10, as_cmap=True)
#cmap='YlGnBu'
g = sns.heatmap(att_p10_pd_2, annot=sequen_pd,fmt="s",
                cmap=cmap, vmax=.3, center=0.025, linewidths=.5, cbar_kws={"shrink": .5},ax=axes)
#g = sns.heatmap(att_p10_pd_2, cmap=cmap, vmax=.3, center=0, linewidths=.2, square=True, cbar_kws={"shrink": .5},ax=axes)
plt.setp(g.get_yticklabels(), rotation='horizontal')
plt.setp(g.get_xticklabels(), rotation='horizontal')
plt.title('Attention intensity of 10 AMPs',fontsize=14)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
#plt.savefig('experiment_results//motifts_31_40.png',dpi=600,format='png')
plt.show()

