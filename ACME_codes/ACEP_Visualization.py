
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



data_all = pd.read_csv('pssm_files0_3555_all\\seq_all_data.csv', index_col=0)

train = data_all.iloc[0:1424].reset_index(drop=True)
tune = data_all.iloc[1424:2132].reset_index(drop=True)
test = data_all.iloc[2132:3556].reset_index(drop=True)
train_tune = data_all.iloc[0:2132].reset_index(drop=True)
train_tune_test = data_all.iloc[0:3556].reset_index(drop=True)



# x_train_index, x_train_length, x_train_pssm, x_train_onehot,x_train_aac,y_train= ConvertSequence2Feature(sequence_data=train, pssmdir=os.path.join('pssm_files0_3555_all','cut','train'))
x_test_index, x_test_length, x_test_pssm, x_test_onehot, x_test_aac, y_test = ConvertSequence2Feature(sequence_data=test, pssmdir=os.path.join('pssm_files0_3555_all','cut','test'))
# x_tune_index, x_tune_length, x_tune_pssm, x_tune_onehot,x_tune_aac,y_tune= ConvertSequence2Feature(sequence_data=tune, pssmdir=os.path.join('pssm_files0_3555_all','cut','tune'))
x_train_tune_index, x_train_tune_length, x_train_tune_pssm, x_train_tune_onehot,x_train_tune_aac,y_train_tune= ConvertSequence2Feature(sequence_data=train_tune, pssmdir=os.path.join('pssm_files0_3555_all','cut','train_tune'))
# x_train_tune_test_index, x_train_tune_test_length, x_train_tune_test_pssm, x_train_tune_test_onehot,x_train_tune_test_aac,y_train_tune_test= ConvertSequence2Feature(sequence_data=train_tune_test, pssmdir=os.path.join('pssm_files0_3555_all','cut','train_tune_test'))



model = load_model('result_data//ACEP_model_train_241_9304.h5',
                        custom_objects={'EmbeddingRST_model': EmbeddingRST_model})
print(model.evaluate([x_test_aac,x_test_onehot, x_test_pssm], y_test, batch_size=16, verbose=0))
print(model.summary())
#plot_model(model, to_file='test1.png',show_shapes=True)



from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


#---------------------------embedding tensor-------------------------------------
aac_alphabet=['X','A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
Emb_Tensor = model.get_layer('emb_tensor_pm').get_weights()[0]


n_clusters = range(2,21)
sum_distances = []
silhouette = []
for i in n_clusters:
    kmeans = KMeans(n_clusters=i, random_state=0).fit(Emb_Tensor)
    sum_distances.append(kmeans.inertia_)
    silhouette.append(silhouette_score(Emb_Tensor, kmeans.labels_, metric='euclidean'))


n_clusters_np = np.array([n_clusters]).reshape((-1,1))
sum_distances_np = np.array([sum_distances]).reshape((-1,1))
silhouette_np = np.array([silhouette]).reshape((-1,1))
fig3_data = np.hstack([n_clusters_np,sum_distances_np,silhouette_np])
fig3_pd = pd.DataFrame(data=fig3_data,columns=['Clusters','Distances','Silhouette'])
print(fig3_pd.head())


import seaborn as sns
sns.set(style="white")

f, axes = plt.subplots()
sns.lineplot(x="Clusters", y="Distances", data=fig3_pd,marker="o", ax=axes,linewidth=1.5,markersize=7)
axes.plot(5, sum_distances_np[3], 'o', c='r',markersize=7)
text_font1={
    #'family':'Times New Roman',
    'style':'italic',
    'weight':'normal',
      'color':'r',
      'size':20
}
axes.text(6, 4.5, 'k=5', fontdict=text_font1)
text_font2={
    #'family':'Times New Roman',
    #'style':'italic',
    'weight':'normal',
      'color':'k',
      'size':20
}
axes.set_title('Within-cluster sum-of-squares',fontsize=25)
axes.set_xlabel('Clusters k',fontdict=text_font2)
axes.set_ylabel('Distances',fontdict=text_font2)
axes.tick_params(colors='k',labelsize = 20)
plt.grid(linestyle='--')
plt.savefig('result_figure//fig4a.svg',dpi=600,format='svg')
plt.show()


f, axes = plt.subplots()
sns.lineplot(x="Clusters", y="Silhouette",data=fig3_pd,marker="o", ax=axes,linewidth=1.5,markersize=7)
axes.plot(5, silhouette_np[3], 'o', c='r',markersize=7)
text_font1={
    #'family':'Times New Roman',
    'style':'italic',
    'weight':'normal',
      'color':'r',
      'size':20
}
axes.text(5.9, 0.15, 'k=5', fontdict=text_font1)
text_font2={
    #'family':'Times New Roman',
    #'style':'italic',
    'weight':'normal',
      'color':'k',
      'size':20
}
axes.set_title('Average silhouette',fontsize=25)
axes.set_xlabel('Clusters k',fontdict=text_font2)
axes.set_ylabel('Silhouette',fontdict=text_font2)
axes.tick_params(colors='k',labelsize = 20)
plt.grid(linestyle='--')

plt.savefig('result_figure//fig4b.svg',dpi=600,format='svg')
plt.show()

import seaborn as sns
sns.set(style="darkgrid")

clusters = 5
kmeans = KMeans(n_clusters=clusters, random_state=0).fit(Emb_Tensor)
print(kmeans.labels_)
#print(kmeans.labels_.shape)
print(kmeans.cluster_centers_.shape)
#print(kmeans.inertia_)

x_tsne  = TSNE(n_components=2, init='pca', perplexity=5,method='exact',
               random_state=1).fit_transform(np.concatenate([Emb_Tensor,kmeans.cluster_centers_],axis=0))
emb_tsne = x_tsne[0:21]
cluster_centers = x_tsne[21:]
#print(emb_tsne.shape)
#print(cluster_centers.shape)

categ_label = kmeans.labels_+1
categ_label = categ_label.tolist()
categ_label = ['C'+str(i) for i in categ_label]
categ_label = pd.Categorical(categ_label, categories=['C1', 'C2', 'C3', 'C4', 'C5'])

cluster_data = pd.DataFrame({'Dim1':emb_tsne[1:,0],'Dim2':emb_tsne[1:,1],'Cluster':categ_label[1:]})
print(cluster_data)
print(cluster_data.dtypes)


text_font3={
    #'family':'Times New Roman',
    'style':'normal',
    'weight':'normal',
      'color':'k',
      'size':14}

text_font4={
    #'family':'Times New Roman',
    'style':'normal',
    'weight':'normal',
      'color':'k',
      'size':12}

import seaborn as sns
sns.set(style="darkgrid")
sns.scatterplot(x="Dim1", y="Dim2",
            hue="Cluster",style="Cluster",
            data=cluster_data,s=50)

plt.title('Amino acid embedding tensors',fontdict=text_font3)
print(categ_label.tolist())
text_color = ['b','b','b','goldenrod','g','r','goldenrod','g','chocolate',
              'b','b','b','chocolate','b','b','m','goldenrod','goldenrod','b','b','b']
for i in range(1,21):
    plt.text(emb_tsne[i,0]+10, emb_tsne[i,1]+8, aac_alphabet[i], fontsize=14,color=text_color[i])
plt.xlim(-400,400)
plt.ylim(-400,400)
plt.savefig('result_figure//fig6.svg',dpi=600,format='svg')
plt.show()





#-------------------------attention intendity---------------------------------------------
attention_weights = Model(inputs=model.input,
                                 outputs=model.get_layer('attention_weights_pm').output)
#attention_weights.summary()
x_pm_p, x_pm_n= x_train_tune_pssm[0:500],x_train_tune_pssm[700:1200]
x_ot_p, x_ot_n= x_train_tune_onehot[0:500],x_train_tune_onehot[700:1200]
x_aac_p, x_aac_n= x_train_tune_aac[0:500],x_train_tune_aac[700:1200]
attention_weights_p = attention_weights.predict([x_aac_p,x_ot_p,x_pm_p])
attention_weights_n = attention_weights.predict([x_aac_n,x_ot_n,x_pm_n])

a10list=[18,26,52,67,105,135,206,216,224,269]
attention_weights_p10 = attention_weights_p[a10list,:]

attention_weights_p_avg = np.mean(attention_weights_p,0)
attention_weights_n_avg = np.mean(attention_weights_n,0)


pos_lab = list(range(1,41))
pos_lab = ['P'+"{:0>2d}".format(i) for i in pos_lab]
seq_lab = list(range(1,11))
seq_lab = ['Seq'+str(i) for i in seq_lab]
att_p10_T = attention_weights_p10.T
att_p10_T_pd = pd.DataFrame(data=att_p10_T,index=pos_lab,
                            columns=seq_lab)
att_p10_T_pd_2 = att_p10_T_pd.iloc[20:40,:]

sns.set(style="white")
f, axes = plt.subplots(nrows=1, ncols=1,figsize=(5, 10))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

g = sns.heatmap(att_p10_T_pd_2,  cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},ax=axes)
plt.setp(g.get_yticklabels(), rotation='horizontal')
plt.setp(g.get_xticklabels(), rotation='vertical')
plt.title('Attention intensity of 10 AMPs',fontsize=18)

plt.yticks(fontsize=16)
plt.savefig('result_figure//fig7a.svg',dpi=600,format='svg')
plt.show()


att_pos = np.array(range(21,41))
att_pos = att_pos.tolist()
att_pos = ['P'+"{:0>2d}".format(i) for i in att_pos]
att_pos = att_pos*500
att_pos_cata = pd.Categorical(att_pos)
att_wei = attention_weights_p[:,20:40].flatten()
att_pd = pd.DataFrame({'pos':att_pos_cata,'att_wei':att_wei})
print(att_pd)


sns.set(style="darkgrid")
f, axes = plt.subplots(nrows=1, ncols=1,figsize=(5, 10))

sns.set_color_codes("pastel")

g = sns.barplot(x="att_wei", y="pos", data=att_pd, ax=axes)
axes.set(xlim=(0, 0.3), ylabel="",
        xlabel="")
plt.title('Average attention intensity of 500 AMPs',fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.setp(g.get_yticklabels(), rotation='horizontal')
plt.setp(g.get_xticklabels(), rotation='vertical')
plt.savefig('result_figure//fig7b.svg',dpi=600,format='svg')
plt.show()


#--------------------------------------------------------
pos_lab = list(range(1,41))
pos_lab = ['P'+"{:0>2d}".format(i) for i in pos_lab]
seq_lab = list(range(1,11))
seq_lab = ['Seq'+str(i) for i in seq_lab]
att_p10_T = attention_weights_p10.T
att_p10_T_pd = pd.DataFrame(data=att_p10_T,index=pos_lab,
                            columns=seq_lab)
att_p10_T_pd_2 = att_p10_T_pd.iloc[0:40,:]

sns.set(style="white")
f, axes = plt.subplots(nrows=1, ncols=1,figsize=(5, 10))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

g = sns.heatmap(att_p10_T_pd_2,  cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},ax=axes)
plt.setp(g.get_yticklabels(), rotation='horizontal')
plt.setp(g.get_xticklabels(), rotation='vertical')
plt.title('Attention intensity of 10 AMPs',fontsize=14)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('result_figure//fig7a40.svg',dpi=600,format='svg')
plt.show()

att_pos = np.array(range(1,41))
att_pos = att_pos.tolist()
att_pos = ['P'+"{:0>2d}".format(i) for i in att_pos]
att_pos = att_pos*500
att_pos_cata = pd.Categorical(att_pos)
att_wei = attention_weights_p.flatten()
att_pd = pd.DataFrame({'pos':att_pos_cata,'att_wei':att_wei})
print(att_pd)


sns.set(style="darkgrid")
f, axes = plt.subplots(nrows=1, ncols=1,figsize=(5, 10))

sns.set_color_codes("pastel")


g = sns.barplot(x="att_wei", y="pos", data=att_pd, ax=axes)
axes.set(xlim=(0, 0.3), ylabel="",
        xlabel="")
plt.title('Average attention intensity of 500 AMPs',fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.setp(g.get_yticklabels(), rotation='horizontal')
plt.setp(g.get_xticklabels(), rotation='vertical')
plt.savefig('result_figure//fig7b40.svg',dpi=600,format='svg')
plt.show()



#--------------------------Feature fusion vector--------------------------------------------
merge_out = Model(inputs=model.input,
                                 outputs=model.get_layer('merge_select').output)
#merge_out.summary()
samples = np.arange(400,900,1)
x_pm1 = x_train_tune_pssm[samples]
x_ot1 = x_train_tune_onehot[samples]
x_aac1 = x_train_tune_aac[samples]
y_1 = y_train_tune[samples]
merge_out = merge_out.predict([x_aac1,x_ot1,x_pm1])
#print(merge_out,merge_out.shape)
#print(y_small_test,y_small_test.shape)
x_feature_vector = TSNE(n_components=2,init='pca',perplexity=40,method='exact').fit_transform(merge_out)

amp_non = ['null']*samples.shape[0]
for i in range(samples.shape[0]):
    if y_1[i]==1:
        amp_non[i]='Antimicrobial '
    else:
        amp_non[i]='Non-antimicrobial '
amp_non = pd.Categorical(amp_non)
feature_vector_pd = pd.DataFrame({'Dim1':x_feature_vector[:,0],'Dim2':x_feature_vector[:,1],'Category':amp_non})

import seaborn as sns
sns.set(style="darkgrid")
sns.scatterplot(x="Dim1", y="Dim2",
            hue="Category",
            data=feature_vector_pd)
plt.title('The feature vectors of AMPs and non-AMPs',fontdict=text_font3)
plt.savefig('result_figure//fig9.svg',dpi=600,format='svg')

plt.show()


#---------------------------the fusion ratio------------------------------------



select_weights = Model(inputs=model.input,
                                 outputs=model.get_layer('select_weights').output)

select_np = select_weights.predict([x_train_tune_aac, x_train_tune_onehot, x_train_tune_pssm], batch_size=16, verbose=0)


length_np = np.array(x_train_tune_length)

length_class = np.array(np.zeros(select_np.shape[0],))
for i in range(select_np.shape[0]):
    for j in range(11):
        if j*10+10<=length_np[i]<=j*10+19:
            length_class[i] = j*10+10
    if 120<= length_np[i]:
        length_class[i] = 120

items_num = select_np.shape[0]
u1_ = np.array(['u1\'']*items_num)
u2_ = np.array(['u2\'']*items_num)
u3_ = np.array(['u3\'']*items_num)
u123_ = np.hstack([u1_,u2_,u3_])
select_data_np_u1 = np.vstack([length_class,select_np[:,2]])
select_data_np_u1 = select_data_np_u1.T
select_data_np_u2 = np.vstack([length_class,select_np[:,1]])
select_data_np_u2 = select_data_np_u2.T
select_data_np_u3 = np.vstack([length_class,select_np[:,0]])
select_data_np_u3 = select_data_np_u3.T

select_data_np = np.vstack([select_data_np_u1,select_data_np_u2,select_data_np_u3])

select_pd = pd.DataFrame({'length_class':select_data_np[:,0],'attention':select_data_np[:,1],'Ratio':u123_})
print(select_pd.head())

import seaborn as sns
sns.set(style="darkgrid")

sns.lineplot(x="length_class", y="attention",hue = 'Ratio', marker='o',
             data=select_pd)

plt.title('Sequence length and fusion ratio',fontdict=text_font3)
plt.xlabel('The length of the sequence',fontdict=text_font4)
plt.ylabel('The fusion ratio',fontdict=text_font4)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('result_figure//fig8.svg',dpi=600,format='svg')
plt.show()


#---------------------evaluation-------------------------------


from sklearn.preprocessing import binarize
from sklearn import metrics

def evaluate_model(y1,y2):
    y_pred_prob = y1
    y_label = y2
    y_pred_class = binarize([y_pred_prob], 0.5)[0]
    # print(y_pred_prob)
    # print(y_pred_class)

    TN, FP, FN, TP = metrics.confusion_matrix(y_label, y_pred_class).ravel()
    print(metrics.confusion_matrix(y_label, y_pred_class))

    ACC = metrics.accuracy_score(y_label, y_pred_class)
    print('Classification Accuracy:', ACC)

    Error = 1 - metrics.accuracy_score(y_label, y_pred_class)
    print('Classification Error:', Error)

    Sens = metrics.recall_score(y_label, y_pred_class)
    print('Sensitivity:', Sens)

    Spec = TN / float(TN + FP)
    print('Specificity:', Spec)

    FPR = FP / float(TN + FP)
    print('False Positive Rate:', FPR)

    Precision = metrics.precision_score(y_label, y_pred_class)
    print('Precision:', Precision)

    F1_score = metrics.f1_score(y_label, y_pred_class)
    print('F1 score:', F1_score)

    MCC = metrics.matthews_corrcoef(y_label, y_pred_class)
    print('Matthews correlation coefficient:', MCC)

    AUC = metrics.roc_auc_score(y_label, y_pred_prob)
    print('ROC Curves and Area Under the Curve (AUC):', AUC)


evaluate_model(model.predict([x_test_aac,x_test_onehot, x_test_pssm]).flatten(), y_test)



