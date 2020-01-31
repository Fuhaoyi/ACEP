
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



x_train_index, x_train_length, x_train_pssm, x_train_onehot,x_train_aac,y_train= ConvertSequence2Feature(sequence_data=train, pssmdir=os.path.join('AMPs_Experiment_Dataset','PSSM_files','train'))
x_test_index, x_test_length, x_test_pssm, x_test_onehot, x_test_aac, y_test = ConvertSequence2Feature(sequence_data=test, pssmdir=os.path.join('AMPs_Experiment_Dataset','PSSM_files','test'))
x_tune_index, x_tune_length, x_tune_pssm, x_tune_onehot,x_tune_aac,y_tune= ConvertSequence2Feature(sequence_data=tune, pssmdir=os.path.join('AMPs_Experiment_Dataset','PSSM_files','tune'))
x_train_tune_index, x_train_tune_length, x_train_tune_pssm, x_train_tune_onehot,x_train_tune_aac,y_train_tune= ConvertSequence2Feature(sequence_data=train_tune, pssmdir=os.path.join('AMPs_Experiment_Dataset','PSSM_files','train_tune'))
x_train_tune_test_index, x_train_tune_test_length, x_train_tune_test_pssm, x_train_tune_test_onehot,x_train_tune_test_aac,y_train_tune_test= ConvertSequence2Feature(sequence_data=train_tune_test, pssmdir=os.path.join('AMPs_Experiment_Dataset','PSSM_files','train_tune_test'))



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
plt.xticks(fontsize=16)
plt.savefig('experiment_results\\Heatmap_AMP10_1.png',dpi=600,format='png')
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
plt.savefig('experiment_results\\Average_AMP500_1.png',dpi=600,format='png')
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
plt.savefig('experiment_results//Heatmap_AMP10_2.png',dpi=600,format='png')
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
plt.savefig('experiment_results\\Average_AMP500_2.png',dpi=600,format='png')
plt.show()


