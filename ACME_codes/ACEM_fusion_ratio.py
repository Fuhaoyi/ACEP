
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

#---------------------------Fusion ratio------------------------------------
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

sns.set(style="darkgrid")

sns.lineplot(x="length_class", y="attention",hue = 'Ratio', marker='o',
             data=select_pd)

plt.title('Sequence length and fusion ratio',fontdict=text_font3)
plt.xlabel('The length of the sequence',fontdict=text_font4)
plt.ylabel('The fusion ratio',fontdict=text_font4)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('experiment_results//Fusion_ratio.png',dpi=600,format='png')
plt.show()






