
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



#--------------------------Fusion feature vector--------------------------------------------
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


sns.set(style="darkgrid")
sns.scatterplot(x="Dim1", y="Dim2",
            hue="Category",
            data=feature_vector_pd)
plt.title('The feature vectors of AMPs and non-AMPs',fontdict=text_font3)
plt.savefig('experiment_results//Fusion_feature.png',dpi=600,format='png')

plt.show()


