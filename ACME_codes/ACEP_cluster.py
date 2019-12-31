
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
plt.savefig('experiment_results//Cluster_number1.png',dpi=600,format='png')
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

plt.savefig('experiment_results//Cluster_number2.png',dpi=600,format='png')
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
plt.savefig('experiment_results//Cluster.png',dpi=600,format='png')
plt.show()






