import pandas as pd
import numpy as np
import os
from EmbeddingRST import EmbeddingRST_model
from feature_generation_pre import ConvertSequence2Feature
from sklearn.preprocessing import binarize


test = pd.read_csv('AMP_prediction\\inputs\\sequences.csv')
x_test_index, x_test_length, x_test_pssm, x_test_onehot, x_test_aac = ConvertSequence2Feature(sequence_data=test, pssmdir=os.path.join('AMP_prediction','inputs','PSSM_files'))



from keras.models import load_model

def save_pred_prob():
    model_best = load_model('models\\ACEP_model_train_241_9304.h5', custom_objects={'EmbeddingRST_model': EmbeddingRST_model})
    print(model_best.summary())
    y_pred_prob = model_best.predict([x_test_aac, x_test_onehot, x_test_pssm])
    y_pred_class = binarize([y_pred_prob.flatten()], 0.5)[0]
    y_pred_prob_np = np.hstack([x_test_index.reshape((-1,1)), test.iloc[:,0].values.reshape(-1,1), y_pred_prob,y_pred_class.reshape((-1,1))])
    y_pred_prob_pd = pd.DataFrame(y_pred_prob_np, columns=['index', 'sequence', 'probability','AMP/nonAMP'])
    y_pred_prob_pd.to_csv('AMP_prediction\\outputs\\outputs.csv')

save_pred_prob()