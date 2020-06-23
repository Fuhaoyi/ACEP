import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import binarize
from sklearn import metrics
from model_evaluation import evaluation




comparison_data = pd.read_csv('experiment_results\\comparison_data_ACEP.csv')
label = comparison_data['label'].values
model_best_pred_prob = comparison_data['ACEP'].values


AMPScanner = comparison_data['AMPScanner'].values
CAMPr3_SVM = comparison_data['CAMPr3_SVM'].values
CAMPr3_RF = comparison_data['CAMPr3_RF'].values
CAMPr3_ANN = comparison_data['CAMPr3_ANN'].values
CAMPr3_DA = comparison_data['CAMPr3_DA'].values


print('-------------model093_pred_prob-------------')
evaluation(model_best_pred_prob, label)
print()
print('-------------AMPScanner-------------')
evaluation(AMPScanner, label)
print()
print('-------------CAMPr3_SVM-------------')
evaluation(CAMPr3_SVM, label)
print()
print('-------------CAMPr3_RF-------------')
evaluation(CAMPr3_RF, label)
print()
print('-------------CAMPr3_ANN-------------')
evaluation(CAMPr3_ANN, label)
print()
print('-------------CAMPr3_DA-------------')
evaluation(CAMPr3_DA, label)
print()


import seaborn as sns
sns.set(style="white")

fpr, tpr, thresholds = metrics.roc_curve(label, model_best_pred_prob)
plt.plot(fpr, tpr, label='ACEP:97.78%')
fpr, tpr, thresholds = metrics.roc_curve(label, AMPScanner)
plt.plot(fpr, tpr, label='AMPScanner:96.30%')
fpr, tpr, thresholds = metrics.roc_curve(label, CAMPr3_RF)
plt.plot(fpr, tpr, label='CAMPr3-RF:93.63%')
fpr, tpr, thresholds = metrics.roc_curve(label, CAMPr3_SVM)
plt.plot(fpr, tpr, label='CAMPr3-SVM:90.62%')
fpr, tpr, thresholds = metrics.roc_curve(label, CAMPr3_DA)
plt.plot(fpr, tpr, label='CAMPr3-DA:89.97%')
fpr, tpr, thresholds = metrics.roc_curve(label, CAMPr3_ANN)
plt.plot(fpr, tpr, label='CAMPr3-ANN:84.05%')


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


plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for various methods',fontdict=text_font3)
plt.legend()
plt.xlabel('False Positive Rate (1 - Specificity)',fontdict=text_font4)
plt.ylabel('True Positive Rate (Sensitivity)',fontdict=text_font4)
plt.grid(linestyle='--')


plt.savefig('experiment_results\\ROC_curve.png',dpi=600,format='png')
plt.show()