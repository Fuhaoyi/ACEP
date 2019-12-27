import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import binarize
from sklearn import metrics



def evaluate_model(label,pred_prob):
    y_pred_prob = pred_prob
    y_label = label
    y_pred_class = binarize([y_pred_prob], 0.5)[0]
    # print(y_pred_prob)
    # print(y_pred_class)

    confusion = metrics.confusion_matrix(y_label, y_pred_class)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    print(confusion)

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



comparison_data = pd.read_csv('comparison_data_ACEP.csv')
label = comparison_data['label'].values
model_best_pred_prob = comparison_data['ACEP'].values


AMPScanner = comparison_data['AMPScanner'].values
CAMPr3_SVM = comparison_data['CAMPr3_SVM'].values
CAMPr3_RF = comparison_data['CAMPr3_RF'].values
CAMPr3_ANN = comparison_data['CAMPr3_ANN'].values
CAMPr3_DA = comparison_data['CAMPr3_DA'].values


print('-------------model093_pred_prob-------------')
evaluate_model(label, model_best_pred_prob)
print()
print('-------------AMPScanner-------------')
evaluate_model(label, AMPScanner)
print()
print('-------------CAMPr3_SVM-------------')
evaluate_model(label, CAMPr3_SVM)
print()
print('-------------CAMPr3_RF-------------')
evaluate_model(label, CAMPr3_RF)
print()
print('-------------CAMPr3_ANN-------------')
evaluate_model(label, CAMPr3_ANN)
print()
print('-------------CAMPr3_DA-------------')
evaluate_model(label, CAMPr3_DA)
print()


import seaborn as sns
sns.set(style="white")

fpr, tpr, thresholds = metrics.roc_curve(label, model_best_pred_prob)
plt.plot(fpr, tpr, label='Our Method:97.23%')
fpr, tpr, thresholds = metrics.roc_curve(label, AMPScanner)
plt.plot(fpr, tpr, label='AMPScanner:96.30%')
fpr, tpr, thresholds = metrics.roc_curve(label, CAMPr3_RF)
plt.plot(fpr, tpr, label='CAMPr3_RF:93.63')
fpr, tpr, thresholds = metrics.roc_curve(label, CAMPr3_SVM)
plt.plot(fpr, tpr, label='CAMPr3_SVM:90.62%')
fpr, tpr, thresholds = metrics.roc_curve(label, CAMPr3_DA)
plt.plot(fpr, tpr, label='CAMPr3_DA:89.97%')
fpr, tpr, thresholds = metrics.roc_curve(label, CAMPr3_ANN)
plt.plot(fpr, tpr, label='CAMPr3_ANN:84.05%')


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
plt.title('ROC curve for diabetes classifier',fontdict=text_font3)
plt.legend()
plt.xlabel('False Positive Rate (1 - Specificity)',fontdict=text_font4)
plt.ylabel('True Positive Rate (Sensitivity)',fontdict=text_font4)
plt.grid(linestyle='--')


plt.savefig('fig5.svg',dpi=600,format='svg')
plt.show()