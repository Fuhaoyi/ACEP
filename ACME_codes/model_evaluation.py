from sklearn.preprocessing import binarize
from sklearn import metrics

def evaluation(y1,y2):
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