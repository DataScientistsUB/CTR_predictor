'''
COMPARISON OF 2 MODEL'S PREDICTIONS BY USING AUC-ROC or AUC-PRC
(PERFORMANCE PLOT TO COMPARE PERFORMANCE OF CTR PREDICTION MODELS)

Allows comparison of 2 models by two methods:

- plotting AUC-ROC and calculating AUC values
- plotting AUC-PRC and calculating AUC values


(AUC-ROC: Receiving operator curve)
(AUC-PRC: Precision Recall curve)

Authors:
    Cristina Serrano <cserrano@gmx.com>
    (add yourself if you add/modify anything)

Attributions:
    -
    
To try:
y_true = np.round(np.random.rand(500,1),0)
y_pred1 = np.round(np.random.rand(500,1),0)
y_pred2 = np.round(np.random.rand(500,1),0)


Version: 0.1
Creation Date: 16/02/2016
Date Last Modification: 16/02/2016
Revision History:
    Rev 0.1 - File Created, Cristina Serrano, 09/02/2016
'''

import numpy as np
from sklearn.metrics import metrics
import matplotlib.pyplot as plt

def compareAUCroc (y_pred1array, y_pred2array, y_truearray):
    """
    * Compares AUCroc curves for the predictions of 2 different models (plot + AUC value)

    Args:
        y_pred1array (numpy array of int): vector of predicted values (0 or 1) for model 1.
        y_pred2array (numpy array of int): vector of predicted values (0 or 1) for model 2.
        y_truearray (numpy array of int): vector of true values (0 or 1).

    Returns:
        (nothing)        
        the function plots a comparative AOC-ROC curve with value of AUC included
    """

    
    #AUC score
    #a) predicted = 0
    fpr1, tpr1, _ = metrics.roc_curve(y_truearray, y_pred1array)
    roc_auc1 = metrics.auc(fpr1, tpr1)
    
    #b) predicted = 0.06998
    fpr2, tpr2, _ = metrics.roc_curve(y_truearray, y_pred2array)
    roc_auc2 = metrics.auc(fpr2, tpr2)
    
    
    # Plot ROC curve 
    plt.figure()
    plt.plot(fpr1, tpr1, label='ROC curve case A (area = %0.2f)' % roc_auc1)
    plt.plot(fpr1, tpr1, label='ROC curve case B (area = %0.2f)' % roc_auc2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.legend(loc="lower right")
    plt.show()
    return 

def compareAUCprc (y_pred1array, y_pred2array, y_truearray):
    
    """
    Compares AUC-PRC curves for 2 predictions.

    Args:
        y_pred1array (numpy array of floats): vector of predicted values for model 1.
        y_pred2array (numpy array of floats): vector of predicted values for model 2.
        y_truearray (numpy array of floats): vector of true values.

    Returns:
        (nothing)        
        the function plots a comparative AOC-PRC curve with value of AUC included
    """    
    #AUCprc score
    
    #a) predicted = 0
    precision1, recall1, threshold1 = metrics.precision_recall_curve(y_truearray, y_pred1array)
    average_precision1 = metrics.average_precision_score(y_truearray, y_pred1array)
    
    #b) predicted = 0.06998
    precision2, recall2, threshold2 = metrics.precision_recall_curve(y_truearray, y_pred2array)
    average_precision2 = metrics.average_precision_score(y_truearray, y_pred2array)
    
    # Plot curve
    plt.clf()
    plt.plot(recall1, precision1, label='case A: AUC={0:0.2f}'.format(average_precision1))
    plt.plot(recall2, precision2, label='case A: AUC={0:0.2f}'.format(average_precision2))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall')
    plt.legend(loc="lower left")
    plt.show()
    return