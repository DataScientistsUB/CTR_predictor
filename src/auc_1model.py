'''
PERFORMANCE OF 1 MODEL'S PREDICTIONS BY USING AUC-ROC or AUC-PRC

two methods:

- plotting AUC-ROC and calculating AUC values
- plotting AUC-PRC and calculating AUC values


(AUC-ROC: Receiving operator curve)
(AUC-PRC: Precision Recall curve)

Authors:
    Cristina Serrano <cserrano@gmx.com>
    (add yourself if you add/modify anything)

Attributions:
    -
    

Version: 0.2
Creation Date: 28/02/2016
Date Last Modification: 16/02/2016
Revision History:
    Rev 0.1 - File Created, Cristina Serrano, 09/02/2016
    Rev 0.2 - File adapted to output of ctr.py, 28/02/2016
'''

import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd




def AUCroc (filename="y_y_pred.txt"):
    """
    * AUCroc curves for the predictions (plot + AUC value)

    Args:
        filename: name of output file for model

    Returns:
        (nothing)        
        the function plots a comparative AOC-ROC and AUC-PRC curve with value of AUC included
        
    """
    
    #MODEL 
    df = pd.read_csv(filename, sep=' ', names=["y","y_pred"])
    y_predarray = np.array(df.y_pred)
    y_truearray = np.array(df.y)

    #AUC ROC CURVE - 1 MODEL
    
    #AUC score
    fpr, tpr, _ = metrics.roc_curve(y_truearray, y_predarray)
    roc_auc = metrics.auc(fpr, tpr)
    
    # Plot ROC curve 
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve case (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.legend(loc="lower right")
    plt.show()
    return 

def AUCprc (filename="y_y_pred.txt"):
    
    """
    Compares AUC-PRC curve 

    Args:
        filename: name of output file for model
        

    Returns:
        (nothing)        
        the function plots a comparative AOC-PRC curve with value of AUC included
    """    
    #MODEL 
    df = pd.read_csv(filename, sep=' ', names=["y","y_pred"])
    y_predarray = np.array(df.y_pred)
    y_truearray = np.array(df.y)

    #AUCprc
    precision, recall, threshold = metrics.precision_recall_curve(y_truearray, y_predarray)
    average_precision = metrics.average_precision_score(y_truearray, y_predarray)
    
    # Plot curve
    plt.clf()
    plt.plot(recall, precision, label='case A: AUC={0:0.2f}'.format(average_precision))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall')
    plt.legend(loc="lower left")
    plt.show()
    return

AUCroc("y_y_pred.txt")
AUCprc("y_y_pred.txt")