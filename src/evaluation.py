# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 00:54:53 2016



LOGLOSS FUNCTION IMPLEMENTATION 
(PERFORMANCE MODEL SCORE CALCULATION FOR CTR PREDICTION)

The weighted versiont takes into account the cost of a False Positive (w0) and 
False Negative (w1).

Function
    computeTotalLogLoss(pred, true, epsylon)

Parameters:
    pred - predicted value
    true - true outcome/value
    epsylon - value used to avoid pure zeros/ones. By default, epsilon = 10e-12
    
Function weighted
    computeTotalLogLossWeighted(pred, true, weight0, weight1, epsylon)

Parameters:
    pred - predicted value
    true - true outcome/value
    weight0 - cost of a False Positive (Predicting 1 when real value is 0)
    weight1 - cost of a False Negative (Predicting 0 when real value is 1)
    epsylon - value used to avoid pure zeros/ones. By default, epsilon = 10e-12
    """


import scipy as sp

def computeLogLoss(p, y, epsylon=10e-12):
    """
    Source: "Scalable Machine Learning", EdX
    Calculates the value of log loss for a given probabilty and label.

    Note:
        log(0) is undefined, so when p is 0 we need to add a small value (epsilon) to it
        and when p is 1 we need to subtract a small value (epsilon) from it.

    Args:
        p (float): A probabilty between 0 and 1.
        y (int): A label.  Takes on the values 0 and 1.

    Returns:
        float: The log loss value.
    """

    if p == 0.0: p = epsilon
    if p == 1.0: p = 1.0-epsilon
    
    if y == 1: loss = -sp.log(p)
    if y == 0: loss = -sp.log(1.0-p)
    
    return loss

def computeTotalLogLoss(pred,true, epsilon=10e-12):

    loss, total = 0.0, 0.0
    for i in range(0,len(pred)):
        loss += computeLogLoss(pred[i],true[i], epsilon)
        total += 1.0

    if total == 0:
        return('nan')
    else:
        return(round(loss/total, 10))
        
def computeLogLossWeighted(p, y, weight0 = 1.0, weight1 = 1.0, epsylon=10e-12):
    """
    Source: "Scalable Machine Learning", EdX
    Calculates the value of log loss for a given probabilty and label.

    Note:
        log(0) is undefined, so when p is 0 we need to add a small value (epsilon) to it
        and when p is 1 we need to subtract a small value (epsilon) from it.

    Args:
        p (float): A probabilty between 0 and 1.
        y (int): A label.  Takes on the values 0 and 1.
        weight0 (float): cost of predicting 1 when real value is 0
        weight1 (float): cost of predicting 0 when real value is 1

    Returns:
        float: The log loss value.
    """

    if p == 0.0: p = epsilon
    if p == 1.0: p = 1.0-epsilon
    
    if y == 1: loss = - weight1 * sp.log(p)
    if y == 0: loss = - weight0 * sp.log(1.0-p)
    
    return loss

def computeTotalLogLossWeighted(pred,true, weight0 = 1.0, weight1 = 1.0, epsilon=10e-12):

    loss, total = 0.0, 0.0
    for i in range(0,len(pred)):
        loss += computeLogLossWeighted(pred[i],true[i], weight0, weight1, epsilon)
        total += 1.0

    if total == 0:
        return('nan')
    else:
        return(round(loss/total, 10))