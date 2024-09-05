import torch
import torch.nn as nn
from activation import Softmax
import numpy as np

# Implementing Cross-Entropy Loss Function
class CrossEntropyLoss(nn.Module):

    # Beginning Definition
    def __init__(self, y_pred, y_true):
        super(CrossEntropyLoss, self).__init__()

        # Utilizing Softmax for Predicted Values
        y_pred = Softmax(y_pred)
        loss = 0

        # Calculating Cross-Entropy Loss Values
        for x in range(len(y_pred)):
            loss = loss + (-1 * y_true[x]*np.log(y_pred[x]))

        # Outputting Calculation
        return loss