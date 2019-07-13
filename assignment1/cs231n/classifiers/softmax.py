from builtins import range
import numpy as np
import math
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]
    num_train = X.shape[0]
    weighted_sum = np.zeros([num_train,num_classes])
    exp_sum = np.zeros([num_train,num_classes])
    reg_sum = np.zeros([num_train,num_classes])

    for i in range(num_train):
        weighted_sum[i] = np.dot(X[i], W)
        exp_sum[i] = np.exp(weighted_sum[i])
        reg_sum[i] = exp_sum[i] / np.sum(exp_sum[i])

        loss += -1 * math.log(reg_sum[i][y[i]])

    loss /= num_train

    for i in range(num_train):
        for j in range(num_classes):
            dW[:, j] += reg_sum[i][j] * X[i]
            if j == y[i]:
                dW[:, j] -= X[i]

    dW /= num_train

    loss += reg* np.sum(W*W)
    dW += 2*reg*W


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]
    num_train = X.shape[0]

    weighted_sum = np.dot(X,W)
    exp_sum = np.exp(weighted_sum)
    reg_sum = exp_sum / np.sum(exp_sum, axis =1, keepdims = True)

    correct_class_score = reg_sum[np.arange(num_train), y]
    correct_class_loss = -1 * np.log(correct_class_score)
    loss += np.sum(correct_class_loss)
    loss /= num_train

    reg_sum[np.arange(num_train), y] -= 1
    dW = np.dot(X.T, reg_sum)
    dW /= num_train

    loss += reg* np.sum(W*W)
    dW += 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
