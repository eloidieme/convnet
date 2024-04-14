import numpy as np
import matplotlib.pyplot as plt
import tqdm
import json
import os
from typing import List

from cnnClassifier import logger


def softmax(x):
    """Compute softmax values for each sets of scores in x along columns."""
    e_x = np.exp(x - np.max(x, axis=0))  # subtract max for numerical stability
    return e_x / e_x.sum(axis=0)


class CNN:
    def __init__(self, X_train, Y_train, network_params, metadata, validation=None, seed=None) -> None:
        if seed:
            np.random.seed(seed)
        self.X_train = X_train
        self.Y_train = Y_train
        self.validation = validation
        if validation:
            self.X_val = validation[0]
            self.Y_val = validation[1]
            self.y_val = validation[2]
        self.n1 = network_params['n1']  # number of filters at layer 1
        # width of the filters at layer 1 (final shape: d x k1)
        self.k1 = network_params['k1']
        self.n2 = network_params['n2']  # number of filters at layer 2
        # width of the filters at layer 1 (final shape: d x k2)
        self.k2 = network_params['k2']
        self.eta = network_params['eta']  # learning rate
        self.rho = network_params['rho']  # momentum term
        self.d = metadata['dimensionality']  # dimensionality
        self.K = metadata['n_classes']  # number of classes
        self.n_len = metadata['n_len']  # max len of a name
        self.n_len1 = self.n_len - self.k1 + 1
        self.n_len2 = self.n_len1 - self.k2 + 1
        self.f_size = self.n2 * self.n_len2

        self._init_params()

    def _init_params(self, p=0.01):
        sigma1 = np.sqrt(2/(p*self.d*self.k1*self.n1))
        sigma2 = np.sqrt(2/(self.n1*self.k2*self.n2))
        sigma3 = np.sqrt(2/self.f_size)

        self.F = []  # filters
        self.F.append(np.random.normal(
            0.0, sigma1, (self.d, self.k1, self.n1)))
        self.F.append(np.random.normal(
            0.0, sigma2, (self.n1, self.k2, self.n2)))
        self.W = np.random.normal(
            0.0, sigma3, (self.K, self.f_size))  # FC layer weights

    def make_mf_matrix(self, F, n_len):
        dd, k, nf = F.shape
        V_F = []
        for i in range(nf):
            V_F.append(F[:, :, i].flatten(order='F'))
        V_F = np.array(V_F)
        zero_nlen = np.zeros((dd, nf))
        kk = n_len - k
        Mf = []
        for i in range(kk+1):
            tup = [zero_nlen.T for _ in range(kk + 1)]
            tup[i] = V_F
            Mf.append(np.concatenate(tup, axis=1))
        Mf = np.concatenate(Mf, axis=0)
        return Mf

    def make_mx_matrix(self, x_input, d, k, nf, n_len):
        X_input = x_input.reshape((-1, n_len), order='F')
        I_nf = np.eye(nf)
        Mx = []
        for i in range(n_len - k + 1):
            Mx.append(np.kron(I_nf, X_input[:d, i:i+k].ravel(order='F').T))
        Mx = np.concatenate(Mx, axis=0)
        return Mx

    def compute_loss(self, X_batch, Y_batch, F, W):
        MFs = [self.make_mf_matrix(F[0], self.n_len),
               self.make_mf_matrix(F[1], self.n_len1)]
        _, _, P_batch = self.evaluate_classifier(X_batch, MFs, W)
        n_samples = Y_batch.shape[1]
        log_probs = np.log(P_batch)
        cross_entropy = -np.sum(Y_batch * log_probs)
        average_loss = cross_entropy / n_samples
        return average_loss

    def evaluate_classifier(self, X_batch, MFs, W):
        s1 = MFs[0] @ X_batch
        X1_batch = np.maximum(0, s1)
        s2 = MFs[1] @ X1_batch
        X2_batch = np.maximum(0, s2)
        S_batch = W @ X2_batch
        P_batch = softmax(S_batch)
        return X1_batch, X2_batch, P_batch

    def compute_gradients(self, X_batch, Y_batch, F, W):
        MFs = [self.make_mf_matrix(F[0], self.n_len),
               self.make_mf_matrix(F[1], self.n_len1)]
        dF1 = np.zeros(F[0].size)
        dF2 = np.zeros(F[1].size)
        X1_batch, X2_batch, P_batch = self.evaluate_classifier(X_batch, MFs, W)
        n = X_batch.shape[1]
        fact = 1/n
        G_batch = -(Y_batch - P_batch)
        dW = fact*(G_batch @ X2_batch.T)
        G_batch = W.T @ G_batch
        G_batch = G_batch * (X2_batch > 0)
        for i in range(n):
            gi = G_batch[:, i]
            xi = X1_batch[:, i]
            v = gi.T @ self.make_mx_matrix(xi, self.n1,
                                           self.k2, self.n2, self.n_len1)
            dF2 += fact*v
        G_batch = MFs[1].T @ G_batch
        G_batch = G_batch * (X1_batch > 0)
        for i in range(n):
            gi = G_batch[:, i]
            xi = X_batch[:, i]
            v = gi.T @ self.make_mx_matrix(xi,
                                           self.d, self.k1, self.n1, self.n_len)
            dF1 += fact*v
        return dW, dF1.reshape(F[0].shape, order='F'), dF2.reshape(F[1].shape, order='F')

    def compute_accuracy(self, X, y, F, W):
        MFs = [self.make_mf_matrix(F[0], self.n_len),
               self.make_mf_matrix(F[1], self.n_len1)]
        P = self._evaluate_classifier(X, MFs, W)[-1]
        y_pred = np.argmax(P, axis=0)
        correct = y_pred[y == y_pred].shape[0]
        return correct / y_pred.shape[0]
