import numpy as np
import matplotlib.pyplot as plt
import tqdm
import json
from copy import deepcopy
import os
from typing import List

from cnnClassifier import logger

def softmax(x):
    x = x - np.max(x, axis=0)  # for numerical stability
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=0)


class Model:
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
        self.k1 = network_params['k1']  # width of the filters at layer 1 (final shape: d x k1)
        self.n2 = network_params['n2']  # number of filters at layer 2
        self.k2 = network_params['k2']  # width of the filters at layer 1 (final shape: d x k2)
        self.eta = network_params['eta']  # learning rate
        self.rho = network_params['rho']  # momentum term
        self.d = metadata['dimensionality']  # dimensionality
        self.K = metadata['n_classes']  # number of classes
        self.n_len = metadata['n_len']  # max len of a name
        self.n_len1 = self.n_len - self.k1 + 1
        self.n_len2 = self.n_len1 - self.k2 + 1
        self.f_size = self.n2 * self.n_len2

        self._init_params()


    def _init_params(self, p = 0.01):
        sigma1 = np.sqrt(2/(p*self.d*self.k1*self.n1))
        sigma2 = np.sqrt(2/(self.n1*self.k2*self.n2))
        sigma3 = np.sqrt(2/self.f_size)

        self.F = []  # filters
        self.F.append(np.random.normal(0.0, sigma1, (self.d, self.k1, self.n1)))
        self.F.append(np.random.normal(0.0, sigma2, (self.n1, self.k2, self.n2)))
        self.W = np.random.normal(0.0, sigma3, (self.K, self.f_size))  # FC layer weights

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
        MFs = [self.make_mf_matrix(F[0], self.n_len), self.make_mf_matrix(F[1], self.n_len1)]
        _, _, P_batch = self.evaluate_classifier(X_batch, MFs, W)
        fact = 1/X_batch.shape[1]
        lcross_sum = np.sum(np.diag(-Y_batch.T@np.log(P_batch)))
        return fact*lcross_sum
    
    def evaluate_classifier(self, X_batch, MFs, W):
        s1 = MFs[0] @ X_batch
        X1_batch = np.maximum(0, s1)
        s2 = MFs[1] @ X1_batch
        X2_batch = np.maximum(0, s2)
        S_batch = W @ X2_batch
        P_batch = softmax(S_batch)
        return X1_batch, X2_batch, P_batch

    def compute_gradients(self, X_batch, Y_batch, F, W):
        MFs = [self.make_mf_matrix(F[0], self.n_len), self.make_mf_matrix(F[1], self.n_len1)]
        dF1 = np.zeros(np.prod(F[0].shape))
        dF2 = np.zeros(np.prod(F[1].shape))
        X1_batch, X2_batch, P_batch = self.evaluate_classifier(X_batch, MFs, W)
        n = X_batch.shape[1]
        fact = 1/n
        G_batch = -(Y_batch - P_batch)
        dW = fact*(G_batch@X2_batch.T)
        G_batch = W.T @ G_batch
        G_batch = G_batch * (X2_batch > 0)
        for i in range(n):
            gi = G_batch[:, i]
            xi = X1_batch[:, i]
            v = gi.T @ self.make_mx_matrix(xi, self.n1, self.k2, self.n2, self.n_len1)
            dF2 += fact*v
        G_batch = MFs[1].T @ G_batch
        G_batch = G_batch * (X1_batch > 0)
        for i in range(n):
            gi = G_batch[:, i]
            xi = X_batch[:, i]
            v = gi.T @ self.make_mx_matrix(xi, self.d, self.k1, self.n1, self.n_len)
            dF1 += fact*v
        return dW, dF1.reshape(self.d, self.k1, self.n1), dF2.reshape(self.n1, self.k2, self.n2)
    
    def numerical_gradients(self, X_inputs, Ys, F, W, h):
        dFs = [np.zeros_like(F[0]), np.zeros_like(F[1])]
        ns = [F[0].shape[2], F[1].shape[2]]

        try_F = deepcopy(F)
        for l in range(len(try_F)):
            for i in range(ns[l]):
                dF = np.zeros(np.prod(F[l][:, :, i].shape))
                for j in range(len(dF)):
                    try1_F = deepcopy(try_F)
                    try1_F[l][:, :, i].flat[j] -= h
                    l1 = self.compute_loss(X_inputs, Ys, try1_F, W)
                    try2_F = deepcopy(try_F)
                    try2_F[l][:, :, i].flat[j] += h
                    l2 = self.compute_loss(X_inputs, Ys, try2_F, W)
                    dF[j] = (l2 - l1) / (2 * h)
                dFs[l][:, :, i] = dF.reshape(F[l][:, :, i].shape)

        try_W = np.copy(W)
        dW = np.zeros(np.prod(try_W.shape))
        for j in range(len(dW)):
            W_try1 = np.array(try_W, copy=True)
            W_try1.flat[j] = try_W.flat[j] - h
            l1 = self.compute_loss(X_inputs, Ys, F, W_try1)
            W_try2 = np.array(try_W, copy=True)
            W_try2.flat[j] = try_W.flat[j] + h
            l2 = self.compute_loss(X_inputs, Ys, F, W_try2)
            dW[j] = (l2 - l1) / (2 * h)
        dW = dW.reshape(try_W.shape)

        return dW, dFs[0], dFs[1]

    def validate_gradient(self, X_batch, Y_batch, F, W, h=1e-6, eps=1e-10):
        _, dF1, dF2 = self.compute_gradients(X_batch, Y_batch, F, W)
        aGrads = [dF1, dF2]
        _, dF1n, dF2n = self.numerical_gradients(X_batch, Y_batch, F, W, h)
        nGrads = [dF1n, dF2n]
        rel_errs = []
        for k in range(2):
            rel_err = np.zeros_like(aGrads[k])
            for i in range(rel_err.shape[0]):
                for j in range(rel_err.shape[1]):
                    for l in range(rel_err.shape[2]):
                        rel_err[i, j, l] = (np.abs(aGrads[k][i, j, l] - nGrads[k][i, j, l])) / \
                            (max(eps, np.abs(aGrads[k][i, j, l]) +
                                np.abs(nGrads[k][i, j, l])))
            rel_errs.append(rel_err)
        max_diff = [np.max(rel_err) for rel_err in rel_errs]
        return max_diff

    def compute_accuracy(self, X, y, F, W):
        MFs = [self.make_mf_matrix(F[0], self.n_len), self.make_mf_matrix(F[1], self.n_len1)]
        P = self._evaluate_classifier(X, MFs, W)[-1]
        y_pred = np.argmax(P, axis=0)
        correct = y_pred[y == y_pred].shape[0]
        return correct / y_pred.shape[0]
