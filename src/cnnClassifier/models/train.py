import numpy as np
import matplotlib.pyplot as plt
import tqdm
import json
import copy
import os
from abc import ABC, abstractmethod
from typing import List

from cnnClassifier import logger

N_CLASSES = 10
N_FEATURES = 3072
N_HIDDEN_NODES = 50


def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def sigmoid(x):
    """ Standard definition of the sigmoid function """
    return np.exp(x) / (np.exp(x) + 1)


class Model(ABC):
    def __init__(self, X_train, Y_train, gd_params, lamda=0.0, validation=None, cyclical_lr = False, step_decay=False, mbce=False, seed=None) -> None:
        if seed:
            np.random.seed(seed)
        self.X_train = X_train
        self.Y_train = Y_train
        self.gd_params = gd_params
        self.lamda = lamda
        self.validation = validation
        if validation:
            self.X_val = validation[0]
            self.Y_val = validation[1]
            self.y_val = validation[2]
        self.step_decay = step_decay
        self.mbce = mbce
        self.cyclical_lr = cyclical_lr

    @abstractmethod
    def _init_params(self, means: List[float], stds: List[float]):
        pass

    @abstractmethod
    def _evaluate_classifier(self, X: np.ndarray, Ws: List[np.ndarray], bs: List[np.ndarray]):
        pass

    @abstractmethod
    def _compute_gradients(self, X_batch: np.ndarray, Y_batch: np.ndarray, Ws: List[np.ndarray], bs: List[np.ndarray]):
        pass

    def get_current_learning_rate(self, t, eta_min=1e-5, eta_max=1e-1, n_s=800):
        #n_s = 2*np.floor(self.X_train.shape[1]/self.gd_params['n_batch'])
        cycle = np.floor(1 + t / (2 * n_s))
        x = np.abs(t / n_s - 2 * cycle + 1)
        eta_t = eta_min + (eta_max - eta_min) * np.maximum(0, (1 - x))

        return eta_t

    def compute_grads_num_slow(self, X, Y, Ws, bs, h=1e-6):
        grads_W = [np.zeros(W.shape) for W in Ws]
        grads_b = [np.zeros(b.shape) for b in bs]

        for layer in range(len(Ws)):
            for i in range(len(bs[layer])):
                b_try = np.array(bs[layer])
                b_try[i] -= h
                bs_try = bs[:]
                bs_try[layer] = b_try
                c1 = self._compute_cost(X, Y, Ws, bs_try)

                b_try[i] += 2*h
                bs_try[layer] = b_try
                c2 = self._compute_cost(X, Y, Ws, bs_try)

                grads_b[layer][i] = (c2 - c1) / (2 * h)

            for i in range(Ws[layer].shape[0]):
                for j in range(Ws[layer].shape[1]):
                    W_try = np.array(Ws[layer])
                    W_try[i, j] -= h
                    Ws_try = Ws[:]
                    Ws_try[layer] = W_try
                    c1 = self._compute_cost(X, Y, Ws_try, bs)

                    W_try[i, j] += 2*h
                    Ws_try[layer] = W_try
                    c2 = self._compute_cost(X, Y, Ws_try, bs)

                    grads_W[layer][i, j] = (c2 - c1) / (2 * h)

        return grads_W, grads_b

    def _compute_cost(self, X: np.ndarray, Y: np.ndarray, Ws: List[np.ndarray], bs: List[np.ndarray]):
        P = self._evaluate_classifier(X, Ws, bs)[-1]
        fact = 1/X.shape[1]
        if self.mbce:
            lcross_sum = np.sum(
                (-1/N_CLASSES)*np.diag((np.ones_like(Y) - Y).T @ np.log(1 - P) + Y.T @ np.log(P)))
        else:
            lcross_sum = np.sum(np.diag(-Y.T@np.log(P)))
        reg_sum = 0
        for W in Ws:
            reg_sum += np.sum(np.ravel(W**2))
        return fact*lcross_sum + self.lamda*reg_sum

    def mini_batch_gd(self, gd_params, grid_search=False):
        Ws_train, bs_train = copy.deepcopy(self.Ws), copy.deepcopy(self.bs)
        train_costs = [self._compute_cost(
            self.X_train, self.Y_train, Ws_train, bs_train)]
        val_costs = [self._compute_cost(
            self.X_val, self.Y_val, Ws_train, bs_train)] if self.validation else None
        n_batch, n_epochs, eta = gd_params["n_batch"], gd_params["n_epochs"], gd_params["eta"]
        if self.cyclical_lr:
            t = 0
            eta = self.get_current_learning_rate(t)
        n = self.X_train.shape[1]
        for i in range(n_epochs):
            print(f"Epoch {i+1}/{n_epochs}")
            for j in tqdm.tqdm(range(1, int(n/n_batch) + 1)):
                start = (j-1)*n_batch
                end = j*n_batch
                perm = np.random.permutation(n)
                X_batch = self.X_train[:, perm][:, start:end]
                Y_batch = self.Y_train[:, perm][:, start:end]
                grads = self._compute_gradients(
                    X_batch, Y_batch, Ws_train, bs_train)
                for idx, W_grad in enumerate(grads[0]):
                    Ws_train[idx] -= eta*W_grad
                for idx, b_grad in enumerate(grads[1]):
                    bs_train[idx] -= eta*b_grad
                if self.cyclical_lr:
                    t += 1
                    eta = self.get_current_learning_rate(t)
            print(f"\t * Learning rate: {eta}")
            current_train_loss = self._compute_cost(
                self.X_train, self.Y_train, Ws_train, bs_train)
            print(f"\t * Train loss: {current_train_loss}")
            train_costs.append(current_train_loss)
            if self.validation:
                current_val_loss = self._compute_cost(
                    self.X_val, self.Y_val, Ws_train, bs_train)
                print(f"\t * Validation loss: {current_val_loss}")
                val_costs.append(current_val_loss)
                if self.step_decay and (i+1) % 10 == 0:
                    eta /= 10
        if not grid_search:
            self.Ws_trained = Ws_train
            self.bs_trained = bs_train
        return Ws_train, bs_train, train_costs, val_costs

    def validate_gradient(self, X, Y, h=1e-6, eps=1e-10):
        n_features = X.shape[0]
        reduced_Ws = []
        for W in self.Ws:
            reduced_Ws.append(W[:, :n_features])
        ga_W = self._compute_gradients(X, Y, reduced_Ws, self.bs)[0]
        gn_W = self.compute_grads_num_slow(X, Y, reduced_Ws, self.bs, h)[0]
        rel_err_W = np.zeros((len(ga_W), ga_W[0].shape[0], ga_W[0].shape[1]))
        for k in range(2):
            for i in range(ga_W[k].shape[0]):
                for j in range(ga_W[k].shape[1]):
                    rel_err_W[k, i, j] = (np.abs(ga_W[k][i, j] - gn_W[k][i, j])) / \
                        (max(eps, np.abs(ga_W[k][i, j]) +
                         np.abs(gn_W[k][i, j])))
        ga_b = self._compute_gradients(X, Y, reduced_Ws, self.bs)[1]
        gn_b = self.compute_grads_num_slow(X, Y, reduced_Ws, self.bs, h)[1]
        rel_err_b = np.zeros((len(ga_b), ga_b[0].shape[0]))
        for k in range(2):
            for i in range(ga_b[k].shape[0]):
                rel_err_b[k, i] = (np.abs(ga_b[k][i] - gn_b[k][i])) / \
                    (max(eps, np.abs(ga_b[k][i]) + np.abs(gn_b[k][i])))
        max_diff = max(np.max(rel_err_W), np.max(rel_err_b))
        logger.info("Maximum difference between numerical and analytical gradients: %.2e", max_diff)
        return max_diff

    def compute_accuracy(self, X, y, Ws, bs):
        P = self._evaluate_classifier(X, Ws, bs)[-1]
        y_pred = np.argmax(P, axis=0)
        correct = y_pred[y == y_pred].shape[0]
        return correct / y_pred.shape[0]

    def _grid_search(self, grid, n_epochs=20):
        '''
        grid = {
            "n_batch": [...],
            "eta": [...],
            "lamda": [...]
        }
        '''
        train_losses = {}
        val_losses = {}
        accuracies = {}
        for n_batch in grid["n_batch"]:
            for eta in grid["eta"]:
                for lamda in grid["lamda"]:
                    gd_params = {"n_batch": n_batch,
                                 "n_epochs": n_epochs, "eta": eta}
                    Ws_train, bs_train, train_costs, val_costs = self.mini_batch_gd(
                        gd_params)
                    train_losses[f"{n_batch}_{eta}_{lamda}"] = train_costs
                    val_losses[f"{n_batch}_{eta}_{lamda}"] = val_costs
                    accuracies[f"{n_batch}_{eta}_{lamda}"] = self.compute_accuracy(
                        self.X_val, self.y_val, Ws_train, bs_train)
        return max(accuracies, key=accuracies.get), [train_losses, val_losses, accuracies]

    def run_grid_search(self, grid, n_epochs, output_path):
        grid_res = self._grid_search(grid, n_epochs)
        with open(output_path, 'w+') as f:
            json.dump(grid_res[1], f)
        print(f"Best accuracy: {grid_res[1][-1][grid_res[0]]}")

    def run_training(self, gd_params, figure_savepath = None, test_data = None, model_savepath = None):
        Ws_train, bs_train, train_costs, val_costs = self.mini_batch_gd(gd_params)
        logger.info("Training completed.")

        if model_savepath:
            os.makedirs(f"{model_savepath}/Ws", exist_ok=True)
            os.makedirs(f"{model_savepath}/bs", exist_ok=True)
            for idx, W in enumerate(Ws_train):
                np.save(f"{model_savepath}/Ws/W_{idx}", W)
            for idx, b in enumerate(bs_train):
                np.save(f"{model_savepath}/bs/b_{idx}", b)

        if test_data:
            (X_test, y_test) = test_data
            accuracy = self.compute_accuracy(X_test, y_test, Ws_train, bs_train)
            print(f"Accuracy on test data: {accuracy}")
            logger.info("Accuracy on test data: %.3f", accuracy)

        plt.plot(np.arange(0, gd_params["n_epochs"] + 1),
                 train_costs, label="training loss")
        plt.plot(np.arange(0, gd_params["n_epochs"] + 1),
                 val_costs, label="validation loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        if self.cyclical_lr:
            eta = "Cyclical"
        else:
            eta = gd_params["eta"]
        plt.title(
            f"Training curves (n_batch = {gd_params['n_batch']}, n_epochs = {gd_params['n_epochs']}, eta = {eta}, lambda = {self.lamda})")
        plt.grid()
        plt.xlim(0, gd_params["n_epochs"])
        if figure_savepath:
            plt.savefig(figure_savepath, bbox_inches='tight')
        plt.clf()
        logger.info("Figure saved.")


class OneLayerClassifier(Model):
    def __init__(self, X_train, Y_train, gd_params, lamda=0, validation=None, cyclical_lr = False, step_decay=False, mbce=False, seed=None) -> None:
        super().__init__(X_train, Y_train, gd_params,
                         lamda, validation, cyclical_lr, step_decay, mbce, seed)
        self.Ws, self.bs = self._init_params()

    def _init_params(self, means=[0.0], stds=[0.01]):
        W = np.random.normal(means[0], stds[0], (N_CLASSES, N_FEATURES))
        b = np.random.normal(means[0], stds[0], (N_CLASSES, 1))
        return [W], [b]

    def _evaluate_classifier(self, X, Ws, bs):
        s = Ws[0]@X + bs[0]
        if self.mbce:
            P = sigmoid(s)
        else:
            P = softmax(s)
        return [P]

    def _compute_gradients(self, X_batch, Y_batch, Ws, bs):
        if self.mbce:
            P = self._evaluate_classifier(X_batch, Ws, bs)
            grad_loss_P = (-Y_batch.T / P.T + (1 - Y_batch.T) /
                           (1 - P.T)) / Y_batch.shape[0]
            grad_P_S = P.T * (1 - P.T)
            grad_loss_S = grad_loss_P * grad_P_S

            grad_W = np.dot(X_batch, grad_loss_S).T
            grad_b = np.reshape(np.sum(grad_loss_S, axis=0), (-1, 1))
        else:
            nb = X_batch.shape[1]
            fact = 1/nb
            P = self._evaluate_classifier(X_batch, Ws, bs)[0]
            G = -(Y_batch - P)

            grad_W = fact*(G@X_batch.T)
            grad_b = fact*(G@np.ones((nb, 1)))

        grad_W += 2*self.lamda*Ws[0]

        return [grad_W], [grad_b]


class TwoLayerClassifier(Model):
    def __init__(self, X_train, Y_train, gd_params, lamda=0, validation=None, cyclical_lr = False, step_decay=False, mbce=False, seed=None) -> None:
        super().__init__(X_train, Y_train, gd_params,
                         lamda, validation, cyclical_lr, step_decay, mbce, seed)
        self.Ws, self.bs = self._init_params()

    def _init_params(self, means=[0.0, 0.0], stds=[1/np.sqrt(N_FEATURES), 1/np.sqrt(N_HIDDEN_NODES)]):
        W1 = np.random.normal(means[0], stds[0], (N_HIDDEN_NODES, N_FEATURES))
        b1 = np.zeros((N_HIDDEN_NODES, 1))
        W2 = np.random.normal(means[1], stds[1], (N_CLASSES, N_HIDDEN_NODES))
        b2 = np.zeros((N_CLASSES, 1))
        return [W1, W2], [b1, b2]

    def _evaluate_classifier(self, X, Ws, bs):
        s1 = Ws[0]@X + bs[0]
        H = np.maximum(np.zeros_like(s1), s1)
        s = Ws[1]@H + bs[1]
        if self.mbce:
            P = sigmoid(s)
        else:
            P = softmax(s)
        return [H, P]

    def _compute_gradients(self, X_batch, Y_batch, Ws, bs):
        nb = X_batch.shape[1]
        fact = 1/nb
        H, P = self._evaluate_classifier(X_batch, Ws, bs)
        G = -(Y_batch - P)

        grad_W2 = fact*(G@H.T)
        grad_b2 = fact*(G@np.ones((nb, 1)))

        G = Ws[1].T@G
        G = G * (H > 0)

        grad_W1 = fact*(G@X_batch.T)
        grad_b1 = fact*(G@np.ones((nb, 1)))

        grad_W1 += 2*self.lamda*Ws[0]
        grad_W2 += 2*self.lamda*Ws[1]

        return [grad_W1, grad_W2], [grad_b1, grad_b2]
