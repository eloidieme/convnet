import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

from cnnClassifier import logger

CLASS_LABELS = ["Arabic", "Chinese", "Czech", "Dutch", "English", "French", "German", "Greek", "Irish",
                "Italian", "Japanese", "Korean", "Polish", "Portuguese", "Russian", "Scottish", "Spanish", "Vietnamese"]

class BalancedBatchSampler:
    def __init__(self, labels, n_samples, n_classes):
        self.labels = labels
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.indices = [torch.where(self.labels == i)[0] for i in range(1, n_classes + 1)]

    def __iter__(self):
        n_batches = len(self.indices[0]) // self.n_samples
        for _ in range(n_batches):
            batch = []
            for class_indices in self.indices:
                chosen_indices = class_indices[torch.multinomial(torch.ones(len(class_indices)), self.n_samples, replacement=False)]
                batch.append(chosen_indices)

            batch = torch.cat(batch)
            batch = batch[torch.randperm(len(batch))]
            yield batch.tolist()

    def __len__(self):
        return len(self.indices[0]) // self.n_samples


class CNN:
    def __init__(self, X_train, Y_train, y_train, network_params, gd_params, metadata, validation=None, balanced = False, seed=None) -> None:
        if seed:
            np.random.seed(seed)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.X_train = torch.tensor(X_train, dtype=torch.float).to(self.device)
        self.Y_train = torch.tensor(Y_train, dtype=torch.float).to(self.device)
        self.y_train = torch.tensor(y_train, dtype=torch.float).to(self.device)
        self.validation = validation
        if validation:
            self.X_val = torch.tensor(validation[0], dtype=torch.float).to(self.device)
            self.Y_val = torch.tensor(validation[1], dtype=torch.float).to(self.device)
            self.y_val = torch.tensor(validation[2], dtype=torch.float).to(self.device)
        self.n1 = network_params['n1']  # number of filters at layer 1
        # width of the filters at layer 1 (final shape: d x k1)
        self.k1 = network_params['k1']
        self.n2 = network_params['n2']  # number of filters at layer 2
        # width of the filters at layer 1 (final shape: d x k2)
        self.k2 = network_params['k2']
        self.eta = network_params['eta']  # learning rate
        self.rho = network_params['rho']  # momentum term
        self.n_batch = gd_params['n_batch']  # no of samples in one batch
        self.n_epochs = gd_params['n_epochs']  # no of epochs to train
        self.d = metadata['dimensionality']  # dimensionality
        self.K = metadata['n_classes']  # number of classes
        self.n_len = metadata['n_len']  # max len of a name
        self.n_len1 = self.n_len - self.k1 + 1
        self.n_len2 = self.n_len1 - self.k2 + 1
        self.f_size = self.n2 * self.n_len2
        self.balanced = balanced

        self._init_params()

    def _init_params(self, p=0.01):
        sigma1 = np.sqrt(2/(p*self.d*self.k1*self.n1))
        sigma2 = np.sqrt(2/(self.n1*self.k2*self.n2))
        sigma3 = np.sqrt(2/self.f_size)

        self.F = []  # filters
        self.F.append(torch.normal(
            0.0, sigma1, size=(self.d, self.k1, self.n1)).to(self.device))
        self.F.append(torch.normal(
            0.0, sigma2, size=(self.n1, self.k2, self.n2)).to(self.device))
        self.W = torch.normal(
            0.0, sigma3, size=(self.K, self.f_size)).to(self.device)  # FC layer weights

    @staticmethod
    def make_mf_matrix(F, n_len):
        dd, k, nf = F.shape
        V_F = []
        for i in range(nf):
            V_F.append(F[:, :, i].t().flatten())
        V_F = torch.tensor(np.array(V_F), dtype=torch.float)
        zero_nlen = torch.zeros((dd, nf))
        kk = n_len - k
        Mf = []
        for i in range(kk+1):
            tup = [zero_nlen.T for _ in range(kk + 1)]
            tup[i] = V_F
            Mf.append(torch.cat(tup, dim=1))
        Mf = torch.cat(Mf, dim=0).to(self.device)
        return Mf

    def make_mx_matrix(self, x_input, d, k, nf, n_len):
        X_input = x_input.reshape((n_len, -1)).t()
        I_nf = torch.eye(nf)
        Mx = []
        for i in range(n_len - k + 1):
            Mx.append(torch.kron(I_nf, X_input[:d, i:i+k].t().flatten()))
        Mx = torch.cat(Mx, dim=0).to(self.device)
        return Mx

    def compute_loss(self, X_batch, Y_batch, F, W):
        MFs = [CNN.make_mf_matrix(F[0], self.n_len),
               CNN.make_mf_matrix(F[1], self.n_len1)]
        _, _, P_batch = CNN.evaluate_classifier(X_batch, MFs, W)
        n_samples = Y_batch.shape[1]
        log_probs = torch.log(P_batch)
        cross_entropy = -torch.sum(Y_batch * log_probs)
        average_loss = cross_entropy / n_samples
        return average_loss

    @staticmethod
    def evaluate_classifier(X_batch, MFs, W):
        s1 = torch.matmul(MFs[0], X_batch)
        X1_batch = F.relu(s1)
        s2 = torch.matmul(MFs[1], X1_batch)
        X2_batch = F.relu(s2)
        S_batch = torch.matmul(W, X2_batch)
        P_batch = F.softmax(S_batch, dim=0)
        return X1_batch, X2_batch, P_batch

    def compute_gradients(self, X_batch, Y_batch, F, W):
        MFs = [CNN.make_mf_matrix(F[0], self.n_len),
               CNN.make_mf_matrix(F[1], self.n_len1)]
        dF1 = torch.zeros(F[0].numel())
        dF2 = torch.zeros(F[1].numel())
        X1_batch, X2_batch, P_batch = CNN.evaluate_classifier(X_batch, MFs, W)
        n = X_batch.shape[1]
        fact = 1/n
        G_batch = -(Y_batch - P_batch)
        dW = fact*(torch.matmul(G_batch, X2_batch.t()))
        G_batch = torch.matmul(W.t(), G_batch)
        G_batch = G_batch * (X2_batch > 0).float()
        for i in range(n):
            dF2 += torch.matmul(
                fact*G_batch[:, i].t(),
                self.make_mx_matrix(X1_batch[:, i], self.n1, self.k2, self.n2, self.n_len1)
            )
        G_batch = MFs[1].T @ G_batch
        G_batch = G_batch * (X1_batch > 0)
        for i in range(n): 
            dF1 += torch.matmul(
                fact*G_batch[:, i].t(),
                self.make_mx_matrix(X_batch[:, i], self.d, self.k1, self.n1, self.n_len)
            )
        temp_shape_1 = (F[0].shape[2], F[0].shape[1], F[0].shape[0])
        temp_shape_2 = (F[1].shape[2], F[1].shape[1], F[1].shape[0])
        return dW, dF1.reshape(temp_shape_1).permute(2,1,0), dF2.reshape(temp_shape_2).permute(2,1,0)

    def compute_accuracy(self, X, y, F, W):
        MFs = [CNN.make_mf_matrix(F[0], self.n_len),
               CNN.make_mf_matrix(F[1], self.n_len1)]
        P = CNN.evaluate_classifier(X, MFs, W)[-1]
        y_pred = torch.argmax(P, dim=0) + 1
        correct = y_pred[y == y_pred].shape[0]
        return correct / y_pred.shape[0]

    def compute_confusion_matrix(self, X, y, F, W):
        MFs = [CNN.make_mf_matrix(F[0], self.n_len),
               CNN.make_mf_matrix(F[1], self.n_len1)]
        P = CNN.evaluate_classifier(X, MFs, W)[-1]
        n_classes, _ = P.shape
        y_pred = torch.argmax(P, dim=0)
        confusion_matrix = torch.zeros((n_classes, n_classes), dtype=torch.int)
        for true_class, pred_class in zip(y, y_pred):
            confusion_matrix[int(true_class) - 1, pred_class] += 1
        return confusion_matrix

    def plot_confusion_matrix(self, conf_matrix, classes, savepath, title='Confusion Matrix', cmap=plt.cm.Blues):
        plt.clf()
        plt.figure(figsize=(15, 15))
        sns.heatmap(conf_matrix.numpy(), annot=True, fmt='d', cmap=cmap,
                    xticklabels=classes, yticklabels=classes)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(savepath)

    def mini_batch_gd(self):
        n_classes = 18
        n_samples_per_class = int(min(torch.bincount(self.y_train.int() - 1)))

        dataset = TensorDataset(self.X_train.t(), self.Y_train.t())
   
        train_losses = [self.compute_loss(
            self.X_train, self.Y_train, self.F, self.W)]
        val_losses = [self.compute_loss(
            self.X_val, self.Y_val, self.F, self.W)] if self.validation else None
        val_accs = [self.compute_accuracy(
            self.X_val, self.y_val, self.F, self.W)] if self.validation else None
        
        mw = torch.zeros_like(self.W)
        m1 = torch.zeros_like(self.F[0])
        m2 = torch.zeros_like(self.F[1])
        
        if self.balanced:
            balanced_batch_sampler = BalancedBatchSampler(self.y_train, n_samples_per_class, n_classes)
            dataloader = DataLoader(dataset, batch_sampler=balanced_batch_sampler)
        else:
            dataset = TensorDataset(self.X_train.t(), self.Y_train.t())
            dataloader = DataLoader(dataset, batch_size=self.n_batch, shuffle=True)

        for i in range(self.n_epochs):
            print(f"Epoch {i+1}/{self.n_epochs}")
            for X_batch, Y_batch in tqdm.tqdm(dataloader, desc="Processing batches"):
                dW, dF1, dF2 = self.compute_gradients(
                    X_batch.t(), Y_batch.t(), self.F, self.W)
                mw = self.rho*mw + (1 - self.rho)*dW
                m1 = self.rho*m1 + (1 - self.rho)*dF1
                m2 = self.rho*m2 + (1 - self.rho)*dF2
                self.W -= self.eta*mw
                self.F[0] -= self.eta*m1
                self.F[1] -= self.eta*m2
            current_train_loss = self.compute_loss(
                self.X_train, self.Y_train, self.F, self.W)
            print(f"\t * Train loss: {current_train_loss}")
            train_losses.append(current_train_loss)
            if self.validation:
                current_val_loss = self.compute_loss(
                    self.X_val, self.Y_val, self.F, self.W)
                print(f"\t * Validation loss: {current_val_loss}")
                val_losses.append(current_val_loss)
                current_val_acc = self.compute_accuracy(
                    self.X_val, self.y_val, self.F, self.W)
                val_accs.append(current_val_acc)
        conf_mat = self.compute_confusion_matrix(
            self.X_val, self.y_val, self.F, self.W)
        self.plot_confusion_matrix(conf_mat, CLASS_LABELS, f"./reports/figures/conf_mat")
        return train_losses, val_losses, val_accs

    def run_training(self, figure_savepath=None, test_data=None, model_savepath=None):
        train_losses, val_losses, val_accs = self.mini_batch_gd()
        logger.info("Training completed.")

        if model_savepath:
            os.makedirs(f"{model_savepath}", exist_ok=True)
            torch.save(self.F[0], f"{model_savepath}/F1")
            torch.save(self.F[1], f"{model_savepath}/F2")
            torch.save(self.W, f"{model_savepath}/W")

        if test_data:
            (X_test, y_test) = test_data
            X_test = torch.tensor(X_test, dtype=torch.float).to(self.device)
            y_test = torch.tensor(y_test, dtype=torch.float).to(self.device)
            accuracy = self.compute_accuracy(X_test, y_test, self.F, self.W)
            print(f"Accuracy on test data: {accuracy}")
            logger.info("Accuracy on test data: %.3f", accuracy)

        plt.clf()
        plt.plot(np.arange(self.n_epochs + 1),
                 train_losses, label="training loss")
        plt.plot(np.arange(self.n_epochs + 1),
                 val_losses, label="validation loss")
        plt.xlabel(f"epochs")
        plt.ylabel("cross-entropy loss")
        plt.legend()
        plt.title(
            f"Training curves (n_batch = {self.n_batch}, n_epochs = {self.n_epochs}, eta = {self.eta})")
        plt.grid()
        if figure_savepath:
            plt.savefig(figure_savepath + "_loss", bbox_inches='tight')
        plt.clf()
        plt.plot(np.arange(self.n_epochs + 1),
                 val_accs, label="validation accuracies")
        plt.xlabel(f"epochs")
        plt.ylabel("accuracy")
        plt.legend()
        plt.title(
            f"Accuracy (n_batch = {self.n_batch}, n_epochs = {self.n_epochs}, eta = {self.eta})")
        plt.grid()
        if figure_savepath:
            plt.savefig(figure_savepath + "_accuracy", bbox_inches='tight')

        logger.info("Figure saved.")
