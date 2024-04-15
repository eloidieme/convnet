import cupy as cp
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
import os

from cnnClassifier import logger

import cupy.cuda.profiler

CLASS_LABELS = ["Arabic", "Chinese", "Czech", "Dutch", "English", "French", "German", "Greek", "Irish",
                "Italian", "Japanese", "Korean", "Polish", "Portuguese", "Russian", "Scottish", "Spanish", "Vietnamese"]


def softmax(x):
    """Compute softmax values for each sets of scores in x along columns."""
    e_x = cp.exp(x - cp.max(x, axis=0))  # subtract max for numerical stability
    return e_x / e_x.sum(axis=0)


class CNN_GPU:
    def __init__(self, X_train, Y_train, network_params, gd_params, metadata, validation=None, seed=None) -> None:
        if seed:
            cp.random.seed(seed)
        self.X_train = cp.asarray(X_train)
        self.Y_train = cp.asarray(Y_train)
        self.validation = validation
        if validation:
            self.X_val = cp.asarray(validation[0])
            self.Y_val = cp.asarray(validation[1])
            self.y_val = cp.asarray(validation[2])
        self.n1 = network_params['n1']  # number of filters at layer 1
        self.k1 = network_params['k1']  # width of the filters at layer 1 (final shape: d x k1)
        self.n2 = network_params['n2']  # number of filters at layer 2
        self.k2 = network_params['k2']  # width of the filters at layer 1 (final shape: d x k2)
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

        self._init_params()

    def _init_params(self, p=0.01):
        sigma1 = cp.sqrt(2/(p*self.d*self.k1*self.n1))
        sigma2 = cp.sqrt(2/(self.n1*self.k2*self.n2))
        sigma3 = cp.sqrt(2/self.f_size)

        self.F = []  # filters
        self.F.append(cp.random.normal(0.0, sigma1, (self.d, self.k1, self.n1)))
        self.F.append(cp.random.normal(0.0, sigma2, (self.n1, self.k2, self.n2)))
        self.W = cp.random.normal(0.0, sigma3, (self.K, self.f_size))  # FC layer weights

    def make_mf_matrix(self, F, n_len):
        dd, k, nf = F.shape
        V_F = []
        for i in range(nf):
            V_F.append(F[:,:,i].flatten(order='F'))
        V_F = cp.array(V_F)
        zero_nlen = cp.zeros((dd, nf))
        kk = n_len - k
        Mf = []
        for i in range(kk+1):
            tup = [zero_nlen.T for _ in range(kk + 1)]
            tup[i] = V_F
            Mf.append(cp.concatenate(tup, axis=1))
        Mf = cp.concatenate(Mf, axis=0)
        return Mf

    def make_mx_matrix(self, x_input, d, k, nf, n_len):
        X_input = x_input.reshape((-1, n_len), order='F')
        I_nf = cp.eye(nf)
        Mx = []
        for i in range(n_len - k + 1):
            Mx.append(cp.kron(I_nf, X_input[:d, i:i+k].ravel(order='F').T))
        Mx = cp.concatenate(Mx, axis=0)
        return Mx

    def compute_loss(self, X_batch, Y_batch, F, W):
        MFs = [self.make_mf_matrix(F[0], self.n_len), self.make_mf_matrix(F[1], self.n_len1)]
        _, _, P_batch = self.evaluate_classifier(X_batch, MFs, W)
        cross_entropy = -cp.sum(Y_batch * cp.log(P_batch))
        average_loss = cross_entropy / Y_batch.shape[1]
        return average_loss

    def evaluate_classifier(self, X_batch, MFs, W):
        s1 = MFs[0] @ X_batch
        X1_batch = cp.maximum(0, s1)
        s2 = MFs[1] @ X1_batch
        X2_batch = cp.maximum(0, s2)
        S_batch = W @ X2_batch
        P_batch = softmax(S_batch)
        return X1_batch, X2_batch, P_batch

    def compute_gradients(self, X_batch, Y_batch, F, W):
        MFs = [self.make_mf_matrix(F[0], self.n_len), self.make_mf_matrix(F[1], self.n_len1)]
        dF1 = cp.zeros(F[0].size)
        dF2 = cp.zeros(F[1].size)
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
            v = gi.T @ self.make_mx_matrix(xi, self.n1, self.k2, self.n2, self.n_len1)
            dF2 += fact*v
        G_batch = MFs[1].T @ G_batch
        G_batch = G_batch * (X1_batch > 0)
        for i in range(n):
            gi = G_batch[:, i]
            xi = X_batch[:, i]
            v = gi.T @ self.make_mx_matrix(xi, self.d, self.k1, self.n1, self.n_len)
            dF1 += fact*v
        return dW, dF1.reshape(F[0].shape, order='F'), dF2.reshape(F[1].shape, order='F')

    def compute_accuracy(self, X, y, F, W):
        MFs = [self.make_mf_matrix(F[0], self.n_len), self.make_mf_matrix(F[1], self.n_len1)]
        _, _, P = self.evaluate_classifier(X, MFs, W)
        y_pred = cp.argmax(P, axis=0) + 1
        correct = cp.sum(y == y_pred)
        return correct / y.shape[0]

    def compute_confusion_matrix(self, X, y, F, W):
        MFs = [self.make_mf_matrix(F[0], self.n_len), self.make_mf_matrix(F[1], self.n_len1)]
        _, _, P = self.evaluate_classifier(X, MFs, W)
        y_pred = cp.argmax(P, axis=0)
        confusion_matrix = cp.zeros((self.K, self.K), dtype=int)
        for true_class, pred_class in zip(y, y_pred):
            confusion_matrix[int(true_class) - 1, pred_class] += 1
        return confusion_matrix

    def plot_confusion_matrix(self, conf_matrix, classes, savepath, title='Confusion Matrix', cmap=plt.cm.Blues):
        conf_matrix_cpu = cp.asnumpy(conf_matrix)  # Transfer matrix back to CPU for plotting
        plt.clf()
        plt.figure(figsize=(15, 15))
        sns.heatmap(conf_matrix_cpu, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(savepath)

    def mini_batch_gd(self, n_update=500):
        train_losses = [self.compute_loss(self.X_train, self.Y_train, self.F, self.W)]
        val_losses = [self.compute_loss(self.X_val, self.Y_val, self.F, self.W)] if self.validation else []
        val_accs = [self.compute_accuracy(self.X_val, self.y_val, self.F, self.W)] if self.validation else []
        n = self.X_train.shape[1]
        for i in range(self.n_epochs):
            print(f"Epoch {i+1}/{self.n_epochs}")
            for j in tqdm.tqdm(range(1, int(n/self.n_batch) + 1)):
                start = (j-1)*self.n_batch
                end = j*self.n_batch
                perm = cp.random.permutation(n)
                X_batch = self.X_train[:, perm][:, start:end]
                Y_batch = self.Y_train[:, perm][:, start:end]
                dW, dF1, dF2 = self.compute_gradients(X_batch, Y_batch, self.F, self.W)
                self.W -= self.eta * dW
                self.F[0] -= self.eta * dF1
                self.F[1] -= self.eta * dF2
                if (i * self.n_batch + j) % n_update == 0:
                    current_train_loss = self.compute_loss(self.X_train, self.Y_train, self.F, self.W)
                    print(f"\t * Train loss: {current_train_loss:.4f}")
                    train_losses.append(current_train_loss)
                    if self.validation:
                        current_val_loss = self.compute_loss(self.X_val, self.Y_val, self.F, self.W)
                        print(f"\t * Validation loss: {current_val_loss:.4f}")
                        val_losses.append(current_val_loss)
                        current_val_acc = self.compute_accuracy(self.X_val, self.y_val, self.F, self.W)
                        val_accs.append(current_val_acc)
                        conf_mat = self.compute_confusion_matrix(self.X_val, self.y_val, self.F, self.W)
                        self.plot_confusion_matrix(conf_mat, CLASS_LABELS, f"./reports/figures/conf_mat_{i*self.n_batch + j}.png")
        return train_losses, val_losses, val_accs

    def run_training(self, n_update=500, figure_savepath=None, test_data=None, model_savepath=None):
        train_losses, val_losses, val_accs = self.mini_batch_gd(n_update)
        logger.info("Training completed.")
        
        if model_savepath:
            os.makedirs(model_savepath, exist_ok=True)
            cp.save(os.path.join(model_savepath, "F1.npy"), self.F[0])
            cp.save(os.path.join(model_savepath, "F2.npy"), self.F[1])
            cp.save(os.path.join(model_savepath, "W.npy"), self.W)

        if test_data:
            X_test, y_test = test_data
            X_test = cp.asarray(X_test)
            y_test = cp.asarray(y_test)
            accuracy = self.compute_accuracy(X_test, y_test, self.F, self.W)
            print(f"Accuracy on test data: {accuracy:.3f}")
            logger.info("Accuracy on test data: %.3f", accuracy)

        plt.clf()
        plt.plot(cp.asnumpy(cp.arange(0, len(train_losses)))*n_update, cp.asnumpy(train_losses), label="training loss")
        plt.plot(cp.asnumpy(cp.arange(0, len(val_losses)))*n_update, cp.asnumpy(val_losses), label="validation loss")
        plt.xlabel("update steps")
        plt.ylabel("cross-entropy loss")
        plt.legend()
        plt.title(f"Training curves (n_batch = {self.n_batch}, n_epochs = {self.n_epochs}, eta = {self.eta})")
        plt.grid()
        if figure_savepath:
            plt.savefig(figure_savepath + "_loss.png", bbox_inches='tight')
        plt.clf()
        plt.plot(cp.asnumpy(cp.arange(0, len(val_accs)))*n_update, cp.asnumpy(val_accs), label="validation accuracies")
        plt.xlabel("update steps")
        plt.ylabel("accuracy")
        plt.legend()
        plt.title(f"Accuracy (n_batch = {self.n_batch}, n_epochs = {self.n_epochs}, eta = {self.eta})")
        plt.grid()
        if figure_savepath:
            plt.savefig(figure_savepath + "_accuracy.png", bbox_inches='tight')

        logger.info("Figure saved.")

