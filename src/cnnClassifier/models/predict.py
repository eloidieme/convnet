import numpy as np
import torch
from cnnClassifier.data.make_dataset import DataLoader
from cnnClassifier.models.model import CNN, CLASS_LABELS

k1 = 5  # width of the filters at layer 1 (final shape: d x k1)
load = DataLoader()
d = load.meta['dimensionality']  # dimensionality
K = load.meta['n_classes']  # number of classes
n_len = load.meta['n_len']  # max len of a name
n_len1 = n_len - k1 + 1

def predict(names, F1_path, F2_path, W_path):
    X_test = np.zeros((d*n_len, len(names)))
    for idx, name in enumerate(names):
        x_input = load.encode_name(name)
        X_test[:,idx] = x_input
    X_test = torch.tensor(X_test, dtype=torch.float)

    W = torch.load(W_path)
    F = [torch.load(F1_path), torch.load(F2_path)]
    MFs = [CNN.make_mf_matrix(F[0], n_len), CNN.make_mf_matrix(F[1], n_len1)]
    P_pred = CNN.evaluate_classifier(X_test, MFs, W)[-1]
    y_pred = np.argmax(P_pred, axis=0)
    return np.array(CLASS_LABELS)[y_pred]
