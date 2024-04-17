import numpy as np
from cnnClassifier.data.make_dataset import DataLoader
from cnnClassifier.models.model import CNN

def main_training():
    load = DataLoader()

    data_file = np.load('data/train_val_data.npz')

    X_train = data_file['X_train']
    Y_train = data_file['Y_train']
    y_train = data_file['y_train']
    X_val = data_file['X_val']
    Y_val = data_file['Y_val']
    y_val = data_file['y_val']

    balanced = True

    if balanced:
        eta = 0.1
    else:
        eta = 0.001

    network_params = {
        'n1': 40,
        'k1': 5,
        'n2': 30,
        'k2': 3,
        'eta': eta,
        'rho': 0.9
    }

    gd_params = {
        'n_batch': 100,
        'n_epochs': 50
    }

    model = CNN(X_train, Y_train, y_train, network_params, gd_params, load.meta, validation=(X_val, Y_val, y_val), balanced=True, seed=400)
    model.run_training("./reports/figures/train_fig", test_data=(X_val, y_val), model_savepath="./models")

if __name__ == "__main__":
    main_training()