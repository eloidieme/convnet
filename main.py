import numpy as np
from cnnClassifier.data.make_dataset import DataLoader
from cnnClassifier.models.train import CNN

load = DataLoader()
data = load.make_data()

data_file = np.load('data/train_val_data.npz')

X_train = data_file['X_train']
Y_train = data_file['Y_train']
y_train = data_file['y_train']
X_val = data_file['X_val']
Y_val = data_file['Y_val']
y_val = data_file['y_val']

network_params = {
    'n1': 5,
    'k1': 5,
    'n2': 5,
    'k2': 5,
    'eta': 0.001,
    'rho': 0.9
}

gd_params = {
    'n_batch': 100,
    'n_epochs': 100
}

model = CNN(X_train, Y_train, network_params, gd_params, load.meta,
            validation=(X_val, Y_val, y_val), seed=400)

model.run_training(500, "./reports/figures/train_fig", test_data=(X_val, y_val), model_savepath="./models")