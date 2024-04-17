import numpy as np
from cnnClassifier.data.make_dataset import DataLoader
from cnnClassifier.models.train import CNN
from cnnClassifier.models.train_pytorch import CNN_Torch
import cProfile, pstats

def main_training():
    load = DataLoader()

    data_file = np.load('data/train_val_data.npz')

    X_train = data_file['X_train']
    Y_train = data_file['Y_train']
    y_train = data_file['y_train']
    X_val = data_file['X_val']
    Y_val = data_file['Y_val']
    y_val = data_file['y_val']

    network_params = {
        'n1': 20,
        'k1': 5,
        'n2': 20,
        'k2': 3,
        'eta': 0.001,
        'rho': 0.9
    }

    gd_params = {
        'n_batch': 100,
        'n_epochs': 2
    }

    #model = CNN(X_train, Y_train, y_train, network_params, gd_params, load.meta, validation=(X_val, Y_val, y_val), seed=400)
    model = CNN_Torch(X_train, Y_train, y_train, network_params, gd_params, load.meta, validation=(X_val, Y_val, y_val), balanced=False, seed=400)

    profiler = cProfile.Profile()
    profiler.enable()
    model.run_training("./reports/figures/train_fig", test_data=(X_val, y_val), model_savepath="./models")
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats()

if __name__ == "__main__":
    main_training()
