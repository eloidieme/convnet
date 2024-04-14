import numpy as np
from time import time
from pathlib import Path
from typing import Tuple

from cnnClassifier import logger

PROJECT_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = Path(f"{PROJECT_DIR}/data")

class DataLoader:
    def __init__(self) -> None:
        self.names_path = Path(f"{DATA_DIR}/ascii_names.txt")
        self.val_idx = Path(f"{DATA_DIR}/Validation_Inds.txt")
        self.all_names, self.y = self._extract_names()
        self.meta = self._make_metadata(self.all_names, self.y)

    def _extract_names(self) -> Tuple[np.ndarray, np.ndarray]:
        with open(self.names_path, 'r') as file:
            S = file.read()

        # Split the string into lines (names)
        names = S.split('\n')
        # Remove the last empty name if present
        if len(names[-1]) < 1:
            names.pop()

        y = np.zeros(len(names))
        all_names = []

        for i in range(len(names)):
            nn = names[i].split()
            l = int(nn[-1])  # Convert the last element to an integer
            if len(nn) > 2:
                name = ' '.join(nn[:-1])  # Join all but the last element
            else:
                name = nn[0]

            name = name.lower()
            y[i] = l
            all_names.append(name)

        all_names = np.array(all_names)
    
        return all_names, y

    def _make_metadata(self, all_names, y):
        characters = np.unique(list(''.join(all_names))) # unique characters
        d = len(characters) # dimensionality
        n_len = np.max(np.vectorize(len)(all_names)) # max len of a name
        K = len(np.unique(y)) # number of classes
        char_to_ind = {}
        for idx, char in enumerate(list(characters)):
            char_to_ind[char] = idx
        
        return {'dimensionality': d, 'n_classes': K, 'n_len': n_len, 'char_to_ind': char_to_ind}

    def _make_X(self, all_names: np.ndarray):
        X = []
        for name in all_names:
            encoded = self.encode_name(name)
            X.append(encoded)

        return np.array(X).T
    
    def _make_Y(self, y: np.ndarray):
        Y = []
        for label in y:
            one_hot = np.zeros(self.meta['n_classes'])
            one_hot[int(label) - 1] = 1
            Y.append(one_hot)
        Y = np.array(Y).T
        return Y

    def _make_split(self, X, Y, y):
        with open(self.val_idx, 'r') as file:
            S = file.read()

        indices_str = S.split(' ')[:-1]
        indices = []
        for idx in indices_str:
            indices.append(int(idx))

        X_train, X_val, Y_train, Y_val, y_train, y_val = np.delete(X, indices, axis=1), X[:,indices], np.delete(Y, indices, axis=1), Y[:,indices], np.delete(y, indices), y[indices]
        return {"train": (X_train, Y_train, y_train), "validation": (X_val, Y_val, y_val)}
    
    def encode_name(self, name: str):
        d, n_len, char_to_ind = self.meta['dimensionality'], self.meta['n_len'], self.meta['char_to_ind']
        encoded = []
        if len(name) > n_len:
            name = name[:n_len]
        name = name.lower()
        for char in name:
            oh = [0]*d
            oh[char_to_ind[char]] = 1
            encoded.append(oh)
        while len(encoded) < n_len:
            encoded.append([0]*d)

        encoded = np.array(encoded).T
        return np.ravel(encoded)

    def make_data(self, save_data: bool = True):
        X = self._make_X(self.all_names)
        Y = self._make_Y(self.y)
        data = self._make_split(X, Y, self.y)

        if save_data:
            np.savez(
                Path(f'{DATA_DIR}/train_val_data'),
                X_train=data['train'][0], 
                Y_train=data['train'][1],
                y_train=data['train'][2],
                X_val=data['validation'][0],
                Y_val=data['validation'][1],
                y_val=data['validation'][2]
            )
        return data
    
if __name__ == '__main__':
    load = DataLoader()
    logger.info("Making train and validation data.")
    data = load.make_data()
    logger.info(f"Data successfully imported in '{DATA_DIR}/train_val_data'.")


