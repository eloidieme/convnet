import numpy as np
from cnnClassifier.models.train import CNN

## TO-DO
W = np.load('./models/best_with_compensating/W_best.npy')
F = [np.load('./models/best_with_compensating/F1_best.npy'), np.load('./models/best_with_compensating/F2_best.npy')]
MFs = [CNN.make_mf_matrix(F[0], n_len), CNN.make_mf_matrix(F[1], n_len1)]
P_pred = CNN.evaluate_classifier(X_test, MFs, W)[-1]
y_pred = np.argmax(P_pred, axis=0) + 1