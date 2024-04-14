from scipy.signal import convolve2d
import numpy as np
from copy import deepcopy
from cnnClassifier.data.make_dataset import DataLoader

np.random.seed(400)

load = DataLoader()
data = load.make_data()

data_file = np.load('data/train_val_data.npz')

X_train = data_file['X_train']
Y_train = data_file['Y_train']
y_train = data_file['y_train']
X_val = data_file['X_val']
Y_val = data_file['Y_val']
y_val = data_file['y_val']

n1 = 5  # number of filters at layer 1
k1 = 5  # width of the filters at layer 1 (final shape: d x k1)
n2 = 5  # number of filters at layer 2
k2 = 5  # width of the filters at layer 1 (final shape: d x k2)
eta = 0.001  # learning rate
rho = 0.9  # momentum term

d = load.meta['dimensionality']  # dimensionality
K = load.meta['n_classes']  # number of classes
n_len = load.meta['n_len']  # max len of a name

n_len1 = n_len - k1 + 1
n_len2 = n_len1 - k2 + 1

f_size = n2 * n_len2

p = 0.01  # density of non-zero elements in X_input
sigma1 = np.sqrt(2/(p*d*k1*n1))
sigma2 = np.sqrt(2/(n1*k2*n2))
sigma3 = np.sqrt(2/f_size)

F = []  # filters
F.append(np.random.normal(0.0, sigma1, (d, k1, n1)))
F.append(np.random.normal(0.0, sigma2, (n1, k2, n2)))
W = np.random.normal(0.0, sigma3, (K, f_size))  # FC layer weights

x_input = X_train[:, 0]  # sample name
X_input = x_input.reshape((-1, n_len), order='F')

def make_mf_matrix(F, n_len):
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


def make_mx_matrix(x_input, d, k, nf, n_len):
    X_input = x_input.reshape((-1, n_len), order='F')
    I_nf = np.eye(nf)
    Mx = []
    for i in range(n_len - k + 1):
        Mx.append(np.kron(I_nf, X_input[:d, i:i+k].ravel(order='F').T))
    Mx = np.concatenate(Mx, axis=0)
    return Mx

###### TEST MF & MX MATRICES ######


d, k, nf = F[0].shape
assert ((n_len-k+1)*nf, n_len*d) == make_mf_matrix(F[0], n_len).shape
assert ((n_len - k + 1)*nf, k*nf *
        d) == make_mx_matrix(X_input, d, k, nf, n_len).shape

X_t = np.random.standard_normal((4, 4))
F_t = np.random.standard_normal((4, 2, 1))

s1 = convolve2d(X_t, np.flip(F_t[:, :, 0]), mode='valid')[0]
s2 = []
for i in range(3):
    s2.append(X_t[:, i:i+2].flatten() @ F_t[:, :, 0].flatten())
s2 = np.array(s2)
mf = make_mf_matrix(F_t, 4)
s3 = mf @ X_t.flatten(order='F')
d, k, nf = F_t.shape
mx = make_mx_matrix(X_t.flatten(order='F'), d, k, nf, 4)
s4 = mx @ F_t[:, :, 0].flatten(order='F')
print(s1, s2, s3, s4, sep='\n')
assert np.allclose(s1, s2) and np.allclose(s2, s3) and np.allclose(s3, s4)

d, k, nf = F[0].shape
MX = make_mx_matrix(x_input, d, k, nf, n_len)
MF = make_mf_matrix(F[0], n_len)
s1 = MX @ F[0].flatten(order='F')
s2 = MF @ x_input
assert np.allclose(s1, s2)

##################################


def softmax(x):
    x = x - np.max(x, axis=0)  # for numerical stability
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=0)


def compute_loss(X_batch, Y_batch, F, W):
    MFs = [make_mf_matrix(F[0], n_len), make_mf_matrix(F[1], n_len1)]
    _, _, P_batch = evaluate_classifier(X_batch, MFs, W)
    fact = 1/X_batch.shape[1]
    lcross_sum = np.sum(np.diag(-Y_batch.T@np.log(P_batch)))
    return fact*lcross_sum

def evaluate_classifier(X_batch, MFs, W):
    s1 = MFs[0] @ X_batch
    X1_batch = np.maximum(0, s1)
    s2 = MFs[1] @ X1_batch
    X2_batch = np.maximum(0, s2)
    S_batch = W @ X2_batch
    P_batch = softmax(S_batch)
    return X1_batch, X2_batch, P_batch


def compute_gradients(X_batch, Y_batch, F, W):
    MFs = [make_mf_matrix(F[0], n_len), make_mf_matrix(F[1], n_len1)]
    dF1 = np.zeros(np.prod(F[0].shape))
    dF2 = np.zeros(np.prod(F[1].shape))
    X1_batch, X2_batch, P_batch = evaluate_classifier(X_batch, MFs, W)
    n = X_batch.shape[1]
    fact = 1/n
    G_batch = -(Y_batch - P_batch)
    dW = fact*(G_batch@X2_batch.T)
    G_batch = W.T @ G_batch
    G_batch = G_batch * (X2_batch > 0)
    for i in range(n):
        gi = G_batch[:, i]
        xi = X1_batch[:, i]
        v = gi.T @ make_mx_matrix(xi, n1, k2, n2, n_len1)
        dF2 += fact*v
    G_batch = MFs[1].T @ G_batch
    G_batch = G_batch * (X1_batch > 0)
    for i in range(n):
        gi = G_batch[:, i]
        xi = X_batch[:, i]
        v = gi.T @ make_mx_matrix(xi, d, k1, n1, n_len)
        dF1 += fact*v
    return dW, dF1.reshape(d, k1, n1), dF2.reshape(n1, k2, n2)


def numerical_gradients(X_inputs, Ys, F, W, h):
    dFs = [np.zeros_like(F[0]), np.zeros_like(F[1])]
    ns = [F[0].shape[2], F[1].shape[2]]

    try_F = deepcopy(F)
    for l in range(len(try_F)):
        for i in range(ns[l]):
            dF = np.zeros(np.prod(F[l][:, :, i].shape))
            for j in range(len(dF)):
                try1_F = deepcopy(try_F)
                try1_F[l][:, :, i].flat[j] -= h
                l1 = compute_loss(X_inputs, Ys, try1_F, W)

                try2_F = deepcopy(try_F)
                try2_F[l][:, :, i].flat[j] += h
                l2 = compute_loss(X_inputs, Ys, try2_F, W)
                dF[j] = (l2 - l1) / (2 * h)
            dFs[l][:, :, i] = dF.reshape(F[l][:, :, i].shape)

    try_W = np.copy(W)
    dW = np.zeros(np.prod(try_W.shape))
    for j in range(len(dW)):
        W_try1 = np.array(try_W, copy=True)
        W_try1.flat[j] = try_W.flat[j] - h

        l1 = compute_loss(X_inputs, Ys, F, W_try1)

        W_try2 = np.array(try_W, copy=True)
        W_try2.flat[j] = try_W.flat[j] + h

        l2 = compute_loss(X_inputs, Ys, F, W_try2)

        dW[j] = (l2 - l1) / (2 * h)
    dW = dW.reshape(try_W.shape)

    return dW, dFs[0], dFs[1]


X_batch = X_train[:, :2]
Y_batch = Y_train[:, :2]
MFs = [make_mf_matrix(F[0], n_len), make_mf_matrix(F[1], n_len1)]
X1_batch, X2_batch, P_batch = evaluate_classifier(X_batch, MFs, W)
print(X_batch.shape, X1_batch.shape, X2_batch.shape, P_batch.shape, sep='\n')

dW, dF1, dF2 = compute_gradients(X_batch, Y_batch, F, W)
dWn, dF1n, dF2n = numerical_gradients(X_batch, Y_batch, F, W, 1e-6)

print(np.max(np.abs(dW - dWn)), np.max(np.abs(dF1 - dF1n)),
      np.max(np.abs(dF2 - dF2n)), sep='\n')
