from scipy.signal import convolve2d
import numpy as np
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

n1 = 1  # number of filters at layer 1
k1 = 2  # width of the filters at layer 1 (final shape: d x k1)
n2 = 1  # number of filters at layer 2
k2 = 2  # width of the filters at layer 1 (final shape: d x k2)
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