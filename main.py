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

# USE HE INIT FOR SIGMAS
sigma1 = 0.3
sigma2 = 0.3
sigma3 = 0.3

F = []
F.append(np.random.normal(0.0, sigma1, (d, k1, n1)))
F.append(np.random.normal(0.0, sigma2, (n1, k2, n2)))
W = np.random.normal(0.0, sigma3, (K, f_size))

def make_mf_matrix(F, n_len):
    dd, k, nf = F.shape
    V_F = []
    for i in range(nf):
        V_F.append(F[:,:,i].flatten())
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

def make_mx_matrix(X_input, d, k, nf, n_len):
    I_nf = np.eye(nf)
    Mx = []
    for i in range(n_len - k + 1):
        Mx.append(np.kron(I_nf, X_input[:d,i:i+k].ravel(order='F').T))
    Mx = np.concatenate(Mx, axis=0)
    return Mx

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def evaluate_classifier(X_batch, MFs, W):
    s1 = MFs[0] @ X_batch
    X1_batch = np.maximum(np.zeros_like(s1), s1)
    s2 = MFs[1] @ X1_batch
    X2_batch = np.maximum(np.zeros_like(s2), s2)
    S_batch = W@X2_batch
    P_batch = softmax(S_batch)
    return X1_batch, X2_batch, P_batch

def compute_loss(X_batch, Y_batch, MFs, W):
    _, _, P_batch = evaluate_classifier(X_batch, MFs, W)
    fact = 1/X_batch.shape[1]
    lcross_sum = np.sum(np.diag(-Y_batch.T@np.log(P_batch)))
    return fact*lcross_sum

def compute_gradients(X_batch, Y_batch, MFs, W):
    X1_batch, X2_batch, P_batch = evaluate_classifier(X_batch, MFs, W)
    n = X_batch.shape[1]
    fact = 1/n
    G_batch = -(Y_batch - P_batch)
    dW = fact*(G_batch@X2_batch.T)
    G_batch = W.T @ G_batch
    G_batch = G_batch * (X2_batch > 0)
    g_0 = G_batch[:, 0]
    X_0 = np.reshape(X1_batch[:, 0], (n1, -1))
    v = g_0.T @ make_mx_matrix(X_0, n1, k2, n2, n_len1)
    dF2 = fact*v
    for i in range(1, n):
        gi = G_batch[:, i]
        Xi = np.reshape(X1_batch[:, i], (n1, -1))
        v = gi.T @ make_mx_matrix(Xi, n1, k2, n2, n_len1)
        dF2 += fact*v
    G_batch = MFs[1].T @ G_batch
    G_batch = G_batch * (X1_batch > 0)
    g_0 = G_batch[:, 0]
    X_0 = np.reshape(X_batch[:, 0], (d, -1))
    v = g_0.T @ make_mx_matrix(X_0, d, k1, n1, n_len)
    dF1 = fact*v
    for i in range(1, n):
        gi = G_batch[:, i]
        Xi = np.reshape(X_batch[:, i], (d, -1))
        v = gi.T @ make_mx_matrix(Xi, d, k1, n1, n_len)
        dF1 += fact*v
    return dW, dF1.reshape(d, k1, n1), dF2.reshape(n1, k2, n2)   

def NumericalGradient(X_inputs, Ys, ConvNet, h):
    try_ConvNet = ConvNet.copy()
    Gs = [None] * (len(ConvNet['F']) + 1)  # Create a list to store gradients for each F and W

    # Compute gradients for convolutional layers
    for l in range(len(ConvNet['F'])):
        try_ConvNet['F'][l] = np.array(ConvNet['F'][l], copy=True)
        
        Gs[l] = np.zeros_like(ConvNet['F'][l])
        nf = ConvNet['F'][l].shape[2]
        
        for i in range(nf):
            try_ConvNet['F'][l] = np.array(ConvNet['F'][l], copy=True)
            F_try = np.squeeze(ConvNet['F'][l][:, :, i])
            G = np.zeros(np.prod(F_try.shape))
            
            for j in range(len(G)):
                F_try1 = np.array(F_try, copy=True)
                F_try1.flat[j] = F_try.flat[j] - h
                try_ConvNet['F'][l][:, :, i] = F_try1.reshape(F_try.shape)
                
                l1 = compute_loss(X_inputs, Ys, MFs, try_ConvNet['W'])
                
                F_try2 = np.array(F_try, copy=True)
                F_try2.flat[j] = F_try.flat[j] + h
                try_ConvNet['F'][l][:, :, i] = F_try2.reshape(F_try.shape)
                
                l2 = compute_loss(X_inputs, Ys, MFs, try_ConvNet['W'])
                
                G[j] = (l2 - l1) / (2 * h)
                try_ConvNet['F'][l][:, :, i] = F_try  # Reset to original F
            
            Gs[l][:, :, i] = G.reshape(F_try.shape)
    
    # Compute the gradient for the fully connected layer
    W_try = ConvNet['W']
    G = np.zeros(np.prod(W_try.shape))
    for j in range(len(G)):
        W_try1 = np.array(W_try, copy=True)
        W_try1.flat[j] = W_try.flat[j] - h
        try_ConvNet['W'] = W_try1
        
        l1 = compute_loss(X_inputs, Ys, MFs, try_ConvNet['W'])
        
        W_try2 = np.array(W_try, copy=True)
        W_try2.flat[j] = W_try.flat[j] + h
        try_ConvNet['W'] = W_try2
        
        l2 = compute_loss(X_inputs, Ys, MFs, try_ConvNet['W'])
        
        G[j] = (l2 - l1) / (2 * h)
        try_ConvNet['W'] = W_try  # Reset to original W
    
    Gs[-1] = G.reshape(W_try.shape)
    
    return Gs

X_batch = X_train[:,:100]
Y_batch = Y_train[:,:100]
ConvNet = {'F': F, 'W': W}
MFs = [make_mf_matrix(F[0], n_len), make_mf_matrix(F[1], n_len1)]

dW, dF1, dF2 = compute_gradients(X_batch, Y_batch, MFs, W)
Gs = NumericalGradient(X_batch, Y_batch, ConvNet, 1e-6)

print(len(Gs))
print(np.max(np.abs(dF1 - Gs[0])))
print(np.max(np.abs(dF2 - Gs[1])))
print(np.max(np.abs(dW - Gs[2])))