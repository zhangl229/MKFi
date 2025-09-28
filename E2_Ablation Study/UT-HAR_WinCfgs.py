import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
from sklearn.metrics import accuracy_score
import time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_dataset():
    
    X_train = np.load('../X_train.npy')
    X_test = np.load('../X_test.npy')

    y_train = np.load('../y_train.npy')
    y_test = np.load('../y_test.npy')

    norm = StandardScaler().fit(X_train.reshape(X_train.shape[0], -1))
    X_train_norm = norm.transform(X_train.reshape(X_train.shape[0], -1)).reshape(X_train.shape)
    X_test_norm = norm.transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape)

    encoder = OneHotEncoder(sparse_output = False)
    Ytr_ohe = encoder.fit_transform(y_train.reshape(-1, 1))
    Yts_ohe = encoder.fit_transform(y_test.reshape(-1, 1))
    
    return X_train_norm, Ytr_ohe, X_test_norm, Yts_ohe
    
def MKFi(X_train, X_test, y_train, y_test, cfgs):
    
    def kernel_ridge_regression(X_train, X_test, y_train, gamma, Lambda):
        def gaussian_kernel(X, Y, gamma):
            normX = (X ** 2).sum(dim = 1, keepdim = True)
            normY = (Y ** 2).sum(dim = 1, keepdim = True).T
            D = normX + normY - 2 * X @ Y.T
            K_gauss = torch.exp(-gamma * D)
            
            return K_gauss
        
        K_train = gaussian_kernel(X_train, X_train, gamma) 
        K_test = gaussian_kernel(X_test, X_train, gamma)
        
        I = torch.eye(K_train.size(0), dtype = torch.float64, device = device)
        alpha = torch.linalg.solve(K_train + Lambda * I, y_train)

        y_pred = K_test @ alpha
        
        return y_pred, alpha
    
    def krr_wins(X_train, X_test, y_train, n_wins):
        
        _, time_steps, features = X_train.shape
        time_step_div, remainder = divmod(time_steps, n_wins)
        
        y_test_preds, alpha_wins = list(), list()
        for win in range(n_wins):
            start = win * time_step_div
            end = (win + 1) * time_step_div if win != n_wins - 1 else time_steps
            
            X_train_win = X_train[:, start:end, :]
            X_test_win = X_test[:, start:end, :]
            
            y_test_pred_win, alpha = kernel_ridge_regression(
                X_train_win.reshape(X_train.shape[0], -1), 
                X_test_win.reshape(X_test.shape[0], -1),
                y_train, 0.0003, 0.001
            )
            y_test_preds.append(y_test_pred_win)
            alpha_wins.append(alpha)
            
        return torch.stack(y_test_preds, dim = -1), alpha_wins
    
    win_scores, alpha_all = list(), list()
    total_params = 0
    for n_wins in cfgs:
        preds, alpha_wins = krr_wins(X_train, X_test, y_train, n_wins)
        win_scores.append(preds)
        alpha_all.append(alpha_wins)
        
        total_params += sum(alpha.numel() for alpha in alpha_wins)
    
    return torch.cat(win_scores, dim = -1), alpha_all, total_params

X_train, y_train, X_test, y_test = load_dataset()
y_test_labels = np.argmax(y_test, axis = 1)

X_train = torch.tensor(X_train, dtype = torch.float64, device = device)
X_test = torch.tensor(X_test, dtype = torch.float64, device = device)
y_train = torch.tensor(y_train, dtype = torch.float64, device = device)
y_test = torch.tensor(y_test, dtype = torch.float64, device = device)

cfgs_list = [ 
    [1, 2], 
    [1, 4], 
    [1, 8], 
    [2, 4], 
    [2, 8], 
    [4, 8], 
    [1, 2, 4], 
    [1, 2, 8], 
    [1, 4, 8], 
    [2, 4, 8], 
    [1, 2, 4, 8]
]
results = {}
for cfgs in cfgs_list:
    
    start_time = time.time()
    win_scores_stack, alpha_all, n_params = MKFi(X_train, X_test, y_train, y_test, cfgs)

    y_test_pred_sum = win_scores_stack.sum(dim = -1)
    y_test_pred_labels = torch.argmax(y_test_pred_sum, dim = 1).cpu().numpy()

    acc = accuracy_score(y_test_labels, y_test_pred_labels) * 100

    time_taken = time.time() - start_time
    
    print(f'Cfgs: {cfgs}, Acc: {acc:.2f}')
    print(f"Params: {n_params}")
    print(f"Time taken: {time_taken:.2f} s\n")

    results[tuple(cfgs)] = {
        'Accuracy': acc,
        'Params': n_params / 1e6,
        'RunTime': time_taken
    }

data = pd.DataFrame([
    {'Cfgs': cfg, 'Accuracy': vals['Accuracy'], 'Params (M)': vals['Params'], 'RunTime': vals['RunTime']}
    for cfg, vals in results.items()
])

path = '../UT-HAR_WinCfgs.csv'
data.to_csv(path, index = False)

