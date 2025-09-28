import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import accuracy_score
import time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True

def load_dataset():
    
    data_lab = np.load('F:/Database/SignFi/lab/data_abs_lab_276.npy')
    y_lab = np.load('F:/Database/SignFi/lab/label_lab_276.npy')

    data_home = np.load('F:/Database/SignFi/home/data_abs_home_276.npy')
    y_home = np.load('F:/Database/SignFi/home/label_home_276.npy')

    data_labhome = np.concatenate((data_lab, data_home), axis = 3)
    y_labhome = np.concatenate((y_lab, y_home), axis = 0)

    X_trans = data_labhome.transpose(3, 0, 1, 2)

    sam, ts, subs, channl = X_trans.shape
    X = X_trans.reshape(sam, ts, subs * channl)

    encoder = OneHotEncoder(sparse_output = False)
    Y_ohe = encoder.fit_transform(y_labhome)

    X_train, X_test, y_train, y_test = train_test_split(X, Y_ohe, test_size = 0.2,
                                                        stratify = Y_ohe, random_state = 1, shuffle = True)
    
    norm = StandardScaler().fit(X_train.reshape(X_train.shape[0], -1))
    X_train_norm = norm.transform(X_train.reshape(X_train.shape[0], -1)).reshape(X_train.shape)
    X_test_norm = norm.transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape)
    
    return X_train_norm, y_train, X_test_norm, y_test
    
def MKFi(X_train, X_test, y_train, y_test, gamma, Lambda):
    
    def kernel_ridge_regression(X_train, X_test, y_train, gamma, Lambda):
        def gaussian_kernel(X, Y, gamma):
            normX = (X ** 2).sum(dim = 1, keepdim = True)
            normY = (Y ** 2).sum(dim = 1, keepdim = True).T
            D = normX + normY - 2 * X @ Y.T
            K_gauss = torch.exp(-gamma * D)
            
            return K_gauss
        
        K_train = gaussian_kernel(X_train, X_train, gamma) 
        K_test = gaussian_kernel(X_test, X_train, gamma)
        
        I = torch.eye(K_train.size(0), device = device)
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
                y_train, gamma, Lambda
            )
            y_test_preds.append(y_test_pred_win)
            alpha_wins.append(alpha)
            
        return torch.stack(y_test_preds, dim = -1), alpha_wins
    
    win_sizes = [1, 2, 4, 8]
    
    win_scores, alpha_all = list(), list()
    total_params = 0
    for n_wins in win_sizes:
        preds, alpha_wins = krr_wins(X_train, X_test, y_train, n_wins)
        win_scores.append(preds)
        alpha_all.append(alpha_wins)
        
        total_params += sum(alpha.numel() for alpha in alpha_wins)
    
    return torch.cat(win_scores, dim = -1), alpha_all, total_params

results = list()
for seed in range(1, 11):
    set_seed(seed)
    
    X_train, y_train, X_test, y_test = load_dataset()
    y_test_labels = np.argmax(y_test, axis = 1)
    
    X_train = torch.tensor(X_train, dtype = torch.float32, device = device)
    X_test = torch.tensor(X_test, dtype = torch.float32, device = device)
    y_train = torch.tensor(y_train, dtype = torch.float32, device = device)
    y_test = torch.tensor(y_test, dtype = torch.float32, device = device)

    start_time = time.time()
    win_scores_stack, alpha_all, n_params = MKFi(X_train, X_test, y_train, y_test, 0.0001, 0.001)
    
    y_test_pred_sum = win_scores_stack.sum(dim = -1)
    y_test_pred_labels = torch.argmax(y_test_pred_sum, dim = 1).cpu().numpy()
    
    acc = accuracy_score(y_test_labels, y_test_pred_labels) * 100
    Trtime = time.time() - start_time
    
    results.append({
        "Seed": seed,
        "Accuracy": acc,
        "Time": Trtime
    })
    print(f'Seed: {seed} | Acc: {acc:.2f} | Params: {n_params} | Time: {Trtime:.2f} s')

df = pd.DataFrame(results)

acc_mean, acc_std = df['Accuracy'].mean(), df['Accuracy'].std()
time_mean, time_std = df['Time'].mean(), df['Time'].std()

print(f'\nAccuracy: {acc_mean:.2f} ± {acc_std:.2f}')
print(f'Time: {time_mean:.2f} ± {time_std:.2f} s')


# Seed: 1 | Acc: 100.00 | Params: 27423360 | Time: 1.81 s
# Seed: 2 | Acc: 100.00 | Params: 27423360 | Time: 1.69 s
# Seed: 3 | Acc: 100.00 | Params: 27423360 | Time: 1.94 s
# Seed: 4 | Acc: 100.00 | Params: 27423360 | Time: 1.94 s
# Seed: 5 | Acc: 100.00 | Params: 27423360 | Time: 1.83 s
# Seed: 6 | Acc: 100.00 | Params: 27423360 | Time: 1.81 s
# Seed: 7 | Acc: 100.00 | Params: 27423360 | Time: 1.92 s
# Seed: 8 | Acc: 100.00 | Params: 27423360 | Time: 1.95 s
# Seed: 9 | Acc: 100.00 | Params: 27423360 | Time: 1.82 s
# Seed: 10 | Acc: 100.00 | Params: 27423360 | Time: 1.95 s

# Accuracy: 100.00 ± 0.00
# Time: 1.87 ± 0.09 s
