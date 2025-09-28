import scipy.io
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
from sklearn.metrics import accuracy_score


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_dataset(test_user_idx = 1):
    
    dataset = scipy.io.loadmat('F:/Database/SignFi/trans_coding/dataset_lab_150.mat')
    user_data = [np.abs(dataset[f'csi{i+1}']) for i in range(5)]
    label_all = dataset['label'].squeeze()

    samples_per_user = 1500

    X_all = []
    for data in user_data:
        data = data.transpose(3, 0, 1, 2)
        sam, ts, sbs, chn = data.shape
        X_all.append(data.reshape(sam, ts, sbs * chn))

    X_train = np.concatenate([X_all[i] for i in range(5) if i != test_user_idx], axis = 0)
    y_train = np.concatenate([label_all[i * samples_per_user : (i + 1) * samples_per_user] for i in range(5) if i != test_user_idx])

    X_test = X_all[test_user_idx]
    y_test = label_all[test_user_idx * samples_per_user : (test_user_idx + 1) * samples_per_user]
    
    Znorm = StandardScaler().fit(X_train.reshape(X_train.shape[0], -1))
    X_train_norm = Znorm.transform(X_train.reshape(X_train.shape[0], -1)).reshape(X_train.shape)
    X_test_norm = Znorm.transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape)
    
    encoder = OneHotEncoder(sparse_output = False)
    Ytr_ohe = encoder.fit_transform(y_train.reshape(-1, 1))
    Yts_ohe = encoder.fit_transform(y_test.reshape(-1, 1))
    
    return X_train_norm, Ytr_ohe, X_test_norm, Yts_ohe

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

acc_list = []
for ts_idx in range(5):
    
    X_train, y_train, X_test, y_test = load_dataset(ts_idx)
    y_test_labels = np.argmax(y_test, axis = 1)
    
    X_train = torch.tensor(X_train, dtype = torch.float32, device = device)
    X_test = torch.tensor(X_test, dtype = torch.float32, device = device)
    y_train = torch.tensor(y_train, dtype = torch.float32, device = device)
    y_test = torch.tensor(y_test, dtype = torch.float32, device = device)
    
    win_scores_stack, alpha_all, n_params = MKFi(X_train, X_test, y_train, y_test, 0.0002, 0.001)
    
    y_test_pred_sum = win_scores_stack.sum(dim = -1)
    y_test_pred_labels = torch.argmax(y_test_pred_sum, dim = 1).cpu().numpy()
    
    acc = accuracy_score(y_test_labels, y_test_pred_labels) * 100
    acc_list.append(acc)
    print(f'Test User {ts_idx+1}  →  Acc: {acc:.2f}')
    
print(f'LOSO MeanAcc: {np.mean(acc_list):.2f}')


# Test User 1  →  Acc: 58.67
# Test User 2  →  Acc: 66.00
# Test User 3  →  Acc: 59.33
# Test User 4  →  Acc: 76.47
# Test User 5  →  Acc: 14.20
# LOSO MeanAcc: 54.93
