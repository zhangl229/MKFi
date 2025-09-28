import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
from sklearn.metrics import accuracy_score
import pandas as pd


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_dataset():
    
    data_path = '../ARIL_data/'

    X_train = np.load(data_path + 'X_train.npy')
    X_test = np.load(data_path + 'X_test.npy')

    y_train = np.load(data_path + 'y_train_activity.npy')
    y_test = np.load(data_path + 'y_test_activity.npy')

    norm = StandardScaler().fit(X_train.reshape(X_train.shape[0], -1))
    X_train_norm = norm.transform(X_train.reshape(X_train.shape[0], -1)).reshape(X_train.shape)
    X_test_norm = norm.transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape)

    encoder = OneHotEncoder(sparse_output = False)
    Ytr_ohe = encoder.fit_transform(y_train.reshape(-1, 1))
    Yts_ohe = encoder.fit_transform(y_test.reshape(-1, 1))
    
    return X_train_norm, Ytr_ohe, X_test_norm, Yts_ohe
    
def MKFi(X_train, X_test, y_train, y_test, n_wins):
    
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
        
        return y_pred
    
    def krr_wins(X_train, X_test, y_train, n_wins):
        
        _, time_steps, features = X_train.shape
        time_step_div, remainder = divmod(time_steps, n_wins)
        
        y_test_preds = list()
        for win in range(n_wins):
            start = win * time_step_div
            end = (win + 1) * time_step_div if win != n_wins - 1 else time_steps
            
            X_train_win = X_train[:, start:end, :]
            X_test_win = X_test[:, start:end, :]
            
            y_test_pred_win = kernel_ridge_regression(
                X_train_win.reshape(X_train.shape[0], -1), 
                X_test_win.reshape(X_test.shape[0], -1),
                y_train, 0.001, 0.001
            )
            y_test_preds.append(y_test_pred_win)
            
        return torch.stack(y_test_preds, dim = -1)
    
    preds = krr_wins(X_train, X_test, y_train, n_wins)
    
    return preds

X_train, y_train, X_test, y_test = load_dataset()
y_test_labels = np.argmax(y_test, axis = 1)

X_train = torch.tensor(X_train, dtype = torch.float32, device = device)
X_test = torch.tensor(X_test, dtype = torch.float32, device = device)
y_train = torch.tensor(y_train, dtype = torch.float32, device = device)
y_test = torch.tensor(y_test, dtype = torch.float32, device = device)

n_wins = [1, 2, 4, 8, 16, 32]
results = []
for n_win in n_wins:
    
    preds = MKFi(X_train, X_test, y_train, y_test, n_win)
    
    for w in range(n_win):
        
        scores_w = preds[:, :, w]  
        pred_labels_w = torch.argmax(scores_w, dim = 1).cpu().numpy()
        acc_w = accuracy_score(y_test_labels, pred_labels_w) * 100
        
        results.append({
            'NWin': n_win,
            'Window': f'win{w+1}',
            'Accuracy': acc_w
        })

    y_test_pred_sum = preds.sum(dim = -1)
    y_test_pred_labels = torch.argmax(y_test_pred_sum, dim = 1).cpu().numpy()

    acc_sum = accuracy_score(y_test_labels, y_test_pred_labels) * 100
    print(f'NWin: {n_win}, Testing Accuracy: {acc_sum:.2f}')
    
    results.append({
        'NWin': n_win,
        'Window': 'sum',
        'Accuracy': acc_sum
    })

data = pd.DataFrame(results, columns = ['NWin', 'Window', 'Accuracy'])

path = '../ARIL_NWin.csv'
data.to_csv(path, index = False)

