import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import accuracy_score
import pandas as pd


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
                y_train, gamma, Lambda
            )
            y_test_preds.append(y_test_pred_win)
            
        return torch.stack(y_test_preds, dim = -1)
    
    win_sizes = [1, 2, 4, 8]
    
    win_scores = list()
    for n_wins in win_sizes:
        preds = krr_wins(X_train, X_test, y_train, n_wins)
        win_scores.append(preds)
    
    return torch.cat(win_scores, dim = -1)

X_train, y_train, X_test, y_test = load_dataset()
y_test_labels = np.argmax(y_test, axis = 1)

X_train = torch.tensor(X_train, dtype = torch.float32, device = device)
X_test = torch.tensor(X_test, dtype = torch.float32, device = device)
y_train = torch.tensor(y_train, dtype = torch.float32, device = device)
y_test = torch.tensor(y_test, dtype = torch.float32, device = device)

gamma_list  = np.round(np.logspace(-6, 0, num = 50), 8).tolist()
lambda_list = np.round(np.logspace(-6, 0, num = 50), 8).tolist()

results = list()
for gamma in gamma_list:
    for Lambda in lambda_list:

        win_scores_stack = MKFi(X_train, X_test, y_train, y_test, gamma, Lambda)

        y_test_pred_sum = win_scores_stack.sum(dim = -1)
        y_test_pred_labels = torch.argmax(y_test_pred_sum, dim = 1).cpu().numpy()

        acc = accuracy_score(y_test_labels, y_test_pred_labels) * 100
        
        results.append({
            'Gamma': gamma,
            'Lambda': Lambda,
            'Accuracy': acc
        })
        print(f'\rAccuracy: {acc:.2f} | Gamma: {gamma:.7f} | Lambda: {Lambda:.7f}', end = '')

data = pd.DataFrame(results, columns = ['Gamma', 'Lambda', 'Accuracy'])
data.to_csv('SignFi-LabHome_SenHyp.csv', index = False)
