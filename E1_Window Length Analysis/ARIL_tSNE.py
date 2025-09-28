import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import re


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

n_win = 32
preds = MKFi(X_train, X_test, y_train, y_test, n_win)
y_test_pred_sum = preds.sum(dim = -1).cpu().numpy()

X_feat = y_test_pred_sum.reshape(y_test_pred_sum.shape[0], -1)

tsne = TSNE(
    n_components = 2,
    perplexity = 15,
    n_iter = 1500,
    random_state = 1
)
X_embedded = tsne.fit_transform(X_feat)

plt.figure(figsize = (6, 6))
scatter = plt.scatter(
    X_embedded[:, 0], X_embedded[:, 1],
    c = y_test_labels,
    cmap = 'tab10',
    alpha = 0.6,
    s = 80
)

ax = plt.gca()
for spine in ax.spines.values():
    spine.set_visible(False)
    
plt.xticks(fontsize = 17, fontname = 'Arial')
plt.yticks(fontsize = 17, fontname = 'Arial')
plt.xticks([])
plt.yticks([])

plt.text(
    0.00, 0.05,
    f'$N = {n_win}$',
    transform = ax.transAxes,
    ha = "left", va = "bottom",
    fontsize = 20, fontname = "Arial"
)
plt.savefig(f'ARIL_tSNE_W{n_win}.pdf', bbox_inches = 'tight')



fig_lg = plt.figure()

label_map = {
    0: "Hand up",
    1: "Hand down",
    2: "Hand left",
    3: "Hand right",
    4: "Hand circle",
    5: "Hand cross"
}

handles, labels = scatter.legend_elements()
labels = [int(re.search(r'\d+', l).group()) for l in labels]
labels = [label_map[int(l)] for l in labels]

legend = fig_lg.legend(
    handles, labels,
    prop = {'family': 'Arial', 'size': 13},
    loc = 'center', 
    markerscale = 1.5,
    ncol = 6,
    frameon = False
)
plt.axis('off')
fig_lg.subplots_adjust(0, 0, 1, 1)

fig_lg.canvas.draw()
bbox = legend.get_window_extent().transformed(fig_lg.dpi_scale_trans.inverted())
fig_lg.set_size_inches(bbox.width, bbox.height)

plt.savefig('ARIL_tSNE_legends.pdf', bbox_inches = 'tight')
