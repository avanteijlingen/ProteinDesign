#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 01:08:32 2023

@author: rkb19187
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from main import *

class Module(nn.Module):
    def __init__(self, Min, Max, input_size):
        super(Module, self).__init__()
        self.sigmoid = nn.Sigmoid()

        self.fc = nn.Sequential(
            nn.Linear(input_size, 50, bias=False),
            nn.ReLU(),
            nn.Linear(50, 5, bias=False),
            nn.Linear(5, 1, bias=False)
        )
                
        self.max = Max
        self.min = Min

    def forward(self, X):
        
        m = self.fc(X)
        return self.sigmoid(m)*(self.max-self.min)+self.min
    
with open("Data.json") as jin:
    Data = json.load(jin)
    
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Active machine learning
    print("PyBioMed generating parameters")
    if os.path.exists("PyBioMed.csv"):
        PyBioMed_data = pandas.read_csv("PyBioMed.csv", index_col=0)
    else:
        PyBioMed_data = pandas.DataFrame()
    X = np.ndarray((0, 9880))
    Y = np.ndarray((0, ))
    for seq in tqdm.tqdm(Data["7Z0X"]):
        if seq == "XRD":
            continue
        Y = np.hstack((Y, Data["7Z0X"][seq]["BindingEnergy"]))
        if seq in PyBioMed_data.index:
            X = np.vstack((X, PyBioMed_data.loc[seq]))
            continue
        peptide_data = {}
        peptide_class = Pyprotein.PyProtein(seq)
        
        for function in ["GetAAComp", "GetALL", "GetAPAAC", "GetCTD", "GetDPComp", "GetGearyAuto", "GetMoranAuto", "GetMoreauBrotoAuto", "GetPAAC", "GetQSO", "GetSOCN", "GetTPComp", "GetTriad"]: #dir(peptide_class):
            peptide_data.update(getattr(peptide_class, function)())
        if PyBioMed_data.shape[0] == 0:
            PyBioMed_data = pandas.DataFrame(columns=list(peptide_data.values()))
        PyBioMed_data.loc[seq] = np.array(list(peptide_data.values()))
        X = np.vstack((X, np.array(list(peptide_data.values()), dtype=np.float64)))
    # Cache them so we arent regenerating on each cycle
    PyBioMed_data.to_csv("PyBioMed.csv")

    # Scale and drop nan
    c = X - X.min(axis=0)
    c = c / ((X.max(axis=0) - X.min(axis=0))/2.0)
    c = c - 1
    X = c[:,~np.isnan(c).any(axis=0)]
    
    X_train, X_test, y_train, y_test = train_test_split(X.astype(np.float32), Y, test_size=0.2, shuffle=True, random_state=95)

    support = RFE(Ridge(), n_features_to_select=50, step=10)
    #support = RFE(LogisticRegression(), n_features_to_select=50, step=20)
    support = support.fit(X_train, y_train)
    X_train = X_train[:,support.support_]
    X_test = X_test[:,support.support_]
    print("Remaining parameters:", support.support_.sum())
    
    
    
    batch_size = 10
    mse_sum = torch.nn.MSELoss(reduction='sum')
    model = Module(Min=Y.min(), Max=Y.max(), input_size=X_train.shape[1])
    model.to(device)
    loss_function = nn.MSELoss()
    #loss_function = r2_score
    SGD = torch.optim.SGD(model.parameters(), lr=0.002)
    SGD_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(SGD, factor=0.2, patience=100, threshold=0, 
                                                               min_lr = 0.000001, verbose=True)
    
    SVRrbf_param_grid = {
            #"kernel": ["rbf", "poly"],
            "kernel": ["rbf"],
            #"degree": [0,1,2,3,4,5],
            "gamma": ["scale", "auto"],
            "C": np.hstack((np.arange(0.1,1.1, 0.1), np.arange(1, 20, 1))), 
            "epsilon": np.linspace(0.01, 5, 20), 
            "max_iter": [-1],
            "tol": [1.0, 0.1, 0.01, 0.001, 0.0001], 
            "verbose":[0]}
    model = SVR()
    HPO_model = RandomizedSearchCV(estimator = model, param_distributions = SVRrbf_param_grid, 
                                   cv = 3, n_jobs = 16, verbose = True, n_iter=100, random_state=947)
    HPO_model.fit(X_train, y_train)
    print("\nBest params from grid search:")
    print(HPO_model.best_params_)
    SVMrbf_hyperparameters = HPO_model.best_params_
    model = SVR(**SVMrbf_hyperparameters)
    
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_test_pred)
    print("Test r2:", r2)
    print("Train r2:", r2_score(y_train, y_train_pred))
    plt.scatter(y_train, y_train_pred)
    plt.scatter(y_test, y_test_pred)
    plt.ylabel("Pred")
    plt.ylabel("Measured")


    
    sys.exit()
    
    X_train = torch.from_numpy(X_train).to(device)
    y_train = torch.from_numpy(y_train).float().to(device)
    X_test = torch.from_numpy(X_test).to(device)
    y_test = torch.from_numpy(y_test).to(device)
    
    train_dataset = TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    test_dataset = TensorDataset(X_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    for epoch in range(SGD_scheduler.last_epoch, SGD_scheduler.last_epoch+500):
        LossSum = None
        model.train()
        total_mse = 0
        count = 0
        for X_train, y_train in tqdm.tqdm(train_dataloader):
            SGD.zero_grad()
            pred = model(X_train)
            pred = pred.flatten()
            loss = torch.sqrt(loss_function(pred, y_train))
            
            total_mse += mse_sum(pred, y_train).item()
            count += pred.size(0)
            
            loss.backward()
            SGD.step()
            
        loss = np.sqrt(total_mse / count)
        SGD_scheduler.step(loss)
        
    
        model.eval()
        for X_train, y_train in tqdm.tqdm(train_dataloader):
            predict = model(X_train).reshape(-1)
            pred_y = predict.cpu().detach().numpy().flatten()
            plt.scatter(y_train.cpu().detach().numpy().flatten(), pred_y, s=11, color="blue", alpha=0.8)
            
        total_mse = 0
        count = 0
        true_all = np.ndarray((0,))
        pred_all = np.ndarray((0,))
        for X_test, y_test in tqdm.tqdm(test_dataloader):
            predict = model(X_test).reshape(-1)
            pred_y = predict.cpu().detach().numpy().flatten()
            pred_all = np.hstack((pred_all, pred_y))
            true_all = np.hstack((true_all, y_test.cpu().detach().numpy().flatten()))
            plt.scatter(y_test.cpu().detach().numpy().flatten(), pred_y, s=10, color="orange", alpha=0.9)
            total_mse += mse_sum(predict, y_test).item()
            count += predict.size(0)
        test_loss = math.sqrt(total_mse / count)
        plt.plot([Y.min(),Y.max()], [Y.min(),Y.max()], lw=1, color="black")
        r2 = r2_score(pred_all, true_all)
        plt.title(f"Test RMSE: {round(test_loss, 2)}, r2: {round(r2, 1)}, EPOCH: {epoch}")
        plt.show()
    
    
    
    