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
import numpy as np

from sklearn.feature_selection import RFECV, RFE, SelectFromModel
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR

from PyBioMed import Pyprotein
from PyBioMed.PyProtein import AAComposition, CTD
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

from main import *

pandas.set_option('display.max_columns', 5)

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
    
def make_pybiomed_data(sequences: list) -> np.ndarray:
    if os.path.exists("PyBioMed.csv"):
        PyBioMed_data = pandas.read_csv("PyBioMed.csv", index_col=0)
    else:
        PyBioMed_data = pandas.DataFrame()
    X = None 
    for seq in tqdm.tqdm(sequences):
        if seq == "XRD":
            continue
        if seq in PyBioMed_data.index:
            if X is None:
                X = np.ndarray((0, PyBioMed_data.shape[1]))
            X = np.vstack((X, PyBioMed_data.loc[seq]))
            continue
        peptide_data = {}
        peptide_class = Pyprotein.PyProtein(seq)
        
        # AA, dipeptide, tripeptide compotiion: "GetAAComp", "GetDPComp", "GetTPComp"
        for function in ["GetAPAAC", "GetCTD", "GetGearyAuto", "GetMoranAuto", "GetMoreauBrotoAuto", "GetPAAC", "GetQSO", "GetSOCN"]: #dir(peptide_class):
            _data = getattr(peptide_class, function)()
            peptide_data.update(_data)
            #print(function, list(_data.keys())[:10])
        if PyBioMed_data.shape[0] == 0:
            PyBioMed_data = pandas.DataFrame(columns=list(peptide_data.keys()))
        PyBioMed_data.loc[seq] = np.array(list(peptide_data.values()))
        if X is None:
            X = np.ndarray((0, PyBioMed_data.shape[1]))
        X = np.vstack((X, np.array(list(peptide_data.values()), dtype=np.float64)))
    # Cache them so we arent regenerating on each cycle
    PyBioMed_data.to_csv("PyBioMed.csv")
    print("PyBioMed_data.shape:", PyBioMed_data.shape)
    # Scale and drop nan
# =============================================================================
#     c = X - X.min(axis=0)
#     c = c / ((X.max(axis=0) - X.min(axis=0))/2.0)
#     c = c - 1
#     X = c[:,~np.isnan(c).any(axis=0)]
#     cols = PyBioMed_data.columns[~np.isnan(c).any(axis=0)]
# =============================================================================
    cols = PyBioMed_data.columns
    X = pandas.DataFrame(X, columns=cols)
    return X

def gen_model(X_data: np.ndarray, Y_data: np.ndarray):
    X_train, X_test, y_train, y_test = train_test_split(X_data.astype(np.float32), Y_data, test_size=0.2, shuffle=True, random_state=95)
    #feature_reducer = RFE(Ridge(), n_features_to_select=50, step=10, n_jobs=10)
    #feature_reducer = RFECV(Ridge(), min_features_to_select=40, step=10, cv=3, scoring="neg_mean_squared_error", n_jobs=10)
    feature_reducer = SelectFromModel(Ridge(), max_features=40)
    feature_reducer = feature_reducer.fit(X_train, y_train)
    try:
        support = feature_reducer.support_
    except AttributeError:
        support = feature_reducer.get_support()
    X_train = X_train[X_train.columns[support]]
    X_test = X_test[X_test.columns[support]]
    #X_train = X_train[:,support]
    #X_test = X_test[:,support]
    print("Remaining parameters:", support.sum())
    SVRrbf_param_grid = {
            "kernel": ["rbf", "poly"],
            #"kernel": ["rbf"],
            "degree": [0,1,2,3,4,5],
            "gamma": ["scale", "auto"],
            "C": np.hstack((np.arange(0.1,1.1, 0.1), np.arange(1, 20, 1))), 
            "epsilon": np.linspace(0.01, 5, 20), 
            "max_iter": [-1],
            "tol": [1.0, 0.1, 0.01, 0.001, 0.0001], 
            "verbose":[0]}
    model = SVR()
    HPO_model = RandomizedSearchCV(estimator = model, param_distributions = SVRrbf_param_grid, 
                                   cv = 3, verbose = True, n_iter=300, random_state=947, n_jobs=10)
    HPO_model.fit(X_train, y_train)
    print("\nBest params from grid search:")
    print(HPO_model.best_params_)
    SVMrbf_hyperparameters = HPO_model.best_params_
    return SVR(**SVMrbf_hyperparameters), X_train, X_test, y_train, y_test
    
def extract_y_data(Data: dict) -> pandas.DataFrame:
    Y = pandas.DataFrame()
    for i, seq in enumerate(training_data_labels):
        for code in ["7Z0X", "6M0J"]:
            if seq in Data[code]:
                Y.at[seq, code] = Data[code][seq]["BindingEnergy"]
                Y.at[seq, "i"] = i
    Y["i"] = Y["i"].astype(np.int64)
    Y = Y.dropna()
    Y = Y.reindex([x for x in Y.index if len(x) == 70])
    return Y

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Active machine learning
    for active_learning_iteration in range(1):
        nML = len([x for x in Data["7Z0X"] if Data["7Z0X"][x]["Source"] == "ML"])
        Data = load_data_from_api()
        for code in ["7Z0X", "6M0J"]:
            for seq in copy.copy(list(Data[code].keys())):
                if Data[code][seq] is None:
                    print(seq)
                    del Data[code][seq]
    
        training_data_labels = np.unique(list(Data["7Z0X"].keys())+list(Data["6M0J"].keys()))
        X = make_pybiomed_data(training_data_labels)
        Y = extract_y_data(Data)
        X = X.iloc[Y["i"].values]
        # Remove parametres with low variance
        sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
        sel.fit(X)
        X = X[X.columns[sel.get_support()]]
        print(X)
        
        # Load the RCSB minimized structures
        Complex_7Z0X = measure_interface("7Z0X")
        Complex_6M0J = measure_interface("6M0J")
        for Complex in [Complex_6M0J, Complex_7Z0X]:
            Complex.spike_interface_resids = " ".join([str(x) for x in Data["interface_resid"]])
            Complex.load_universe()    
            Complex.FindInterface()
            Complex.BuildInterface()
        
        # Generate a bunch of potential new sequences to test and select the best on using ML
        candidates = []
        while len(candidates) < 1000:
            #print(Complex_6M0J.interface_seq in Data["6M0J"])
            while Complex_6M0J.interface_seq in Data["6M0J"] or Complex_6M0J.interface_seq in candidates:
                #print(Complex_6M0J.interface_seq, "found in Data, mutating")
                if len(candidates) % 10 == 0:
                    Complex_6M0J.reset_seq()
                Complex_6M0J.Mutate()
            print(Complex_6M0J.interface_seq, "<- Mutated interface sequence")
            candidates.append(Complex_6M0J.interface_seq)
        
        models = {}
        title = "ML {nML}"
        for target in ["7Z0X", "6M0J"]:
            models[target], X_train, X_test, y_train, y_test = gen_model(X, Y[target].values.flatten())
            models[target].fit(X_train, y_train)
            y_train_pred = models[target].predict(X_train)
            y_test_pred = models[target].predict(X_test)
            r2 = r2_score(y_test, y_test_pred)
            print(target, "Test r2:", r2)
            print(target, "Train r2:", r2_score(y_train, y_train_pred))
            plt.scatter(y_train, y_train_pred, label=f"{target} Train")
            plt.scatter(y_test, y_test_pred, label=f"{target} Test")
            plt.xlabel("Measured")
            plt.ylabel("Pred")
            title = title + f" - {target} Test r2 {r2:.2f}"
        plt.legend()
        
        # Use the ML algorithm to predict the binding energies of a list of random mutated candidates 
        X = make_pybiomed_data(candidates)
        X = X[X_train.columns]
        print(X.shape)
        candidates_pred = pandas.DataFrame(index=candidates)
        for target in ["7Z0X", "6M0J"]:
            candidates_pred[target] = models[target].predict(X)
        #6M0J (increase), 7Z0X (decrease)
        candidates_pred["score"] = candidates_pred["6M0J"] + -(candidates_pred["7Z0X"]*3) # put more importance on loss of binding to 7Z0X
        candidates_pred = candidates_pred.sort_values("score")
        print(candidates_pred)
        
        # Run the choice
        choice = candidates_pred.iloc[0].name
        Complex_7Z0X.interface_seq = choice
        Complex_6M0J.interface_seq = choice
        for Complex, idx in zip([Complex_7Z0X, Complex_6M0J], ["7Z0X", "6M0J"]):
            Complex.MakeMutation()
            print("Running:", f"{Complex.active_folder}/Minimization")
            Complex.Minimize()
            if not os.path.exists(f"{Complex.active_folder}/Minimization.coor"):
                print("Minimization failed, skipping")
                continue
            Complex.load_universe()
            Complex.FindInterface() # If we have already set self.spike_interface_resids then use this
            Complex.BuildInterface()
            #Make measurements and store them
            if Complex.interface_seq not in Data[idx]:
                Complex.MeasureInterface()
                Data = load_data_from_api() # always reload before posting incase there is new data from another source
                entry = {"Source": "ML"}
                entry.update(Complex.score)
                post_entry_to_api(Data, idx, Complex.interface_seq, entry)
                Data = load_data_from_api() # always reload after posting incase there is new data from another source
            else:
                print("Already have:", idx, Complex.interface_seq)
                
                
            
            
            
# =============================================================================
#     batch_size = 10
#     mse_sum = torch.nn.MSELoss(reduction='sum')
#     model = Module(Min=Y.min(), Max=Y.max(), input_size=X_train.shape[1])
#     model.to(device)
#     loss_function = nn.MSELoss()
#     #loss_function = r2_score
#     SGD = torch.optim.SGD(model.parameters(), lr=0.002)
#     SGD_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(SGD, factor=0.2, patience=100, threshold=0, 
#                                                                min_lr = 0.000001, verbose=True)
#     
#     X_train = torch.from_numpy(X_train).to(device)
#     y_train = torch.from_numpy(y_train).float().to(device)
#     X_test = torch.from_numpy(X_test).to(device)
#     y_test = torch.from_numpy(y_test).to(device)
#     
#     train_dataset = TensorDataset(X_train, y_train)
#     train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
#     
#     test_dataset = TensorDataset(X_test, y_test)
#     test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#     
#     for epoch in range(SGD_scheduler.last_epoch, SGD_scheduler.last_epoch+500):
#         LossSum = None
#         model.train()
#         total_mse = 0
#         count = 0
#         for X_train, y_train in tqdm.tqdm(train_dataloader):
#             SGD.zero_grad()
#             pred = model(X_train)
#             pred = pred.flatten()
#             loss = torch.sqrt(loss_function(pred, y_train))
#             
#             total_mse += mse_sum(pred, y_train).item()
#             count += pred.size(0)
#             
#             loss.backward()
#             SGD.step()
#             
#         loss = np.sqrt(total_mse / count)
#         SGD_scheduler.step(loss)
#         
#     
#         model.eval()
#         for X_train, y_train in tqdm.tqdm(train_dataloader):
#             predict = model(X_train).reshape(-1)
#             pred_y = predict.cpu().detach().numpy().flatten()
#             plt.scatter(y_train.cpu().detach().numpy().flatten(), pred_y, s=11, color="blue", alpha=0.8)
#             
#         total_mse = 0
#         count = 0
#         true_all = np.ndarray((0,))
#         pred_all = np.ndarray((0,))
#         for X_test, y_test in tqdm.tqdm(test_dataloader):
#             predict = model(X_test).reshape(-1)
#             pred_y = predict.cpu().detach().numpy().flatten()
#             pred_all = np.hstack((pred_all, pred_y))
#             true_all = np.hstack((true_all, y_test.cpu().detach().numpy().flatten()))
#             plt.scatter(y_test.cpu().detach().numpy().flatten(), pred_y, s=10, color="orange", alpha=0.9)
#             total_mse += mse_sum(predict, y_test).item()
#             count += predict.size(0)
#         test_loss = np.sqrt(total_mse / count)
#         plt.plot([Y.min(),Y.max()], [Y.min(),Y.max()], lw=1, color="black")
#         r2 = r2_score(pred_all, true_all)
#         plt.title(f"Test RMSE: {round(test_loss, 2)}, r2: {round(r2, 1)}, EPOCH: {epoch}")
#         plt.show()
#     
# =============================================================================
    
    
    