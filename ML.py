#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 01:08:32 2023

@author: rkb19187
"""
import torch, os
import numpy as np
import tqdm, pandas
from main import *
from Transformer import *

pandas.set_option('display.max_columns', 5)
   

if __name__ == "__main__":
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
        
    Data = load_data_from_api()
    
    # Load the RCSB minimized structures
    Complex_7Z0X = measure_interface("7Z0X", torch.device("cpu")) # struggling to hold all these models in GPU memory so we'll use CUDA in some place and not in other
    Complex_6M0J = measure_interface("6M0J", torch.device("cpu"))
    for Complex in [Complex_6M0J, Complex_7Z0X]:
        Complex.spike_interface_resids = " ".join([str(x) for x in Data["interface_resid"]])
    
    # Check for missing pairs we can fill in
    missing = []
    for seq in Data["7Z0X"]:
        if "ML" not in Data["7Z0X"][seq]["Source"] or len(seq) != 70:
            continue
        if seq not in Data["6M0J"]:
            missing.append(seq)

    if len(missing) > 0:
        print(f"Found {len(missing)} sequences in 6M0J that we have for 7Z0X")
        for seq in missing:
            if os.path.exists(f"MD/6M0J_{seq}/Minimization.restart.coor"):
                print(seq, "already tried (and failed)")
                continue
            print("Filling in for:", seq)
            Complex_6M0J.active_folder = f"MD/{Complex_6M0J.idx}"
            Complex_6M0J.load_universe()    
            Complex_6M0J.FindInterface()
            Complex_6M0J.BuildInterface()
            Complex_6M0J.interface_seq = seq
            
            Complex_6M0J.MakeMutation()
            print(f"Running minimization with {namd}")
            Complex_6M0J.Minimize()
            if not os.path.exists(f"{Complex_6M0J.active_folder}/Minimization.coor"):
                print("Minimization failed, skipping")
                continue
            Complex_6M0J.load_universe()
            Complex_6M0J.FindInterface() # If we have already set self.spike_interface_resids then use this
            Complex_6M0J.BuildInterface()
            print("Making measurements")
            Complex_6M0J.MeasureInterface()
            Data = load_data_from_api() # always reload before posting incase there is new data from another source
            entry = {"Source": Data["7Z0X"][seq]["Source"]}
            entry.update(Complex_6M0J.score)
            post_entry_to_api(Data, "6M0J", Complex_6M0J.interface_seq, entry)
            Data = load_data_from_api() # always reload after posting incase there is new data from another source


    # Active machine learning
    for active_learning_iteration in range(1000):
        # Reset the active folder to the wild type
        for Complex in [Complex_6M0J, Complex_7Z0X]:
            Complex.active_folder = f"MD/{Complex.idx}"
            Complex.load_universe()    
            Complex.FindInterface()
            Complex.BuildInterface()

        # Load database again, do it very time incase multiple instance are runnign
        Data = load_data_from_api()
        for code in ["7Z0X", "6M0J"]:
            for seq in copy.copy(list(Data[code].keys())):
                if Data[code][seq] is None:
                    print(seq)
                    del Data[code][seq]
        #nML = len([x for x in Data["7Z0X"] if "ML" in Data["7Z0X"][x]["Source"]])

        models = {}
        for idx in ["7Z0X", "6M0J"]:
            train_dataloader, test_dataloader, Min_val, Max_val = encode_data(Data, idx, device)
            models[idx], _ = train_transformer_model(device, train_dataloader, test_dataloader, Min_val, Max_val, checkpoint_fname=f"best_{idx}.pt")


        # Generate a bunch of potential new sequences to test and select the best on using ML
        candidates = []
        while len(candidates) < 100:
            while Complex_6M0J.interface_seq in Data["6M0J"] or Complex_6M0J.interface_seq in candidates:
                #print(Complex_6M0J.interface_seq, "found in Data, mutating")
                if len(candidates) % 10 == 0:
                    Complex_6M0J.reset_seq()
                Complex_6M0J.Mutate()
            print(Complex_6M0J.interface_seq, "<- Mutated interface sequence")
            candidates.append(Complex_6M0J.interface_seq)       
        X_test = make_data(candidates, src_len).squeeze().to(device)
        

        # Make prediction of binding energy with transformer
        print("Predicting binding energies for candidate sequences")
        candidates_pred = pandas.DataFrame(index=candidates)
        for idx in ["7Z0X", "6M0J"]:
            candidates_pred[idx] = models[idx](X_test).flatten().cpu().detach().numpy()
        #6M0J (increase), 7Z0X (decrease)
        candidates_pred["score"] = candidates_pred["6M0J"] + -(candidates_pred["7Z0X"]*2) # put more importance on loss of binding to 7Z0X
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
                print("Measuring the resulting interface")
                Complex.MeasureInterface()
                Data = load_data_from_api() # always reload before posting incase there is new data from another source
                entry = {"Source": "ML-DNN"}
                entry.update(Complex.score)
                post_entry_to_api(Data, idx, Complex.interface_seq, entry)
                Data = load_data_from_api() # always reload after posting incase there is new data from another source
            else:
                print("Already have:", idx, Complex.interface_seq)
                
                
            
            
            
