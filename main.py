# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 20:05:38 2023

@author: Alex
"""
import MDAnalysis as mda
import urllib, os, tqdm, subprocess, sys, shutil, copy, re, time, json, pandas
from ase import Atoms
import numpy as np
from sklearn.metrics import euclidean_distances, r2_score
import matplotlib.pyplot as plt
import torch
from pathlib import Path

from AIMNet2.calculators.aimnet2ase import AIMNet2Calculator
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA

from rdkit import Chem
from PyBioMed.PyMolecule import topology
from PyBioMed import Pyprotein
from PyBioMed.PyProtein import AAComposition, CTD
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv

# Environmental variables check
load_dotenv()
for var in ["NAMD", "VMDTOP", "PSFGEN"]:
    assert var in list(os.environ.keys()), f"Couldnt find environmental variable: {var}"

# Local libraries
import contactarea
import peptideutils as pu 

# =============================================================================
# https://www.ks.uiuc.edu/Training/Tutorials/namd/FEP/tutorial-FEP.pdf
# page 20 Mutation of tyrosine into alanine
# =============================================================================

# Should be in a dotenv
#vmd_dir = "C:/Program Files (x86)/University of Illinois/VMD/plugins/noarch/tcl/readcharmmtop1.2/"
#psfgen = "C:/Users/Alex/Documents/NAMD_2.14_Win64-multicore-CUDA/psfgen.exe"
#namd = "C:/Users/Alex/Documents/NAMD_2.14_Win64-multicore-CUDA/namd2.exe"
vmd_dir = os.environ["VMDTOP"]
psfgen = os.environ["PSFGEN"]
namd = os.environ["NAMD"]


Angstrom2Bohr = 1.88973
eV2kcalmol    = 23.0609

# Since this program is non-generic we will define everything we need
Chains = {"6M0J": {"A": "ACE2",
                   "E": "Spike"
                  }, # Prot_A-Prot_C, THSC20.HVTR26 Fab bound to SARS-CoV-2 Receptor Binding Domain
          "7Z0X": {"H": "Antibody heavy chain",
                   "L": "Antibody light chain",
                   "R": "Spike",
                   } # Prot_A-Prot_B, Crystal structure of SARS-CoV-2 spike receptor-binding domain bound with ACE2
         }

def writepgn(fname, pdbs, segids, outfile):
    pgn = open(fname,'w')
    pgn.write("package require psfgen")
    pgn.write("\n")
    pgn.write(f"topology \"{vmd_dir}/top_all36_prot.rtf\"")
    pgn.write("\n")
    pgn.write(f"topology \"{vmd_dir}/toppar_water_ions_namd.str\"")
    pgn.write("\n")
    for pdb, segid in zip(pdbs, segids):
        pgn.write(f"segment {segid} " + "{"+f"pdb {pdb}"+"}")
        pgn.write("\n")
        pgn.write(f"coordpdb {pdb} {segid} ")
        pgn.write("\n")
    pgn.write("guesscoord\n")
    pgn.write(f"writepsf {outfile}.psf\n")
    pgn.write(f"writepdb {outfile}.pdb\n")
    pgn.write("exit\n")
    pgn.close()
    
def readin(fpath: str) -> str:
	f = open(fpath)
	content = f.read()
	f.close()
	return content

def from_template(fname: str, original: list, new: list) -> None:
    """
    Replace every instance of original[n] with new[n]
    """
    assert len(original) == len(new)
    content = readin(fname)
    for i in range(len(original)):
        content = content.replace(original[i], new[i])
    with open(fname, 'w') as outfile:
        outfile.write(content)

class measure_interface:
    def download_pdb(self):
        if not os.path.exists(f"{self.active_folder}/{self.code}.pdb"):
            urllib.request.urlretrieve(f"http://files.rcsb.org/download/{self.code}.pdb", f"{self.active_folder}/{self.code}.pdb") 
    
    def psf(self):
        if os.path.exists(f"{self.active_folder}/{self.code}_psfgen.psf"):
            self.fix_psf(f"{self.active_folder}/{self.code}_psfgen.psf") # temp, once off fix
            return None
        U = mda.Universe(f"{self.active_folder}/{self.code}.pdb")
        protein = U.select_atoms("protein")
        pdbs = []
        for chainID in Chains[self.code]:
            chain = protein.select_atoms(f"chainID {chainID}")
            pdb_file = f"{self.active_folder}/{self.code}_{chainID}.pdb"
            # We must rename His to Hse for our forcefield
            chain.residues.resnames = [x.replace("HIS", "HSE") for x in chain.residues.resnames]
            chain.write(pdb_file)
            pdbs.append(pdb_file)
        writepgn(f"{self.active_folder}/{self.code}.pgn", pdbs, Chains[self.code], f"{self.active_folder}/{self.code}_psfgen")
        # psfgen adds hydrogens to the structure and creates a topology
        subprocess.check_output([psfgen, f"{self.active_folder}/{self.code}.pgn"])
        self.fix_psf(f"{self.active_folder}/{self.code}_psfgen.psf")
    
    def fix_psf(self, psf_file):
        # psfgen is adding numbers to some resids in the psf for 7Z0X chainID L and H, so we will just convert these (82A -> 82)
        # For this program this will not be a problem, at most it could increase compute time
        with open(psf_file) as input_psf:
            psf_content = input_psf.read()
        with open(psf_file, 'w') as output_psf:
            psf_content = psf_content.split("!NATOM")
            output_psf.write(psf_content[0])
            output_psf.write("!NATOM")
            for line in psf_content[1].split("\n"):
                if len(line.split()) > 8 and re.search('[a-zA-Z]', line) is not None: # The lines that contain the resids contain letters and more than 8 parts
                    resid = line.split()[2]
                    if re.search('[a-zA-Z]', resid) is not None:
                        line = line.replace(resid, re.sub("[a-zA-Z]"," ",resid))
                output_psf.write(line)
                output_psf.write("\n")

    def Minimize(self):
        # Run a short minimization of the system
        if not os.path.exists(f"{self.active_folder}/Minimization.coor"):
            shutil.copy("Minimize.namd", f"{self.active_folder}/Minimize.namd")
            from_template(f"{self.active_folder}/Minimize.namd", ["INPUT", "PARAM_DIR"], [f"{self.code}_psfgen", Path("parameters").absolute().as_posix()])
            #subprocess.check_output([namd, "+p4", f"{self.active_folder}/Minimize.namd", ">", f"{self.active_folder}/Minimize.log"])
            os.system(" ".join([namd, "+p10", f"{self.active_folder}/Minimize.namd", ">", f"{self.active_folder}/Minimize.log"]))
        # =============================================================================
        #     ps = subprocess.Popen([namd, "+p4", f"{self.active_folder}/Minimize.namd", ">", f"{self.active_folder}/Minimize.log"], stdout=subprocess.PIPE)
        #     ps.wait()
        # =============================================================================

    def MeasureHBonds(self):
        self.hbonds = np.ndarray((0, 4), dtype=np.int64)
        hbonds_calc = HBA(universe=self.U)
        for acceptor, donor in zip([self.receptor_chainID, self.spike_chainID], [self.spike_chainID, self.receptor_chainID]):
            hbonds_calc.hydrogens_sel = hbonds_calc.guess_hydrogens(f"protein")
            hbonds_calc.hydrogens_sel = f"protein and segid {donor} and (" + hbonds_calc.hydrogens_sel + ")"
            hbonds_calc.acceptors_sel = hbonds_calc.guess_acceptors(f"protein")
            hbonds_calc.acceptors_sel = f"protein and segid {acceptor} and (" + hbonds_calc.acceptors_sel + ")"
            hbonds_calc.run()
            #Each row of the array contains the: donor atom id, hydrogen atom id, acceptor atom id and the total number of times the hydrogen bond was observed. The array is sorted by frequency of occurrence.
            counts = hbonds_calc.count_by_ids()
            self.hbonds = np.vstack((self.hbonds, counts))
        self.hbonds_calc = hbonds_calc
        self.score["hbonds"] = self.hbonds.shape[0]
        
    def MeasureHydrophobicInteractions(self):
        pass
    
    def MeasureBindingEnergy(self):               
        self.Spike_ase.calc = self.AIMNet2_calc
        self.Receptor_ase.calc = self.AIMNet2_calc
        Complex = self.Receptor_ase+self.Spike_ase
        Complex.calc = self.AIMNet2_calc
        
        # This could be improved by further optimization within the DNN PES and by using 
        # This could also be further improved through correcting for potential BSSE that may be captured by the DNN but that requires more optimization steps 
        self.AIMNet2_calc.do_reset()
        self.AIMNet2_calc.set_charge(0) # guess since its large
        Complex_E = Complex.get_potential_energy()
        self.AIMNet2_calc.do_reset()
        self.AIMNet2_calc.set_charge(0) # guess since its large
        Receptor_E = self.Receptor_ase.get_potential_energy()
        self.AIMNet2_calc.do_reset()
        self.AIMNet2_calc.set_charge(0) # guess since its large
        Spike_E = self.Spike_ase.get_potential_energy()
        BindingEnergy = (Complex_E - (Receptor_E + Spike_E)) * eV2kcalmol
        self.score["BindingEnergy"] = BindingEnergy
        
    def MeasureInterface(self):       
        self.MeasureHBonds()
        self.MeasureBindingEnergy()
        self.score["contact surface area"] = self.surface_contact.calculate(self.Spike_ase, self.Receptor_ase)
    
    def FindInitialInterface(self):
        # First determine which residues are part of the interface
        d = euclidean_distances(self.Spike.positions, self.Receptor.positions)
        Spike_interface = self.Spike[d.min(axis=1) < self.interface_cutoff]
        # Rebuild the broken residues (some atoms dont pass the cutoff) 
        return np.unique(Spike_interface.resids)
        
    def FindInterface(self):
        # Use this if we have already determined which resids are part of the interface
        d = euclidean_distances(self.Spike.select_atoms(f"resid {self.spike_interface_resids}").positions, self.Receptor.positions)
        Receptor_interface = self.Receptor[d.min(axis=0) < self.interface_cutoff]
        self.Receptor_resids = " ".join(np.unique(Receptor_interface.resids).astype(np.str_))

    def BuildInterface(self):
        self.Spike_interface = self.Spike.select_atoms(f"resid {self.spike_interface_resids}")
        self.Receptor_interface = self.Receptor.select_atoms(f"resid {self.Receptor_resids}")
        mda.Merge(self.Spike_interface, self.Receptor_interface).select_atoms("all").write(f"{self.active_folder}/Interface.pdb")
        elements = [x[0] for x in self.Spike_interface.names]
        self.Spike_ase = Atoms(elements, self.Spike_interface.positions)
        elements = [x[0] for x in self.Receptor_interface.names]
        self.Receptor_ase = Atoms(elements, self.Receptor_interface.positions)
        self.interface_seq = pu.translate3to1("-".join(self.Spike_interface.residues.resnames))
        if self.original_seq is None:
            self.original_seq = copy.copy(self.interface_seq)
            
    def load_universe(self, coord_file = "Minimization.coor"):
        self.U = mda.Universe(f"{self.active_folder}/{self.code}_psfgen.psf", f"{self.active_folder}/{coord_file}")
        self.U.trajectory[-1]
        self.Spike = self.U.select_atoms(f"segid {self.spike_chainID}")
        self.Receptor = self.U.select_atoms(f"segid {self.receptor_chainID}")
    
    def Mutate(self):
        sel = np.random.choice(np.arange(len(self.interface_seq)))
        self.interface_seq = list(self.interface_seq)
        new_res = self.interface_seq[sel]
        while new_res == self.interface_seq[sel]:
            self.interface_seq[sel] = np.random.choice(pu.peptideutils_letters1)
        print("Mutated:", sel, "resid:", self.spike_interface_resids.split()[sel])
        self.interface_seq = "".join(self.interface_seq)
        
    def MakeMutation(self):
        self.active_folder = f"MD/{self.code}_{self.interface_seq}"
        os.makedirs(self.active_folder, exist_ok=True)
        self.Spike_interface.residues.resnames = pu.translate1to3(self.interface_seq).split("-") # This propogates all the way up the universe
        pdbs = [f"{self.active_folder}/{self.code}_{self.spike_chainID}.pdb"]
        segids = [self.spike_chainID]
        self.Spike.write(f"{self.active_folder}/{self.code}_{self.spike_chainID}.pdb")
        for chainID in self.receptor_chainID.split():
            self.Receptor.select_atoms(f"segid {chainID}").write(f"{self.active_folder}/{self.code}_{chainID}.pdb")
            pdbs.append(f"{self.active_folder}/{self.code}_{chainID}.pdb")
            segids.append(chainID)
        

        writepgn(f"{self.active_folder}/{self.code}.pgn", 
                 pdbs, 
                 segids, 
                 f"{self.active_folder}/{self.code}_psfgen")
        
        x = subprocess.check_output([psfgen, f"{self.active_folder}/{self.code}.pgn"])
        assert os.path.exists(f"{self.active_folder}/{self.code}_psfgen.psf"), f"Couldnt make: {self.active_folder}/{self.code}_psfgen.psf"
        self.fix_psf(f"{self.active_folder}/{self.code}_psfgen.psf")
        
    def reset_seq(self):
        self.interface_seq = copy.copy(self.original_seq)
        
    def __init__(self, code):
        self.code = code
        self.active_folder = f"MD/{self.code}"
        #Initialize
        os.makedirs(f"{self.active_folder}", exist_ok=True)
        self.download_pdb()
        self.psf()
        
        self.original_seq = None
        
        self.spike_chainID = ""
        self.receptor_chainID = ""
        for chainID in Chains[code]:
            if Chains[code][chainID].upper() == "SPIKE":
                self.spike_chainID = self.spike_chainID + chainID + " "
            else:
                self.receptor_chainID = self.receptor_chainID + chainID + " "
        self.spike_chainID = self.spike_chainID.strip() # Removing trailing ' '
        self.receptor_chainID = self.receptor_chainID.strip()
# =============================================================================
#         print("receptor_chainID:", self.receptor_chainID)
#         print("spike_chainID:", self.spike_chainID)
# =============================================================================
        self.spike_interface_resids = '350 402 403 404 405 406 408 409 416 417 418 419 420 421 422 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508'
        self.interface_cutoff = 10.0 # Angstrom, We need to trim these proteins down to just those at the interface to fit the dispersion calculation in memory
        
        # Load AIMNet2 model for energy calculations
        self.AIMNet2_model_file = "AIMNet2/models/aimnet2_wb97m-d3_ens.jpt"
        self.AIMNet2_model = torch.jit.load(self.AIMNet2_model_file, map_location=device)
        self.AIMNet2_calc = AIMNet2Calculator(self.AIMNet2_model)
        
        #surface contact calculator
        self.surface_contact = contactarea.contactarea(radii_csv = "contactarea/Alvarez2013_vdwradii.csv")
        
        self.score = {}

def makepeptide(peptide):
    """
    
    INFO       Chain termini will be charged
    
    """
    mol = Chem.MolFromSequence(peptide)
    mol.GetAtomWithIdx(0).SetFormalCharge(1)
    mol.GetAtomWithIdx(len(list(mol.GetAtoms()))-1).SetFormalCharge(-1)
    return mol


if __name__ == "__main__":
    np.random.seed(435634)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    
    # For AIMNet backend
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    
    if os.path.exists("Data.json"):
        with open("Data.json") as jin:
            Data = json.load(jin)
    else:
        Data = {"7Z0X": {}, "6M0J": {}}

    
    Complex_7Z0X = measure_interface("7Z0X")
    Complex_6M0J = measure_interface("6M0J")
    
    # First determine which residues are part of the interface
    # We want to get them from both complexes and stack them together since they mostly but not entirely overlap
    if "interface_resid" not in Data:
        spike_resids = np.ndarray((0,), dtype=np.int64)
        for Complex, idx in zip([Complex_7Z0X, Complex_6M0J], ["7Z0X", "6M0J"]):
            Complex.Minimize()
            Complex.load_universe()
            
            spike_resids = np.hstack((spike_resids, Complex.FindInitialInterface()))
            print(spike_resids)
        spike_resids = np.unique(spike_resids)
        print(spike_resids)
        Complex_7Z0X.spike_interface_resids = " ".join(spike_resids.astype(np.str_))
        Complex_6M0J.spike_interface_resids = " ".join(spike_resids.astype(np.str_))
        Data["interface_resid"] = [int(x) for x in spike_resids]
        with open("Data.json", 'w') as jout: jout.write(json.dumps(Data, indent=4))#, separators=(',', ':')))
    else:
        Complex_7Z0X.spike_interface_resids = " ".join([str(x) for x in Data["interface_resid"]])
        Complex_6M0J.spike_interface_resids = " ".join([str(x) for x in Data["interface_resid"]])

    # Run the initial minimization and make measurements
    for Complex, idx in zip([Complex_7Z0X, Complex_6M0J], ["7Z0X", "6M0J"]):
        # Get the XRD structure data
        if "XRD" not in Data[idx]:
            print("Getting pure XRD measurements")
            Complex.load_universe(coord_file = f"{idx}_psfgen.pdb")
            Complex.FindInterface() # If we have already set self.spike_interface_resids then use this
            Complex.BuildInterface()
            Complex.MeasureInterface()
            Data[idx]["XRD"] = {"Source": "XRD"}
            for key in Complex.score.keys():
                Data[idx]["XRD"][key] = Complex.score[key]
            with open("Data.json", 'w') as jout: jout.write(json.dumps(Data, indent=4))
            
        Complex.Minimize()
        Complex.load_universe()
        
        Complex.FindInterface() # If we have already set self.spike_interface_resids then use this
        Complex.BuildInterface()
        #Make measurements and store them
        if Complex.interface_seq not in Data[idx]:
            Complex.MeasureInterface()
            Data[idx][Complex.interface_seq] = {"Source": "Initial"}
            for key in Complex.score.keys():
                Data[idx][Complex.interface_seq][key] = Complex.score[key]
            with open("Data.json", 'w') as jout: jout.write(json.dumps(Data, indent=4))
        else:
            print("Already have data for:", idx, Complex.interface_seq)
            

    # Randomly mutate residues and record the interface
    while len(Data["7Z0X"]) < 1:
        print(f"Random iteration: ", len(Data["7Z0X"])-2)
        for Complex, idx in zip([Complex_7Z0X, Complex_6M0J], ["7Z0X", "6M0J"]):
            while Complex_6M0J.interface_seq in Data[idx]:
                print(Complex_6M0J.interface_seq, "found in Data, mutating")
                if len(Data["7Z0X"]) % 10 == 0:
                    Complex_6M0J.reset_seq()
                Complex_6M0J.Mutate()
            print(Complex_6M0J.interface_seq, "<- Mutated interface sequence")
            Complex_7Z0X.interface_seq = Complex_6M0J.interface_seq
            
            Complex.MakeMutation()
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
                Data[idx][Complex.interface_seq] = {"Source": "Random"}
                for key in Complex.score.keys():
                    Data[idx][Complex.interface_seq][key] = Complex.score[key]
                with open("Data.json", 'w') as jout: jout.write(json.dumps(Data, indent=4))
            else:
                print("Already have data for:", idx, Complex.interface_seq)
    
    for key in ["7Z0X", "6M0J"]:
        print(key, len(Data[key]), "data points")

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
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=195)
    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
    from sklearn.feature_selection import RFE
    from sklearn.svm import SVR
    from sklearn.linear_model import RANSACRegressor, LogisticRegression, LassoLars, ElasticNet, LinearRegression
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.neural_network import MLPRegressor
    support = RFE(Ridge(), n_features_to_select=10, step=10)
    #support = RFE(LogisticRegression(), n_features_to_select=50, step=20)
    support = support.fit(X_train, y_train)
    
    X_train = X_train[:,support.support_]
    X_test = X_test[:,support.support_]
    print("Remaining parameters:", support.support_.sum())
    
    # SVR
    SVRrbf_param_grid = {
            "kernel": ["rbf", "poly"],
            "degree": [0,1,2,3,4,5],
            "gamma": ["scale", "auto"],
            "C": np.hstack((np.arange(0.1,1.1, 0.1), np.arange(1, 20, 1))), 
            "epsilon": np.linspace(0.01, 5, 20), 
            "max_iter": [-1],
            "tol": [1.0, 0.1, 0.01, 0.001, 0.0001], 
            "verbose":[0]}
    model = SVR()
    HPO_model = RandomizedSearchCV(estimator = model, param_distributions = SVRrbf_param_grid, 
                                   cv = 3, n_jobs = 16, verbose = True, n_iter=200, random_state=947)
    HPO_model.fit(X_train, y_train)
    print("\nBest params from grid search:")
    print(HPO_model.best_params_)
    SVMrbf_hyperparameters = HPO_model.best_params_
    model = SVR(**SVMrbf_hyperparameters)
    
# =============================================================================
#     RF_param_grid = {'bootstrap': [True, False],
#                       'criterion': ['squared_error', 'absolute_error'],
#                       'max_depth': [1,2,3,4,5,None],
#                       'max_features': ["sqrt", "log2", None],
#                       'max_leaf_nodes': [None],
#                       'min_impurity_decrease': [0.0],
#                       'min_samples_leaf': [1, 2],
#                       'min_samples_split': [0.5, 1.0],
#                       'min_weight_fraction_leaf': [0.0, 0.01, 0.1],
#                       'n_estimators': [10, 100],
#                       'n_jobs': [4],
#                       'oob_score': [False],
#                       'verbose': [False],
#                       'warm_start': [False, True],
#                       "random_state":[4]}    
#     model = ExtraTreesRegressor()
#     HPO_model = RandomizedSearchCV(estimator = model, param_distributions = RF_param_grid, 
#                                    cv = 3, n_jobs = 16, verbose = True, n_iter=200, random_state=947)
#     HPO_model.fit(X_train, y_train)
#     print("\nBest params from grid search:")
#     print(HPO_model.best_params_)
#     RF_hyperparameters = HPO_model.best_params_
#     model = ExtraTreesRegressor(**HPO_model.best_params_)
# 
# =============================================================================

    #model = SVR(epsilon=0.005, kernel="rbf", C=10.0)
    #model = LassoLars(alpha=0.1)
    #model = Ridge()
    model = RANSACRegressor(Ridge(), min_samples=5)
    #model = ExtraTreesRegressor(max_depth=5)
    #model = KNeighborsRegressor(n_neighbors=5,  weights='distance', algorithm='auto')
    #model = RandomForestRegressor()
    #model = MLPRegressor()
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



