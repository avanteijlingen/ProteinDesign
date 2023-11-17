# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 20:05:38 2023

@author: Alex
"""
import MDAnalysis as mda
import urllib, os, subprocess, sys, shutil, copy, re, time, json, pandas, requests
from ase import Atoms
import numpy as np
from sklearn.metrics import euclidean_distances, r2_score
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from AIMNet2.calculators.aimnet2ase import AIMNet2Calculator
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA

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

vmd_dir = os.environ["VMDTOP"]
psfgen = os.environ["PSFGEN"]
namd = os.environ["NAMD"]


Angstrom2Bohr = 1.88973
eV2kcalmol    = 23.0609

# Since this program is non-generic we will define everything we need
Chains = {"6M0J": {"A": "ACE2",
                   "E": "Spike"
                  }, # Prot_A-Prot_C, Crystal structure of SARS-CoV-2 spike receptor-binding domain bound with ACE2, we want to increase binding with this one
          "7Z0X": {"H": "Antibody heavy chain",
                   "L": "Antibody light chain",
                   "R": "Spike",
                   } # Prot_A-Prot_B, HSC20.HVTR26 Fab bound to SARS-CoV-2 Receptor Binding Domain, we want to decrease binding with this one
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
        if not os.path.exists(f"{self.active_folder}/{self.idx}.pdb"):
            urllib.request.urlretrieve(f"http://files.rcsb.org/download/{self.idx}.pdb", f"{self.active_folder}/{self.idx}.pdb") 
    
    def psf(self):
        if os.path.exists(f"{self.active_folder}/{self.idx}_psfgen.psf"):
            self.fix_psf(f"{self.active_folder}/{self.idx}_psfgen.psf") # temp, once off fix
            return None
        U = mda.Universe(f"{self.active_folder}/{self.idx}.pdb")
        protein = U.select_atoms("protein")
        pdbs = []
        for chainID in Chains[self.idx]:
            chain = protein.select_atoms(f"chainID {chainID}")
            pdb_file = f"{self.active_folder}/{self.idx}_{chainID}.pdb"
            # We must rename His to Hse for our forcefield
            chain.residues.resnames = [x.replace("HIS", "HSE") for x in chain.residues.resnames]
            chain.write(pdb_file)
            pdbs.append(pdb_file)
        writepgn(f"{self.active_folder}/{self.idx}.pgn", pdbs, Chains[self.idx], f"{self.active_folder}/{self.idx}_psfgen")
        # psfgen adds hydrogens to the structure and creates a topology
        subprocess.check_output([psfgen, f"{self.active_folder}/{self.idx}.pgn"])
        self.fix_psf(f"{self.active_folder}/{self.idx}_psfgen.psf")
    
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

    def Minimize(self, min_input = "Minimize.namd"):
        # Run a short minimization of the system
        if not os.path.exists(f"{self.active_folder}/Minimization.coor"):
            shutil.copy(min_input, f"{self.active_folder}/Minimize.namd")
            from_template(f"{self.active_folder}/Minimize.namd", ["INPUT", "PARAM_DIR"], [f"{self.idx}_psfgen", Path("parameters").absolute().as_posix()])
            #subprocess.check_output([namd, "+p4", f"{self.active_folder}/Minimize.namd", ">", f"{self.active_folder}/Minimize.log"])
            os.system(" ".join([namd, "+p10", f"{self.active_folder}/Minimize.namd", ">", f"{self.active_folder}/Minimize.log"]))
        # =============================================================================
        #     ps = subprocess.Popen([namd, "+p4", f"{self.active_folder}/Minimize.namd", ">", f"{self.active_folder}/Minimize.log"], stdout=subprocess.PIPE)
        #     ps.wait()
        # =============================================================================

    def MeasureHBonds(self):
        self.hbonds = np.ndarray((0, 4), dtype=np.int64)
        self.hbonds_detailed = pandas.DataFrame()
        hbonds_calc = HBA(universe=self.U)
        i = 0
        for acceptor, donor in zip([self.receptor_chainID, self.spike_chainID], [self.spike_chainID, self.receptor_chainID]):
            hbonds_calc.hydrogens_sel = hbonds_calc.guess_hydrogens(f"protein")
            hbonds_calc.hydrogens_sel = f"protein and segid {donor} and (" + hbonds_calc.hydrogens_sel + ")"
            hbonds_calc.acceptors_sel = hbonds_calc.guess_acceptors(f"protein")
            hbonds_calc.acceptors_sel = f"protein and segid {acceptor} and (" + hbonds_calc.acceptors_sel + ")"
            hbonds_calc.run()
            #Each row of the array contains the: donor atom id, hydrogen atom id, acceptor atom id and the total number of times the hydrogen bond was observed. The array is sorted by frequency of occurrence.
            counts = hbonds_calc.count_by_ids()
            for row in counts:
                self.hbonds_detailed.at[i, "donor id"] = row[0]
                self.hbonds_detailed.at[i, "hydrogen id"] = row[1]
                self.hbonds_detailed.at[i, "acceptor id"] = row[2]
                if donor == self.spike_chainID:
                    self.hbonds_detailed.at[i, "donor"] = "Spike"
                    self.hbonds_detailed.at[i, "donor resname"]    = self.Spike_interface.select_atoms(f"id {row[0]}").resnames
                    self.hbonds_detailed.at[i, "acceptor resname"] = self.Receptor_interface.select_atoms(f"id {row[2]}").resnames
                else:
                    self.hbonds_detailed.at[i, "donor"] = "not Spike"
                    self.hbonds_detailed.at[i, "donor resname"]    = self.Receptor_interface.select_atoms(f"id {row[0]}").resnames
                    self.hbonds_detailed.at[i, "acceptor resname"] = self.Spike_interface.select_atoms(f"id {row[2]}").resnames
                    
                i+=1
            self.hbonds = np.vstack((self.hbonds, counts))
        self.hbonds_detailed["donor id"] = self.hbonds_detailed["donor id"].astype(np.int64)
        self.hbonds_detailed["hydrogen id"] = self.hbonds_detailed["hydrogen id"].astype(np.int64)
        self.hbonds_detailed["acceptor id"] = self.hbonds_detailed["acceptor id"].astype(np.int64)
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
        Receptor_interface_partial = self.Receptor[d.min(axis=0) < self.interface_cutoff]
        self.Receptor_resids = " ".join(np.unique(Receptor_interface_partial.resids).astype(np.str_))

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
        self.U = mda.Universe(f"{self.active_folder}/{self.idx}_psfgen.psf", f"{self.active_folder}/{coord_file}")
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
        self.active_folder = f"MD/{self.idx}_{self.interface_seq}"
        os.makedirs(self.active_folder, exist_ok=True)
        self.Spike_interface.residues.resnames = pu.translate1to3(self.interface_seq).split("-") # This propogates all the way up the universe
        pdbs = [f"{self.active_folder}/{self.idx}_{self.spike_chainID}.pdb"]
        segids = [self.spike_chainID]
        self.Spike.write(f"{self.active_folder}/{self.idx}_{self.spike_chainID}.pdb")
        for chainID in self.receptor_chainID.split():
            self.Receptor.select_atoms(f"segid {chainID}").write(f"{self.active_folder}/{self.idx}_{chainID}.pdb")
            pdbs.append(f"{self.active_folder}/{self.idx}_{chainID}.pdb")
            segids.append(chainID)
        writepgn(f"{self.active_folder}/{self.idx}.pgn", 
                 pdbs, 
                 segids, 
                 f"{self.active_folder}/{self.idx}_psfgen")
        x = subprocess.check_output([psfgen, f"{self.active_folder}/{self.idx}.pgn"])
        assert os.path.exists(f"{self.active_folder}/{self.idx}_psfgen.psf"), f"Couldnt make: {self.active_folder}/{self.idx}_psfgen.psf"
        self.fix_psf(f"{self.active_folder}/{self.idx}_psfgen.psf")
        
    def reset_seq(self):
        self.interface_seq = copy.copy(self.original_seq)
        
    def __init__(self, idx, device):
        self.idx = idx
        self.active_folder = f"MD/{self.idx}"
        #Initialize
        os.makedirs(f"{self.active_folder}", exist_ok=True)
        self.download_pdb()
        self.psf()
        
        self.original_seq = None
        
        self.spike_chainID = ""
        self.receptor_chainID = ""
        for chainID in Chains[idx]:
            if Chains[idx][chainID].upper() == "SPIKE":
                self.spike_chainID = self.spike_chainID + chainID + " "
            else:
                self.receptor_chainID = self.receptor_chainID + chainID + " "
        self.spike_chainID = self.spike_chainID.strip() # Removing trailing ' '
        self.receptor_chainID = self.receptor_chainID.strip()
# =============================================================================
#         print("receptor_chainID:", self.receptor_chainID)
#         print("spike_chainID:", self.spike_chainID)
# =============================================================================
        self.interface_cutoff = 10.0 # Angstrom, We need to trim these proteins down to just those at the interface to fit the dispersion calculation in memory
        
        # Load AIMNet2 model for energy calculations
        self.AIMNet2_model_file = "AIMNet2/models/aimnet2_wb97m-d3_ens.jpt"
        self.AIMNet2_model = torch.jit.load(self.AIMNet2_model_file, map_location=device)
        self.AIMNet2_calc = AIMNet2Calculator(self.AIMNet2_model)
        
        #surface contact calculator
        self.surface_contact = contactarea.contactarea(radii_csv = "contactarea/Alvarez2013_vdwradii.csv")
        
        self.score = {}


# =============================================================================
# def load_data_from_api(host="http://localhost:5000"):
#     api_url = f"{host}/ProteinDesign"    
#     response = requests.get(api_url)
#     return response.json()
# 
# def post_entry_to_api(idx, seq, entry_data, host="http://localhost:5000"):
#     api_url = f"{host}/ProteinDesign/{idx}/{seq}"
#     print(api_url)
#     post_data = {"Source": "API_test",
#                 "hbonds": 6,
#                 "BindingEnergy": -32.35931305264907,
#                 "contact surface area": 31.93952531149623
#                 }
#     response = requests.post(api_url, json=entry_data)
#     #print(response)
#     #print(response.json())
#     print(response.status_idx)
# =============================================================================
## Have to patch this due to lack of networking
def load_data_from_api():
    if os.path.exists("Data.json"):
        with open("Data.json") as jin:
            Data = json.load(jin)
    else:
        Data = {"7Z0X": {}, "6M0J": {}}
    return Data

def post_entry_to_api(Data: dict, idx: str, seq: str, entry_data: dict):
    Data[idx][seq] = entry_data
    shutil.copy("Data.json", "Data.json.bak")
    with open("Data.json", 'w') as jout: jout.write(json.dumps(Data, indent=4))


if __name__ == "__main__":
    np.random.seed(435634)
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    
    # For AIMNet backend
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    
    Data = load_data_from_api()

    Complex_7Z0X = measure_interface("7Z0X", device)
    Complex_6M0J = measure_interface("6M0J", device)
    
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
        Data = load_data_from_api()
        with open("Data.json", 'w') as jout: jout.write(json.dumps(Data, indent=4))#, separators=(',', ':')))
        Data = load_data_from_api()
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
            
            Data = load_data_from_api() # always reload before posting incase there is new data from another source
            entry = {"Source": "ML"}
            entry.update(Complex.score)
            post_entry_to_api(Data, idx, Complex.interface_seq, entry)
            Data = load_data_from_api() # always reload after posting incase there is new data from another source
            
        Complex.Minimize()
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
            print("Already have data for:", idx, Complex.interface_seq)
            
    sys.exit()
    # Randomly mutate residues and record the interface
    while len(Data["7Z0X"]) < 1000:
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
                Data = load_data_from_api() # always reload before posting incase there is new data from another source
                entry = {"Source": "ML"}
                entry.update(Complex.score)
                post_entry_to_api(Data, idx, Complex.interface_seq, entry)
                Data = load_data_from_api() # always reload after posting incase there is new data from another source
            else:
                print("Already have data for:", idx, Complex.interface_seq)
    
    for key in ["7Z0X", "6M0J"]:
        print(key, len(Data[key]), "data points")

