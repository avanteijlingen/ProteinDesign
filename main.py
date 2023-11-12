# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 20:05:38 2023

@author: Alex
"""
import MDAnalysis as mda
import urllib, os, tqdm, subprocess, sys, shutil, copy, re
from ase import Atoms
import numpy as np
from sklearn.metrics import euclidean_distances
import matplotlib.pyplot as plt
import torch, pandas
import tad_dftd4 as d4
from pathlib import Path
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA
#import torchani

from AIMNet2.calculators.aimnet2ase import AIMNet2Calculator

# Local libraries
import contactarea

# =============================================================================
# https://www.ks.uiuc.edu/Training/Tutorials/namd/FEP/tutorial-FEP.pdf
# page 20 Mutation of tyrosine into alanine
# =============================================================================


vmd_dir = "C:/Program Files (x86)/University of Illinois/VMD/plugins/noarch/tcl/readcharmmtop1.2/"
psfgen = "C:/Users/Alex/Documents/NAMD_2.14_Win64-multicore-CUDA/psfgen.exe"
namd = "C:/Users/Alex/Documents/NAMD_2.14_Win64-multicore-CUDA/namd2.exe"

Angstrom2Bohr = 1.88973
eV2kcalmol    = 23.0609

# Since this program is non-generic we will define everything we need
Chains = {"6M0J": {"A": "ACE2",
                   "E": "Spike"
                  }, # THSC20.HVTR26 Fab bound to SARS-CoV-2 Receptor Binding Domain
          "7Z0X": {"H": "Antibody heavy chain",
                   "L": "Antibody light chain",
                   "R": "Spike",
                   } #Crystal structure of SARS-CoV-2 spike receptor-binding domain bound with ACE2
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
        if not os.path.exists(f"MD/{self.code}/{self.code}.pdb"):
            urllib.request.urlretrieve(f"http://files.rcsb.org/download/{self.code}.pdb", f"MD/{self.code}/{self.code}.pdb") 
    
    def psf(self):
        if os.path.exists(f"MD/{self.code}/{self.code}_psfgen.psf") and 1==0:
            return None
        U = mda.Universe(f"MD/{self.code}/{self.code}.pdb")
        protein = U.select_atoms("protein")
        pdbs = []
        for chainID in Chains[self.code]:
            chain = protein.select_atoms(f"chainID {chainID}")
            pdb_file = f"MD/{idx}/{idx}_{chainID}.pdb"
            # We must rename His to Hse for our forcefield
            chain.residues.resnames = [x.replace("HIS", "HSE") for x in chain.residues.resnames]
            chain.write(pdb_file)
            pdbs.append(pdb_file)
        writepgn(f"MD/{idx}/{idx}.pgn", pdbs, Chains[idx], f"MD/{self.code}/{self.code}_psfgen")
        # psfgen adds hydrogens to the structure and creates a topology
        x = subprocess.check_output([psfgen, f"MD/{self.code}/{self.code}.pgn"])
        self.fix_psf(f"MD/{self.code}/{self.code}_psfgen.psf")
    
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
        if not os.path.exists(f"MD/{idx}/Minimize.log"):
            shutil.copy("Minimize.namd", f"MD/{idx}/Minimize.namd")
            from_template(f"MD/{idx}/Minimize.namd", ["INPUT", "PARAM_DIR"], [f"{idx}_psfgen", Path("parameters").absolute().as_posix()])
            #subprocess.check_output([namd, "+p4", f"MD/{idx}/Minimize.namd", ">", f"MD/{idx}/Minimize.log"])
            os.system(" ".join([namd, "+p4", f"MD/{idx}/Minimize.namd", ">", f"MD/{idx}/Minimize.log"]))
        # =============================================================================
        #     ps = subprocess.Popen([namd, "+p4", f"MD/{idx}/Minimize.namd", ">", f"MD/{idx}/Minimize.log"], stdout=subprocess.PIPE)
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
        # First determine which residues are part of the interface
        interface_cutoff = 10.0 # Angstrom, We need to trim these proteins down to just those at the interface to fit the dispersion calculation in memory
        d = euclidean_distances(self.Spike.positions, self.Receptor.positions)
        Spike_interface = self.Spike[d.min(axis=1) < interface_cutoff]
        Receptor_interface = self.Receptor[d.min(axis=0) < interface_cutoff]

        # Rebuild the broken residues (some atoms dont pass the cutoff) 
        spike_resids = " ".join(np.unique(Spike_interface.resids).astype(np.str_))
        self.Spike_interface = self.Spike.select_atoms(f"resid {spike_resids}")
        Receptor_resids = " ".join(np.unique(Receptor_interface.resids).astype(np.str_))
        self.Receptor_interface = self.Receptor.select_atoms(f"resid {Receptor_resids}")
        mda.Merge(self.Spike_interface, self.Receptor_interface).select_atoms("all").write(f"MD/{self.code}/Interface.pdb")
        
        self.MeasureHBonds()
        
        elements = [x[0] for x in self.Spike_interface.names]
        self.Spike_ase = Atoms(elements, self.Spike_interface.positions)
        elements = [x[0] for x in self.Receptor_interface.names]
        self.Receptor_ase = Atoms(elements, self.Receptor_interface.positions)
        
        self.MeasureBindingEnergy()

        self.score["contact surface area"] = self.surface_contact.calculate(self.Spike_ase, self.Receptor_ase)
        
    def load_universe(self):
        self.U = mda.Universe(f"MD/{self.code}/{self.code}_psfgen.psf", f"MD/{self.code}/Minimization.coor")
        self.U.trajectory[-1]
        self.Spike = self.U.select_atoms(f"segid {self.spike_chainID}")
        self.Receptor = self.U.select_atoms(f"segid {self.receptor_chainID}")
        self.resnames = self.Spike.residues.resnames
    
    def __init__(self, code):
        self.code = code
        #Initialize
        os.makedirs(f"MD/{self.code}", exist_ok=True)
        self.download_pdb()
        self.psf()
        
        self.spike_chainID = ""
        self.receptor_chainID = ""
        for chainID in Chains[code]:
            if Chains[code][chainID].upper() == "SPIKE":
                self.spike_chainID = self.spike_chainID + chainID + " "
            else:
                self.receptor_chainID = self.receptor_chainID + chainID + " "
# =============================================================================
#         print("receptor_chainID:", self.receptor_chainID)
#         print("spike_chainID:", self.spike_chainID)
# =============================================================================
        
        # Load AIMNet2 model for energy calculations
        self.AIMNet2_model_file = "AIMNet2/models/aimnet2_wb97m-d3_ens.jpt"
        self.AIMNet2_model = torch.jit.load(self.AIMNet2_model_file, map_location=device)
        self.AIMNet2_calc = AIMNet2Calculator(self.AIMNet2_model)
        
        #surface contact calculator
        self.surface_contact = contactarea.contactarea(radii_csv = "contactarea/Alvarez2013_vdwradii.csv")
        
        self.score = {}


if __name__ == "__main__":
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    
    # For AIMNet backend
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    
    #idx = "7Z0X"
    idx = "6M0J"
    inter = measure_interface(idx)
    inter.Minimize()
    inter.load_universe()
    inter.MeasureInterface()
    
    print(inter.score)
    

    





# =============================================================================
# 
# inter.Mutate()
# 
# etc etc 
# 
# predict, measure, predict active laerning optimizzation
# 
# =============================================================================


# =============================================================================
# # Convert atom labels to their atomic numbers and pad with zero's
# numbers = d4.utils.pack((
#     d4.utils.to_number([x[0] for x in Spike_interface.types]),
#     d4.utils.to_number([x[0] for x in Receptor_interface.types]),
# ))
# 
# 
# # coordinates in Bohr, MDAnalysis uses Angstrom natively so we need to scale
# positions = d4.utils.pack((
#     torch.from_numpy(Spike_interface.positions * Angstrom2Bohr),
#     torch.from_numpy(Receptor_interface.positions * Angstrom2Bohr),
# ))
# 
# # total charge of both system
# charge = torch.tensor([0.0, 0.0])
# 
# # TPSS0-D4-ATM parameters
# param = {
#     "s6": positions.new_tensor(1.0),
#     "s8": positions.new_tensor(1.85897750),
#     "s9": positions.new_tensor(1.0),
#     "a1": positions.new_tensor(0.44286966),
#     "a2": positions.new_tensor(4.60230534),
# }
# 
# # calculate dispersion energy in Hartree
# energy = torch.sum(d4.dftd4(numbers, positions, charge, param), -1)
# torch.set_printoptions(precision=10)
# print(energy)
# # tensor([-0.0088341432, -0.0027013607])
# print(energy[0] - 2*energy[1])
# # tensor(-0.0034314217)
# =============================================================================
