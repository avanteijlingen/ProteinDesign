# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 20:05:38 2023

@author: Alex
"""
import MDAnalysis as mda
import urllib, os, tqdm, subprocess, sys, shutil, copy
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
        if os.path.exists(f"MD/{self.code}/{self.code}_psfgen.psf"):
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
    
    def Measure(self):
        interface_cutoff = 6.0 # Angstrom, We need to trim these proteins down to just those at the interface to fit the dispersion calculation in memory
        d = euclidean_distances(self.Spike.positions, self.ACE2.positions)
        Spike_interface = self.Spike[d.min(axis=1) < interface_cutoff]
        ACE2_interface = self.ACE2[d.min(axis=0) < interface_cutoff]

        # Rebuild the broken (some atoms dont pass the cutoff) 
        spike_resids = " ".join(np.unique(Spike_interface.resids).astype(np.str_))
        self.Spike_interface = self.Spike.select_atoms(f"resid {spike_resids}")
        
        ace2_resids = " ".join(np.unique(ACE2_interface.resids).astype(np.str_))
        self.ACE2_interface = self.ACE2.select_atoms(f"resid {ace2_resids}")
        #mda.Merge(Spike_interface, ACE2_interface).select_atoms("all").write(f"MD/{idx}/Interface.pdb")
        
        hbonds = np.ndarray((0, 4), dtype=np.int64)
        hbonds_calc = HBA(universe=self.U)
        for acceptor in ["E" , "A"]:
            donor = "A" if acceptor == "E" else "E"
            hbonds_calc.hydrogens_sel = hbonds_calc.guess_hydrogens(f"protein")
            hbonds_calc.hydrogens_sel = f"protein and segid {donor} and (" + hbonds_calc.hydrogens_sel + ")"
            hbonds_calc.acceptors_sel = hbonds_calc.guess_acceptors(f"protein")
            hbonds_calc.acceptors_sel = f"protein and segid {acceptor} and (" + hbonds_calc.acceptors_sel + ")"
            hbonds_calc.run()
            #Each row of the array contains the: donor atom id, hydrogen atom id, acceptor atom id and the total number of times the hydrogen bond was observed. The array is sorted by frequency of occurrence.
            counts = hbonds_calc.count_by_ids()
            print(counts)
            print(donor, acceptor, counts.shape)
            hbonds = np.vstack((hbonds, counts))
        
        self.hbonds = hbonds
        self.hbonds_calc = hbonds_calc
        self.score["hbonds"] = hbonds.shape[0]
    
    def load_universe(self):
        self.U = mda.Universe(f"MD/{self.code}/{self.code}_psfgen.psf", f"MD/{self.code}/Minimization.coor")
        self.U.trajectory[-1]
        self.Spike = self.U.select_atoms("segid E")
        self.ACE2 = self.U.select_atoms("segid A")
        self.resnames = self.Spike.residues.resnames
    
    def __init__(self, code):
        self.code = code
        #Initialize
        os.makedirs(f"MD/{self.code}", exist_ok=True)
        self.download_pdb()
        self.psf()
        
        self.score = {}


idx = "6M0J"
inter = measure_interface(idx)
inter.Minimize()
inter.load_universe()
inter.Measure()

# For AIMNet backend
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

AIMNet2_model = "AIMNet2/models/aimnet2_wb97m-d3_ens.jpt"
model = torch.jit.load(AIMNet2_model, map_location=device)
calc = AIMNet2Calculator(model)

elements = [x[0] for x in inter.Spike_interface.names]
Spike_ase = Atoms(elements, inter.Spike_interface.positions)

elements = [x[0] for x in inter.ACE2_interface.names]
ACE2_ase = Atoms(elements, inter.ACE2_interface.positions)

calc.do_reset()
calc.set_charge(0) # guess since its large

Spike_ase.calc = calc
ACE2_ase.calc = copy.copy(calc)
Complex = ACE2_ase+Spike_ase
Complex.calc = copy.copy(calc)

# This could be improved by further optimization within the DNN PES and by using 
# This could also be further improved through correcting for potential BSSE that may be captured by the DNN but that requires more optimization steps 
Complex_E = Complex.get_potential_energy()
ACE2_E = ACE2_ase.get_potential_energy()
Spike_E = Spike_ase.get_potential_energy()

BindingEnergy = (Complex_E - (ACE2_E + Spike_E)) * eV2kcalmol

# =============================================================================
# 
# inter.Mutate()
# 
# etc etc 
# 
# predict, measure, predict active laerning optimizzation
# 
# =============================================================================


sys.exit()


mda.Merge(Spike_interface, ACE2_interface).select_atoms("all").write(f"MD/{idx}/Interface.pdb")

# Convert atom labels to their atomic numbers and pad with zero's
numbers = d4.utils.pack((
    d4.utils.to_number([x[0] for x in Spike_interface.types]),
    d4.utils.to_number([x[0] for x in ACE2_interface.types]),
))


# coordinates in Bohr, MDAnalysis uses Angstrom natively so we need to scale
positions = d4.utils.pack((
    torch.from_numpy(Spike_interface.positions * Angstrom2Bohr),
    torch.from_numpy(ACE2_interface.positions * Angstrom2Bohr),
))

# total charge of both system
charge = torch.tensor([0.0, 0.0])

# TPSS0-D4-ATM parameters
param = {
    "s6": positions.new_tensor(1.0),
    "s8": positions.new_tensor(1.85897750),
    "s9": positions.new_tensor(1.0),
    "a1": positions.new_tensor(0.44286966),
    "a2": positions.new_tensor(4.60230534),
}

# calculate dispersion energy in Hartree
energy = torch.sum(d4.dftd4(numbers, positions, charge, param), -1)
torch.set_printoptions(precision=10)
print(energy)
# tensor([-0.0088341432, -0.0027013607])
print(energy[0] - 2*energy[1])
# tensor(-0.0034314217)