# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 20:05:38 2023

@author: Alex
"""
import MDAnalysis as mda
import urllib, os, tqdm, subprocess, sys, shutil
import numpy as np
from sklearn.metrics import euclidean_distances
import matplotlib.pyplot as plt
import torch
import tad_dftd4 as d4
from pathlib import Path
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA

# =============================================================================
# https://www.ks.uiuc.edu/Training/Tutorials/namd/FEP/tutorial-FEP.pdf
# page 20 Mutation of tyrosine into alanine
# =============================================================================


vmd_dir = "C:/Program Files (x86)/University of Illinois/VMD/plugins/noarch/tcl/readcharmmtop1.2/"
psfgen = "C:/Users/Alex/Documents/NAMD_2.14_Win64-multicore-CUDA/psfgen.exe"
namd = "C:/Users/Alex/Documents/NAMD_2.14_Win64-multicore-CUDA/namd2.exe"
Angstrom2Bohr = 1.88973

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


for idx in Chains:
    os.makedirs(f"MD/{idx}", exist_ok=True)
    if os.path.exists(f"MD/{idx}/{idx}_psfgen.psf"):
        continue
    if not os.path.exists(f"MD/{idx}/{idx}.pdb"):
        urllib.request.urlretrieve(f"http://files.rcsb.org/download/{idx}.pdb", f"MD/{idx}/{idx}.pdb") 
    
    
    U = mda.Universe(f"MD/{idx}/{idx}.pdb")
    All = U.select_atoms("all")
    protein = U.select_atoms("protein")
    
# =============================================================================
#     chainIDs = np.unique(All.chainIDs)
#     print("chainIDs:", chainIDs)
# =============================================================================
    
    pdbs = []
    for chainID in Chains[idx]:
        chain = protein.select_atoms(f"chainID {chainID}")

        pdb_file = f"MD/{idx}/{idx}_{chainID}.pdb"
        # We must rename His to Hse for our forcefield
        chain.residues.resnames = [x.replace("HIS", "HSE") for x in chain.residues.resnames]
        chain.write(pdb_file)
        
        pdbs.append(pdb_file)
# =============================================================================
#         writepgn(f"MD/{idx}/{idx}_{chainID}.pgn", [pdb_file], f"MD/{idx}/{idx}_{chainID}_psfgen")
#         x = str(subprocess.check_output([psfgen, f"MD/{idx}/{idx}_{chainID}.pgn"]))
# =============================================================================
        
    writepgn(f"MD/{idx}/{idx}.pgn", pdbs, Chains[idx], f"MD/{idx}/{idx}_psfgen")
    # psfgen adds hydrogens to the structure and creates a topology
    x = str(subprocess.check_output([psfgen, f"MD/{idx}/{idx}.pgn"]))

idx = "6M0J"
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

U = mda.Universe(f"MD/{idx}/{idx}_psfgen.psf", f"MD/{idx}/Minimization.dcd")
U.trajectory[-1]
Spike = U.select_atoms("segid E")
ACE2 = U.select_atoms("segid A")


hbonds = HBA(universe=U)
hbonds.hydrogens_sel = hbonds.guess_hydrogens("protein and segid E")
hbonds.acceptors_sel = hbonds.guess_acceptors("protein and segid A")
hbonds.run()

#Each row of the array contains the: donor atom id, hydrogen atom id, acceptor atom id and the total number of times the hydrogen bond was observed. The array is sorted by frequency of occurrence.
counts = hbonds.count_by_ids()
print(counts.shape)

sys.exit()


# We need to trim these proteins down to just those at the interface to fit the dispersion calculation in memory
interface_cutoff = 6.0 # Angstrom
d = euclidean_distances(Spike.positions, ACE2.positions)
Spike_interface = Spike[d.min(axis=1) < interface_cutoff]
ACE2_interface = ACE2[d.min(axis=0) < interface_cutoff]
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