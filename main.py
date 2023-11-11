# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 20:05:38 2023

@author: Alex
"""
import MDAnalysis as mda
import urllib, os, tqdm, subprocess
import numpy as np

vmd_dir = "C:/Program Files (x86)/University of Illinois/VMD/plugins/noarch/tcl/readcharmmtop1.2/"
psfgen = "C:/Users/Alex/Documents/NAMD_2.14_Win64-multicore-CUDA/psfgen.exe"
namd = "C:/Users/Alex/Documents/NAMD_2.14_Win64-multicore-CUDA/namd2.exe"


# Since this program is non-generic we will define everything we need
Chains = {"6M0J": {"A": "ACE2",
                   "E": "Spike"}, # THSC20.HVTR26 Fab bound to SARS-CoV-2 Receptor Binding Domain
          "7Z0X": {"H": "Antibody heavy chain",
                   "L": "Antibody light chain",
                   "R": "Spike",
                   } #Crystal structure of SARS-CoV-2 spike receptor-binding domain bound with ACE2
         }

for idx in Chains:
    os.makedirs(f"MD/{idx}", exist_ok=True)
    if os.path.exists(f"MD/{idx}/Complex.psf"):
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
    pgn = open(f"MD/{idx}/patch.pgn",'w')
    pgn.write("package require psfgen")
    pgn.write("\n")
    pgn.write(f"topology \"{vmd_dir}/top_all22_prot.rtf\"")
    pgn.write("\n")
    pgn.write(f"topology \"{vmd_dir}/toppar_water_ions_namd.str\"")
    pgn.write("\n")
    
    for chainID in Chains[idx]:
        chain = protein.select_atoms(f"chainID {chainID}")

        pdb_file = f"MD/{idx}/{idx}_{chainID}.pdb"
        # We must rename His to Hse for our forcefield
        chain.residues.resnames = [x.replace("HIS", "HSE") for x in chain.residues.resnames]
        chain.write(pdb_file)

        pgn.write(f"segment U{chainID} " + "{"+f"pdb MD/{idx}/{idx}_{chainID}.pdb"+"}")
        pgn.write("\n")
        pgn.write(f"coordpdb MD/{idx}/{idx}_{chainID}.pdb U{chainID} ")
        pgn.write("\n")
        pgn.write("guesscoord\n")
        pgn.write(f"writepsf MD/{idx}/Complex.psf\n")
        pgn.write(f"writepdb MD/{idx}/Complex.pdb\n")
        
    pgn.write("exit\n")
    pgn.close()
    x = str(subprocess.check_output([psfgen, f"MD/{idx}/patch.pgn"]))

idx = "6M0J"
Spike = mda.Universe(f"MD/{idx}/{idx}_E.pdb")
ACE2 = mda.Universe(f"MD/{idx}/{idx}_A.pdb")
