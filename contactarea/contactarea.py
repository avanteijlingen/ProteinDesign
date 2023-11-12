# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 00:23:50 2023

@author: Alex
"""

import ase
from ase import Atoms
from ase.io import read
import pandas
import numpy as np
from sklearn.metrics import euclidean_distances
from tqdm import tqdm


class contactarea:
    def Fibb(self, n):
        goldenRatio = (1 + 5**0.5)/2 #\phi = golden ratio
        i = np.arange(0, n)
        theta = 2 *np.pi * i / goldenRatio
        phi = np.arccos(1 - 2*(i+0.5)/n)
        x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
        #print("theta", theta, theta.shape[0])
        #print("phi", phi, phi.shape[0])
        points = np.array((x,y,z)).T
        return points
    
    def generate_sphere_points(self, n):
       """
       Returns list of coordinates on a sphere using the Golden-
       Section Spiral algorithm.
       """
       points = np.ndarray((n, 3))
       inc = np.pi * (3 - np.sqrt(5))
       offset = 2 / float(n)
       for k in range(int(n)):
          y = k * offset - 1 + (offset / 2)
          r = np.sqrt(1 - y*y)
          phi = k * inc
          point = np.array((np.cos(phi)*r, y, np.sin(phi)*r), dtype=np.float64, copy=True)
          points[k] = point
       return points
   
    def find_neighbor_indices(self, atoms, coords, k):
        """
        Returns list of indices of atoms within probe distance to atom k. 
        """
        radius = self.vdw_radii.at[atoms[k], "vdw_radius"]
        neighbor_indices = []
        d = euclidean_distances(coords[k].reshape(1,3), coords)
        for i in range(d.shape[1]):
            if i == k:
                continue
            radius_i = self.vdw_radii.at[atoms[i], "vdw_radius"]
            if d[0][i] < radius + radius_i + self.radius_probe: #+probe twice?
                neighbor_indices.append(i)
        return neighbor_indices
    
    def calculate(self, mol1, mol2, n_sphere_point=24):
        self.inaccessible_points = np.ndarray((0, 3))
        self.sphere_points = self.generate_sphere_points(n_sphere_point)
        self.area_per_point = 4.0 * np.pi / len(self.sphere_points) # Scaled in the loop by the vdw_radii
        self.areas = pandas.DataFrame(columns=["area", "atom", "vdw_radius"])
        
        contacts = pandas.DataFrame()
        
        n_inaccessible_points = 0
        
        for i in tqdm(range(len(mol1))):
            radius_i = self.vdw_radii.at[mol1[i].symbol, "vdw_radius"]
            
            d12 = euclidean_distances(mol1.positions[i].reshape(1,3), mol2.positions).flatten()
            
            # Skip the more expensive point-by-point test if there is not chance any surface contact will be found
            if d12.min() > self.vdw_radii.values.max() * 2:
                continue
            
            local_sphere_points = (self.sphere_points *radius_i) + mol1.positions[i]
            self.point_dists = euclidean_distances(local_sphere_points, mol2.positions)
            
            # Skip the more expensive point-by-point test if there is not chance any surface contact will be found
            for j in np.where(self.point_dists.min(axis=0) < self.vdw_radii.values.max() * 2)[0]:
                radius_j = self.vdw_radii.at[mol2[j].symbol, "vdw_radius"]    
                cutoff = radius_i + radius_j + self.radius_probe

                for test_point in local_sphere_points:
                    is_accessible = True

                    dist = np.linalg.norm(mol2.positions[j] - test_point)
                    
                    #print("test_point:", test_point, dist)
                    if dist < radius_j:
                        n_inaccessible_points += 1
                        self.inaccessible_points = np.vstack((self.inaccessible_points, test_point))
                        is_accessible = False

        #print("n_inaccessible_points:", n_inaccessible_points)
        #print("contact_area:", n_inaccessible_points * self.area_per_point)
        return n_inaccessible_points * self.area_per_point
        
    def writeConnolly(self, fname):
        atom_types = list(["He"]*self.inaccessible_points.shape[0])# + list(self.atoms)
        ConnollySurface = Atoms(atom_types, self.inaccessible_points)
        ConnollySurface.write(fname)
        
    def __init__(self, radii_csv):
        self.vdw_radii = pandas.read_csv(radii_csv, index_col=0)
        self.radius_probe = 1.4
        
if __name__ == "__main__":
    import MDAnalysis as mda
    code = "6M0J"
    U = mda.Universe(f"../MD/{code}/{code}_psfgen.psf", f"../MD/{code}/Minimization.coor")
    U.trajectory[-1]
    Spike = U.select_atoms(f"segid E")
    Receptor = U.select_atoms(f"segid A")
    surface_contact = contactarea(radii_csv = "Alvarez2013_vdwradii.csv")
    
    elements = [x[0] for x in Spike.names]
    Spike_ase = Atoms(elements, Spike.positions)
    
    elements = [x[0] for x in Receptor.names]
    Receptor_ase = Atoms(elements, Receptor.positions)

    x = surface_contact.calculate(Spike_ase, Receptor_ase)
    surface_contact.writeConnolly(f"../MD/{code}/surface_contact.pdb")
    
    