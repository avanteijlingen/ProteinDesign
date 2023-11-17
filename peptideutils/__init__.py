# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 23:24:20 2019

@author: avtei
"""
import numpy as np
import itertools

global peptideutils_peptoid_letters
peptideutils_letters1 = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
peptideutils_peptoid_letters = ['Na', 'Nab', 'Nd', 'Ne', 'Nf', 'Nfe', 'NfeB4', 'NfeC4', "Nfex", 'Nfn', 'Nfnap', 'Ni', 'Nk', 'Nke', 'Nl', 'Nm', 'NmO', 'Nn', 'Nq', 'Nr', 'Ns', 'Nse', 'Nt', 'Nv', 'Nw', 'Nwe', 'Ny']
peptideutils_letters3 = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HSE', 'ILE', 'LYS', 'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR']

Num2Word = {1:"AminoAcids",
            2:"Di",
            3:"Tri",
            4:"Tetra",
            5:"Penta",
            6:"Hexa",
            7:"Hepta",
            8:"Octa",
            9:"Nona",
            10:"Deca",
            11:"Undeca",
            12:"Dodeca"}

def pep2index(peptide):
    L = len(peptide)
    size = int(20**L)
    solution = 0
    letters_1 = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
    for i in range(1, L+1):
        index = np.where(letters_1 == peptide[i-1])[0][0]
        number = int((size/(20**i)) * index)
        solution += number
    return solution
   
def peptoid2index(peptoid):
    if "-" in peptoid:
        peptoid = peptoid.split("-")
    global peptideutils_peptoid_letters
    L = len(peptoid)
    size = int(len(peptideutils_peptoid_letters)**L)
    solution = 0
    letters_1 = np.array(peptideutils_peptoid_letters)
    for i in range(1, L+1):
        index = np.where(letters_1 == peptoid[i-1])[0][0]
        number = int((size/(len(peptideutils_peptoid_letters)**i)) * index)
        solution += number
    return solution
        
        
def index2pep(index, Length):
    size = 20**Length
    if index >= size:
        print("No peptide this large in the dataset")
        return None
    letters_1 = list("ACDEFGHIKLMNPQRSTVWY")
    step = int(size/20)
    above_size = int(size/20)
    steps = np.array([i*step for i in range(20)])
    solution = []
    letter_i = np.where(steps <= index)[0][-1]
    letter = letters_1[letter_i]
    solution.append(letter)
    while len(solution) < Length:
        index = index%step
        step = int(step/20)
        steps = np.array([i*step for i in range(20)])
        if index < above_size/20 and len(solution) < Length-1:
            letter = "A"
            #print(index <= above_size/20)
        else:
            letter_i = np.where(steps <= index)[0][-1]
            letter = letters_1[letter_i]
            #print(letter_i, letter)
        above_size = int(above_size/20)
        solution.append(letter)
    return "".join(solution)
    
def index2peptoid(index, Length):
    global peptideutils_peptoid_letters
    size = len(peptideutils_peptoid_letters)**Length
    if index >= size:
        print("No peptoid this large in the dataset")
        return None
    letters_1 = list(peptideutils_peptoid_letters)
    step = int(size/len(peptideutils_peptoid_letters))
    above_size = int(size/20)
    steps = np.array([i*step for i in range(len(peptideutils_peptoid_letters))])
    solution = []
    letter_i = np.where(steps <= index)[0][-1]
    letter = letters_1[letter_i]
    solution.append(letter)
    while len(solution) < Length:
        index = index%step
        step = int(step/len(peptideutils_peptoid_letters))
        steps = np.array([i*step for i in range(len(peptideutils_peptoid_letters))])
        if index < above_size/len(peptideutils_peptoid_letters) and len(solution) < Length-1:
            letter = "Na"
            #print(index <= above_size/20)
        else:
            letter_i = np.where(steps <= index)[0][-1]
            letter = letters_1[letter_i]
            #print(letter_i, letter)
        above_size = int(above_size/len(peptideutils_peptoid_letters))
        solution.append(letter)
    return "-".join(solution)

def translate1to3(string):
    global peptideutils_letters1
    global peptideutils_letters3
    code = list(string)
    new_string = ""
    for letter in code:
        index = peptideutils_letters1.index(letter)
        new_string = new_string + peptideutils_letters3[index] + "-"
    new_string = new_string[:-1]
    return new_string

def translate3to1(string):
    global peptideutils_letters1
    global peptideutils_letters3
    code = string.split("-")
    new_string = ""
    for AA in code:
        if AA == "HIS":
            AA = "HSE"
        index = peptideutils_letters3.index(AA)
        new_string = new_string + peptideutils_letters1[index]
    return new_string

def GenerateDatasetIndex(AminoAcids, typ = 1):
    global peptideutils_letters1
    global peptideutils_letters3
    return [''.join(i) for i in itertools.product(peptideutils_letters1, repeat = AminoAcids)]

def charge(string):
    """
    

    Parameters
    ----------
    string : TYPE
        Single letter peptide representation.

    Returns
    -------
    Total charge.

    """
    pos = ["K", "R"]
    neg = ["D", "E"]
    return len([x for x in list(string) if x in pos]) - len([x for x in list(string) if x in neg])

if __name__ == "__main__":
    L = 3
    print(peptoid2index(["Na"]*L))

    print(index2peptoid(0, Length=3))
